"""One record's lead and beat labels; download and checks; patient grouping.

Loading reuses ``mit_bih.io`` (lead chosen by name, sampling rate checked) and
``mit_bih.labels`` in strict mode: an annotation symbol in neither the beat
table nor the non-beat table stops the run rather than being guessed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from mit_bih import acquire
from mit_bih import io as mit_io
from mit_bih import labels
from mit_bih.errors import LeadNotFoundError

from . import config


@dataclass(frozen=True)
class LabelledRecord:
    """One record's chosen lead and its beat annotations, at the record's rate."""

    name: str
    signal: np.ndarray  # chosen lead, mV, float64
    fs: int
    lead: str
    beat_samples: np.ndarray  # int64, beat annotations only, file order
    beat_labels: np.ndarray  # int64, index into mit_bih.config.AAMI_CLASSES
    symbol_counts: dict[str, int]  # every annotation symbol -> count
    n_excluded: int  # non-beat annotations

    @property
    def n_beats(self) -> int:
        return int(self.beat_samples.size)

    @property
    def has_paced(self) -> bool:
        return any(self.symbol_counts.get(s, 0) > 0 for s in config.PACED_SYMBOLS)


def label_beats(samples, symbols) -> tuple[np.ndarray, np.ndarray, int]:
    """``(beat_samples, beat_labels, n_excluded)``; unknown symbols raise."""
    mappings = labels.map_symbols(symbols, strict=True)
    is_beat = np.array([m.is_beat for m in mappings], dtype=bool)
    beat_samples = np.asarray(samples, dtype=np.int64)[is_beat] if is_beat.size else \
        np.zeros(0, dtype=np.int64)
    beat_labels = np.array([m.class_index for m in mappings if m.is_beat], dtype=np.int64)
    return beat_samples, beat_labels, int(is_beat.size - np.count_nonzero(is_beat))


def load_labelled_record(
    record_dir: Path | str,
    name: str,
    fs: int = config.FS,
    preference: tuple[str, ...] = (config.LEAD,),
) -> LabelledRecord:
    """Load one record's preferred lead and its labelled beats.

    INCART: the defaults (257 Hz, lead II). DS2: ``fs=360`` and
    ``preference=mit_bih.config.LEAD_PREFERENCE``.
    """
    rec = mit_io.load_record(record_dir, name, expected_fs=fs, preference=preference)
    ann = mit_io.load_annotations(record_dir, name)
    beat_samples, beat_labels, n_excluded = label_beats(ann.samples, ann.symbols)
    return LabelledRecord(
        name=name,
        signal=rec.signal,
        fs=rec.fs,
        lead=rec.lead,
        beat_samples=beat_samples,
        beat_labels=beat_labels,
        symbol_counts=dict(Counter(ann.symbols)),
        n_excluded=n_excluded,
    )


# ---------------------------------------------------------------------------
# Download and on-disk checks
# ---------------------------------------------------------------------------


def check_record(directory: Path | str, name: str) -> acquire.RecordCheck:
    """Files present, 257 Hz, ``.dat`` not truncated, lead II recorded in mV."""
    directory = Path(directory).expanduser()
    check = acquire.check_record(directory, name, expected_fs=config.FS)
    if not check.ok:
        return check
    header = wfdb.rdheader(str(directory / name))
    try:
        channel, _ = mit_io.select_lead(list(header.sig_name or []), (config.LEAD,))
    except LeadNotFoundError:
        return acquire.RecordCheck(name, False, f"no lead {config.LEAD} in the header")
    units = (header.units or [None] * (channel + 1))[channel]
    if units != "mV":
        return acquire.RecordCheck(name, False, f"lead {config.LEAD} is in {units!r}, expected 'mV'")
    return check


def verify(directory: Path | str, records=config.RECORDS) -> list[acquire.RecordCheck]:
    """Check every expected record. Never touches the network."""
    return [check_record(directory, str(r)) for r in records]


def missing_records(directory: Path | str, records=config.RECORDS) -> list[str]:
    """Records that are absent or fail their check."""
    return [c.record for c in verify(directory, records) if not c.ok]


def download(directory: Path | str | None = None, records=config.RECORDS,
             force: bool = False) -> list[acquire.RecordCheck]:
    """Fetch missing INCART records from PhysioNet, then check everything.

    Idempotent: records that already pass their check are not fetched again.
    Raises if any record still fails afterwards, so a half-complete download
    never looks like success.
    """
    directory = Path(directory or config.raw_dir()).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    wanted = [str(r) for r in records]
    todo = wanted if force else missing_records(directory, wanted)
    if todo:
        wfdb.dl_database(config.DATABASE, dl_dir=str(directory), records=todo)
    checks = verify(directory, wanted)
    bad = [c for c in checks if not c.ok]
    if bad:
        raise RuntimeError("download finished but these records failed their check:\n  "
                           + "\n  ".join(str(c) for c in bad))
    return checks


def fetch_patient_file(directory: Path | str) -> bool:
    """Make sure the patient documentation file is on disk; True if it is.

    Already present: no network call. Any download failure gives False, and
    the results are then grouped by record (see ``build_patient_map``).
    """
    directory = Path(directory).expanduser()
    target = directory / config.PATIENT_FILE
    if target.is_file():
        return True
    try:
        wfdb.dl_files(config.DATABASE, str(directory), [config.PATIENT_FILE])
    except Exception:  # noqa: BLE001 -- any failure means "not available"
        return False
    return target.is_file() and target.stat().st_size > 0


# ---------------------------------------------------------------------------
# Record -> patient
# ---------------------------------------------------------------------------

_RECORD_ID = re.compile(r"\bI\d{2}\b")


@dataclass(frozen=True)
class PatientMap:
    """Which group each record's beats are resampled with in the bootstrap."""

    patient_of: dict[str, str]  # record -> group id
    by_patient: bool  # False: each record is its own group
    note: str

    @property
    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for record, group in self.patient_of.items():
            out.setdefault(group, []).append(record)
        return out


def patient_groups_from_text(text: str) -> list[tuple[str, ...]]:
    """Record ids on each line that has any; one tuple per such line."""
    groups = []
    for line in text.splitlines():
        ids = _RECORD_ID.findall(line)
        if ids:
            groups.append(tuple(ids))
    return groups


def _fallback(records, reason: str) -> PatientMap:
    return PatientMap(
        patient_of={r: r for r in records},
        by_patient=False,
        note=(f"Resampled by record: no valid record-to-patient mapping ({reason}). "
              "This understates the uncertainty, since some records share a patient."),
    )


def build_patient_map(text: str | None, records=config.RECORDS,
                      expected_patients: int = config.N_PATIENTS) -> PatientMap:
    """Group records by patient, or fall back to one group per record.

    The grouping is used only if every record appears exactly once and the
    number of groups equals the published patient count. Diagnosis codes that
    look like record ids (ICD-10 ``I10``, ``I25``) break the first condition,
    so a misread file falls back instead of producing a wrong grouping.
    """
    records = tuple(str(r) for r in records)
    if text is None:
        return _fallback(records, f"{config.PATIENT_FILE} not available")

    groups = patient_groups_from_text(text)
    found = Counter(r for g in groups for r in g)
    repeated = sorted(r for r, n in found.items() if n > 1)
    missing = sorted(set(records) - set(found))
    unexpected = sorted(set(found) - set(records))
    if repeated:
        return _fallback(records, f"record ids appear more than once: {', '.join(repeated)}")
    if missing:
        return _fallback(records, f"records not in the file: {', '.join(missing)}")
    if unexpected:
        return _fallback(records, f"unexpected record ids: {', '.join(unexpected)}")
    if len(groups) != expected_patients:
        return _fallback(records, f"found {len(groups)} groups, expected {expected_patients} patients")

    patient_of = {r: f"P{i:02d}" for i, g in enumerate(groups, start=1) for r in g}
    return PatientMap(
        patient_of={r: patient_of[r] for r in records},
        by_patient=True,
        note=f"Resampled by patient: {len(groups)} patients, from {config.PATIENT_FILE}.",
    )


def load_patient_map(directory: Path | str, records=config.RECORDS,
                     expected_patients: int = config.N_PATIENTS) -> PatientMap:
    """``build_patient_map`` on the patient file in ``directory``, if present."""
    path = Path(directory).expanduser() / config.PATIENT_FILE
    text = path.read_text(errors="replace") if path.is_file() else None
    return build_patient_map(text, records, expected_patients)
