"""Record and annotation loading (PDF 2.1).

The one decision worth calling out: **the lead is selected by signal name, not
by column index.** Most MIT-BIH records carry MLII in channel 0, but not all do.
Indexing blindly would silently mix different leads into one training set -- a
real data-quality bug that no downstream metric would reveal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from . import config
from .errors import DataRootError, LeadNotFoundError, SamplingRateError


@dataclass(frozen=True)
class LoadedRecord:
    """One record's chosen lead plus the metadata needed to audit it."""

    name: str
    signal: np.ndarray
    lead: str
    fs: int
    available_leads: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.signal.size)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


@dataclass(frozen=True)
class LoadedAnnotations:
    """Annotation samples and symbols, in file order."""

    name: str
    samples: np.ndarray
    symbols: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.samples.size)


def _record_path(record_dir: Path | str, name: str) -> Path:
    directory = Path(record_dir).expanduser()
    header = directory / f"{name}.hea"
    if not header.exists():
        raise DataRootError(
            f"no WFDB header at {header}. Check that the MIT-BIH database has "
            f"been downloaded and that ${config.DATA_ROOT_ENV_VAR} points at the "
            f"data root (currently resolving to {config.data_root()}). "
            f"Run: python scripts/download_mitbih.py"
        )
    return directory / name


def select_lead(
    sig_names: list[str] | tuple[str, ...],
    preference: tuple[str, ...] = config.LEAD_PREFERENCE,
) -> tuple[int, str]:
    """Return ``(channel_index, lead_name)`` for the first preferred lead present."""
    normalized = [str(n).strip() for n in sig_names]
    for wanted in preference:
        for idx, actual in enumerate(normalized):
            if actual.upper() == wanted.upper():
                return idx, actual
    raise LeadNotFoundError(
        f"none of the preferred leads {list(preference)} is present; "
        f"record carries {normalized}"
    )


def load_record(
    record_dir: Path | str,
    name: str,
    expected_fs: int | None = config.FS,
    preference: tuple[str, ...] = config.LEAD_PREFERENCE,
) -> LoadedRecord:
    """Load one record and return its preferred lead as a 1-D float array.

    A sampling-rate mismatch is rejected rather than filtered with coefficients
    designed for a different rate -- that would produce a plausible-looking but
    silently wrong signal.
    """
    path = _record_path(record_dir, name)
    record = wfdb.rdrecord(str(path))

    fs = int(round(float(record.fs)))
    if expected_fs is not None and fs != expected_fs:
        raise SamplingRateError(
            f"record {name!r} is sampled at {fs} Hz but the pipeline is "
            f"configured for {expected_fs} Hz; filter coefficients and window "
            f"sizes would not transfer"
        )

    sig_names = list(record.sig_name or [])
    channel, lead = select_lead(sig_names, preference)

    signal = np.asarray(record.p_signal[:, channel], dtype=np.float64)

    return LoadedRecord(
        name=name,
        signal=signal,
        lead=lead,
        fs=fs,
        available_leads=tuple(sig_names),
    )


def load_annotations(
    record_dir: Path | str,
    name: str,
    extension: str = "atr",
) -> LoadedAnnotations:
    """Load the beat/rhythm annotations for one record."""
    path = _record_path(record_dir, name)
    ann_file = path.with_suffix(f".{extension}")
    if not ann_file.exists():
        raise DataRootError(
            f"no annotation file at {ann_file}; the record was downloaded "
            f"without its .{extension} companion"
        )

    ann = wfdb.rdann(str(path), extension)
    return LoadedAnnotations(
        name=name,
        samples=np.asarray(ann.sample, dtype=np.int64),
        symbols=tuple(str(s) for s in ann.symbol),
    )


def available_records(record_dir: Path | str) -> list[str]:
    """Record ids present on disk, sorted. Empty list if the directory is absent."""
    directory = Path(record_dir).expanduser()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.hea"))
