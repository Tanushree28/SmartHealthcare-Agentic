"""Pipeline orchestration and manifest generation (PDF 2.1-2.9).

Per-record output rather than one monolithic array. That keeps peak memory
bounded, makes record-wise splitting a file-selection operation, and lets a
single failed record be reprocessed on its own without redoing the other 47.

Order of operations per record: load -> validate -> filter -> segment -> label
-> normalize -> write. Normalization is applied **last**, using statistics
fitted on training records only, so no test-set amplitude ever influences the
scaling of a training beat.
"""

from __future__ import annotations

import json
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, filters, io, labels, normalize, segment, splits

MANIFEST_COLUMNS = [
    "record",
    "split",
    "status",
    "lead",
    "fs",
    "n_samples",
    "duration_s",
    "n_annotations",
    "n_beats_kept",
    "n_dropped_boundary",
    "n_excluded_nonbeat",
    "n_unknown",
    *[f"n_class_{c}" for c in config.AAMI_CLASSES],
    "drop_reasons",
    "error",
]


@dataclass
class RecordResult:
    """Everything the manifest needs about one record, success or failure."""

    record: str
    status: str  # "ok" | "failed"
    split: str | None = None
    lead: str | None = None
    fs: int | None = None
    n_samples: int = 0
    duration_s: float = 0.0
    beats: segment.BeatSegments | None = None
    error: str | None = None

    def manifest_row(self) -> dict:
        beats = self.beats
        counts = beats.class_counts if beats else Counter()
        row = {
            "record": self.record,
            "split": self.split or "",
            "status": self.status,
            "lead": self.lead or "",
            "fs": self.fs or "",
            "n_samples": self.n_samples,
            "duration_s": round(self.duration_s, 3),
            "n_annotations": beats.n_total if beats else 0,
            "n_beats_kept": beats.n_kept if beats else 0,
            "n_dropped_boundary": beats.n_dropped_boundary if beats else 0,
            "n_excluded_nonbeat": beats.n_excluded_nonbeat if beats else 0,
            "n_unknown": beats.n_unknown if beats else 0,
            "drop_reasons": json.dumps(dict(sorted(beats.drop_reasons.items())))
            if beats
            else "{}",
            "error": self.error or "",
        }
        for cls in config.AAMI_CLASSES:
            row[f"n_class_{cls}"] = int(counts.get(cls, 0))
        return row


@dataclass
class PipelineResult:
    """Aggregate outcome of a full run."""

    records: list[RecordResult] = field(default_factory=list)
    split: splits.SplitAssignment | None = None
    stats: normalize.NormStats | None = None
    output_dir: Path | None = None

    @property
    def ok(self) -> list[RecordResult]:
        return [r for r in self.records if r.status == "ok"]

    @property
    def failed(self) -> list[RecordResult]:
        return [r for r in self.records if r.status == "failed"]

    @property
    def exit_code(self) -> int:
        """Non-zero when any record failed -- partial success is not success."""
        if not self.records:
            return 1
        return 1 if self.failed else 0

    def total_beats(self, split: str | None = None) -> int:
        return sum(
            r.beats.n_kept
            for r in self.ok
            if r.beats and (split is None or r.split == split)
        )

    def class_totals(self, split: str | None = None) -> Counter:
        """Per-class beat counts, optionally restricted to one split.

        ``split=None`` covers every successfully processed record, including
        the excluded paced ones. Pass ``"train"`` for anything that feeds a
        training decision -- class weights computed over held-out beats are
        leakage, however mild.
        """
        total: Counter = Counter()
        for record in self.ok:
            if split is not None and record.split != split:
                continue
            if record.beats:
                total.update(record.beats.class_counts)
        return total


def process_record(
    record_dir: Path | str,
    name: str,
    expected_fs: int | None = config.FS,
    use_notch: bool = config.NOTCH_ENABLED_BY_DEFAULT,
    strict_symbols: bool = False,
) -> RecordResult:
    """Load, denoise and segment one record. Normalization happens later."""
    record = io.load_record(record_dir, name, expected_fs=expected_fs)
    annotations = io.load_annotations(record_dir, name)

    filtered = filters.denoise(record.signal, fs=record.fs, use_notch=use_notch)

    mappings = labels.map_symbols(annotations.symbols, strict=strict_symbols)
    beats = segment.segment_beats(filtered, annotations.samples, mappings)

    if not beats.reconciles():
        raise AssertionError(
            f"manifest arithmetic does not reconcile for record {name}: "
            f"{beats.n_kept} + {beats.n_dropped_boundary} + "
            f"{beats.n_excluded_nonbeat} + {beats.n_unknown} != {beats.n_total}"
        )

    return RecordResult(
        record=name,
        status="ok",
        lead=record.lead,
        fs=record.fs,
        n_samples=record.n_samples,
        duration_s=record.duration_s,
        beats=beats,
    )


def run(
    record_dir: Path | str,
    output_dir: Path | str,
    records=None,
    strategy: str = "record_wise",
    seed: int = config.DEFAULT_SPLIT_SEED,
    exclude_paced: bool = True,
    expected_fs: int | None = config.FS,
    use_notch: bool = config.NOTCH_ENABLED_BY_DEFAULT,
    strict_symbols: bool = False,
) -> PipelineResult:
    """Process every record and write beats, manifest, splits and statistics.

    A single unreadable record is recorded as a failure and the run continues --
    a 48-record job should not lose 47 records' work to one bad file. The exit
    code still reflects the partial failure.
    """
    record_dir = Path(record_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    names = sorted({str(r) for r in (records if records is not None
                                     else io.available_records(record_dir))})
    if not names:
        raise io.DataRootError(
            f"no WFDB records found in {record_dir}. Set "
            f"${config.DATA_ROOT_ENV_VAR} or run scripts/download_mitbih.py"
        )

    if strategy == "ds1_ds2":
        assignment = splits.ds1_ds2_split(records=names, seed=seed)
    elif strategy == "record_wise":
        assignment = splits.record_wise_split(
            names, seed=seed, exclude_paced=exclude_paced
        )
    else:
        raise ValueError(
            f"unknown split strategy {strategy!r}; expected 'record_wise' or "
            f"'ds1_ds2'. Beat-level splitting is deliberately not available."
        )

    result = PipelineResult(split=assignment, output_dir=output_dir)
    processed: dict[str, RecordResult] = {}

    for name in names:
        try:
            record_result = process_record(
                record_dir,
                name,
                expected_fs=expected_fs,
                use_notch=use_notch,
                strict_symbols=strict_symbols,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the run
            record_result = RecordResult(
                record=name,
                status="failed",
                error=f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500],
            )
            record_result._traceback = traceback.format_exc()  # type: ignore[attr-defined]
        else:
            processed[name] = record_result

        record_result.split = assignment.split_of(name) or "excluded"
        result.records.append(record_result)

    # --- Normalization: fitted on training records only (PDF 3.9) -------------
    train_arrays = [
        processed[n].beats.X
        for n in assignment.train
        if n in processed and processed[n].beats and processed[n].beats.n_kept
    ]
    result.stats = normalize.fit(train_arrays) if train_arrays else None

    # --- Write outputs -------------------------------------------------------
    beats_dir = output_dir / "beats"
    beats_dir.mkdir(parents=True, exist_ok=True)

    for name, record_result in processed.items():
        beats = record_result.beats
        assert beats is not None
        X = (
            normalize.apply(beats.X, result.stats)
            if result.stats is not None and beats.n_kept
            else beats.X.astype(np.float32)
        )
        np.savez(
            beats_dir / f"{name}.npz",
            X=X,
            y=beats.y,
            r_peaks=beats.r_peaks,
        )

    write_manifest(result, output_dir / "manifest.csv")
    assignment.write_json(output_dir / "splits.json")
    write_normalization(result, output_dir / "normalization.json")

    return result


def write_manifest(result: PipelineResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.manifest_row() for r in result.records]
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def write_normalization(result: PipelineResult, path: Path) -> Path:
    """Record the statistics *and* which records they came from.

    Naming the source records is what makes the no-leakage claim auditable after
    the fact rather than a promise in a docstring.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_totals = result.class_totals("train")
    all_totals = result.class_totals()
    payload = {
        "method": "zscore",
        "fitted_on": "train_records_only",
        "train_records": list(result.split.train) if result.split else [],
        "stats": result.stats.to_dict() if result.stats else None,
        "aami_class_order": list(config.AAMI_CLASSES),
        "train_class_counts": {c: int(train_totals.get(c, 0))
                               for c in config.AAMI_CLASSES},
        "all_processed_class_counts": {c: int(all_totals.get(c, 0))
                                       for c in config.AAMI_CLASSES},
        "suggested_class_weights": segment.suggested_class_weights(train_totals),
        "note": (
            "Class weights are reported, not applied. Resampling is a training "
            "decision; applying it here would destroy the true class "
            "distribution and make honest evaluation impossible (PDF 2.7)."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
