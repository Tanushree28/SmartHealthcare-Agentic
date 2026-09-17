"""Pair each beat waveform with its RR timing features.

The hybrid network of Phase 1d takes two inputs per beat: the 300-sample
waveform the Phase 1 pipeline saved, and the 6 RR features describing when the
beat arrived. This module produces them aligned.

**Boundary beats are dropped.** RR features need a neighbour on both sides, so
the first and last beat of every record has none. Dropping them keeps the
denominator identical to the Phase 1b ``rr+morphology`` rows this study is
compared against -- 44 beats, 0.09% of DS2. See the spec, section 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, features


@dataclass(frozen=True)
class BeatData:
    """Beats, their RR features, labels, and the record each came from."""

    X: np.ndarray  # (n, BEAT_LENGTH) float32 -- normalized waveforms
    rr: np.ndarray  # (n, 6) float32 -- features.RR_FEATURE_NAMES order
    y: np.ndarray  # (n,) int64 -- AAMI class indices
    record: np.ndarray  # (n,) object -- record id per beat
    n_input: int  # beats before boundary dropping
    n_dropped_edge: int

    @property
    def n_kept(self) -> int:
        return int(self.y.size)

    def reconciles(self) -> bool:
        """Every input beat is either kept or accounted for as dropped."""
        return self.n_kept + self.n_dropped_edge == self.n_input


def load_record(
    beats_dir,
    record: str,
    fs: int = config.FS,
    robust_rr: bool = False,
) -> BeatData:
    """Load one record's ``.npz`` and attach RR features.

    ``robust_rr`` swaps the record-level mean rhythm reference for a rolling
    median -- see ``features.rolling_baseline_rr``. Off by default so Phase 1b
    and 1d results stay reproducible.
    """
    path = _require(beats_dir, record)

    with np.load(path) as handle:
        X = handle["X"].astype(np.float32)
        # int8 on disk; torch's cross-entropy requires int64 targets.
        y = handle["y"].astype(np.int64)
        r_peaks = handle["r_peaks"].astype(np.int64)

    if X.ndim != 2 or X.shape[1] != config.BEAT_LENGTH:
        found = X.shape[1] if X.ndim == 2 else f"array of shape {X.shape}"
        raise ValueError(
            f"record {record}: beat length {found} != expected "
            f"{config.BEAT_LENGTH}; the convolution stack assumes the "
            f"configured length"
        )
    if not (X.shape[0] == y.size == r_peaks.size):
        raise ValueError(
            f"record {record}: X/y/r_peaks disagree "
            f"({X.shape[0]}, {y.size}, {r_peaks.size})"
        )

    rr, keep = features.rr_features(r_peaks, fs=fs, robust=robust_rr)

    return BeatData(
        X=X[keep],
        rr=rr,
        y=y[keep],
        record=np.full(keep.size, record, dtype=object),
        n_input=int(y.size),
        n_dropped_edge=int(y.size - keep.size),
    )


def load_records(
    beats_dir,
    records,
    fs: int = config.FS,
    robust_rr: bool = False,
) -> BeatData:
    """Load several records and concatenate them, in the order given."""
    names = [str(r) for r in records]

    if not names:
        raise ValueError("no records given")

    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    duplicates = sorted(n for n, c in counts.items() if c > 1)
    if duplicates:
        raise ValueError(
            f"duplicate record ids {duplicates}: the same beats would be "
            f"counted twice, which reads as leakage downstream"
        )

    parts = [
        load_record(beats_dir, name, fs=fs, robust_rr=robust_rr)
        for name in names
    ]

    return BeatData(
        X=np.concatenate([p.X for p in parts]),
        rr=np.concatenate([p.rr for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        record=np.concatenate([p.record for p in parts]),
        n_input=sum(p.n_input for p in parts),
        n_dropped_edge=sum(p.n_dropped_edge for p in parts),
    )


def class_counts(data: BeatData) -> np.ndarray:
    """Beats per AAMI class, including classes with none."""
    return np.bincount(data.y, minlength=len(config.AAMI_CLASSES)).astype(np.int64)


def _require(beats_dir, record: str) -> Path:
    path = Path(beats_dir) / f"{record}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no beats for record {record} at {path}. Run: "
            f"./venv/bin/python scripts/preprocess_mitbih.py --strategy ds1_ds2 "
            f"--out-dir ~/physionet-data/processed/mitdb-ds12"
        )
    return path
