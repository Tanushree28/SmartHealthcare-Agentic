"""Feature extraction for the classical baseline (PDF 2.9).

Two feature families, matching the ones de Chazal et al. (2004) used, so a
comparison against their published numbers is meaningful:

**RR-interval features** — timing. Derived from the ``r_peaks`` array Phase 1
already saved, so the raw recordings are never touched again. These carry most
of the signal for the supraventricular (S) class, which is defined by a beat
arriving *early* rather than by an unusual shape.

**Morphology features** — shape. The 300-sample beat mean-pooled down to a
smaller vector.

The first and last beat of every record have no preceding/following interval.
They are **dropped and counted**, never imputed — the same rule Phase 1 applies
to beats whose window crosses a record boundary. Inventing an interval that was
never measured is exactly the kind of quiet fabrication this project avoids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config

#: Column order of the RR feature block. Stable: reordering would silently
#: invalidate any saved feature matrix or fitted model.
RR_FEATURE_NAMES: tuple[str, ...] = (
    "pre_rr",
    "post_rr",
    "local_rr",
    "global_rr",
    "pre_rr_ratio",
    "post_rr_ratio",
)

#: Beats either side of the current one used for the local RR average.
LOCAL_RR_HALF_WINDOW: int = 5

#: 300 samples -> 30 features.
DEFAULT_POOL_FACTOR: int = 10


@dataclass
class FeatureSet:
    """Extracted features plus an account of the beats that were dropped."""

    X: np.ndarray  # (n_kept, n_features) float32
    y: np.ndarray  # (n_kept,) int8
    record: np.ndarray  # (n_kept,) record id per row
    feature_names: tuple[str, ...] = ()
    n_input: int = 0
    n_dropped_edge: int = 0

    @property
    def n_kept(self) -> int:
        return int(self.X.shape[0])

    def reconciles(self) -> bool:
        return self.n_kept + self.n_dropped_edge == self.n_input


def morphology_feature_names(n: int) -> tuple[str, ...]:
    return tuple(f"morph_{i:02d}" for i in range(n))


def pool_morphology(X: np.ndarray, factor: int = DEFAULT_POOL_FACTOR) -> np.ndarray:
    """Mean-pool each beat by an integer factor.

    Averaging rather than subsampling: subsampling every 10th sample would alias
    the QRS complex, which is the sharpest feature in the signal and the one the
    classifier most needs.
    """
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D beat matrix, got shape {arr.shape}")
    if factor < 1:
        raise ValueError(f"pool factor must be >= 1, got {factor}")
    if factor == 1:
        return arr.copy()
    n_beats, length = arr.shape
    if length % factor:
        raise ValueError(
            f"beat length {length} is not divisible by pool factor {factor}; "
            f"a partial trailing bin would mix a different number of samples"
        )
    return arr.reshape(n_beats, length // factor, factor).mean(axis=2)


#: Beats either side of the current one used for the robust rhythm baseline.
#: Wider than LOCAL_RR_HALF_WINDOW because this is meant to describe the
#: patient's underlying rhythm, not the immediate neighbourhood.
BASELINE_RR_HALF_WINDOW: int = 25


def rolling_baseline_rr(
    intervals: np.ndarray,
    half_window: int = BASELINE_RR_HALF_WINDOW,
) -> np.ndarray:
    """A rhythm reference that a run of ectopic beats cannot drag around.

    ``global_rr`` is the mean RR over an entire record, so a single stretch of
    bigeminy shifts the reference for every beat in that patient -- including
    beats nowhere near the episode. Prematurity is relative to the rhythm
    *around* a beat, so the reference should be local and resistant to outliers.

    A rolling median is both. On a steady rhythm it equals the mean, so this
    changes nothing where nothing is wrong.
    """
    values = np.asarray(intervals, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(f"intervals must be 1-D, got shape {values.shape}")
    if half_window < 0:
        raise ValueError(f"half_window must be >= 0, got {half_window}")
    if values.size == 0:
        return np.empty(0, dtype=np.float64)

    out = np.empty(values.size, dtype=np.float64)
    for i in range(values.size):
        lo = max(i - half_window, 0)
        hi = min(i + half_window + 1, values.size)
        out[i] = np.median(values[lo:hi])
    return out


def rr_features(
    r_peaks: np.ndarray,
    fs: int = config.FS,
    half_window: int = LOCAL_RR_HALF_WINDOW,
    robust: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """RR-interval features for every beat that has a neighbour on both sides.

    Returns ``(features, keep_index)`` where ``keep_index`` gives the positions
    in the input that survived. Intervals are in seconds.
    """
    peaks = np.asarray(r_peaks, dtype=np.int64)
    n = peaks.size

    if n < 3:
        # Fewer than three beats leaves nothing with both neighbours.
        return (
            np.empty((0, len(RR_FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    if np.any(np.diff(peaks) <= 0):
        raise ValueError(
            "r_peaks must be strictly increasing; got a non-monotonic or "
            "duplicated sample index, which would produce a negative interval"
        )

    intervals = np.diff(peaks) / float(fs)  # length n-1
    keep = np.arange(1, n - 1, dtype=np.int64)

    pre = intervals[keep - 1]
    post = intervals[keep]

    # Local average over the surrounding beats, clipped at the record edges.
    local = np.empty(keep.size, dtype=np.float64)
    for out_i, beat_i in enumerate(keep):
        lo = max(beat_i - half_window - 1, 0)
        hi = min(beat_i + half_window, intervals.size)
        local[out_i] = intervals[lo:hi].mean()

    # The rhythm a beat is judged "early" against. Computed from this record's
    # own beats only, so it carries no information from any other record or
    # split.
    #
    # robust=False keeps the Phase 1b definition -- the record-level mean -- so
    # published results stay reproducible. robust=True uses a rolling median,
    # which an arrhythmic episode cannot drag around; see rolling_baseline_rr.
    if robust:
        baseline = rolling_baseline_rr(intervals)[keep - 1]
    else:
        baseline = np.full(keep.size, float(intervals.mean()))

    features = np.column_stack(
        [
            pre,
            post,
            local,
            baseline,
            pre / baseline,
            post / baseline,
        ]
    ).astype(np.float32)

    return features, keep


def extract_record(
    X: np.ndarray,
    y: np.ndarray,
    r_peaks: np.ndarray,
    record: str,
    fs: int = config.FS,
    pool_factor: int = DEFAULT_POOL_FACTOR,
    use_rr: bool = True,
    use_morphology: bool = True,
) -> FeatureSet:
    """Build the feature matrix for one record's Phase 1 output."""
    if not (use_rr or use_morphology):
        raise ValueError("at least one of use_rr / use_morphology must be True")

    beats = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int8)
    peaks = np.asarray(r_peaks, dtype=np.int64)

    if not (beats.shape[0] == labels.size == peaks.size):
        raise ValueError(
            f"record {record}: X/y/r_peaks disagree "
            f"({beats.shape[0]}, {labels.size}, {peaks.size})"
        )

    n_input = int(labels.size)

    if use_rr:
        rr, keep = rr_features(peaks, fs=fs)
    else:
        # Without RR features no beat needs a neighbour, so nothing is dropped.
        keep = np.arange(n_input, dtype=np.int64)
        rr = np.empty((keep.size, 0), dtype=np.float32)

    blocks: list[np.ndarray] = []
    names: list[str] = []

    if use_rr:
        blocks.append(rr)
        names.extend(RR_FEATURE_NAMES)

    if use_morphology:
        pooled = pool_morphology(beats, pool_factor)[keep]
        blocks.append(pooled)
        names.extend(morphology_feature_names(pooled.shape[1]))

    matrix = (
        np.concatenate(blocks, axis=1).astype(np.float32)
        if blocks
        else np.empty((keep.size, 0), dtype=np.float32)
    )

    return FeatureSet(
        X=matrix,
        y=labels[keep],
        record=np.full(keep.size, record, dtype=object),
        feature_names=tuple(names),
        n_input=n_input,
        n_dropped_edge=n_input - int(keep.size),
    )


def extract_records(
    beats_dir,
    records,
    fs: int = config.FS,
    pool_factor: int = DEFAULT_POOL_FACTOR,
    use_rr: bool = True,
    use_morphology: bool = True,
) -> FeatureSet:
    """Concatenate features across records, preserving record provenance.

    The ``record`` column is what makes it possible to check after the fact that
    no beat crossed a split boundary.
    """
    from pathlib import Path

    beats_dir = Path(beats_dir)
    parts: list[FeatureSet] = []

    for name in records:
        path = beats_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"no beats file at {path}; run scripts/preprocess_mitbih.py first"
            )
        with np.load(path) as data:
            parts.append(
                extract_record(
                    data["X"], data["y"], data["r_peaks"], str(name),
                    fs=fs, pool_factor=pool_factor,
                    use_rr=use_rr, use_morphology=use_morphology,
                )
            )

    if not parts:
        raise ValueError("no records given")

    names = parts[0].feature_names
    return FeatureSet(
        X=np.concatenate([p.X for p in parts], axis=0),
        y=np.concatenate([p.y for p in parts], axis=0),
        record=np.concatenate([p.record for p in parts], axis=0),
        feature_names=names,
        n_input=sum(p.n_input for p in parts),
        n_dropped_edge=sum(p.n_dropped_edge for p in parts),
    )
