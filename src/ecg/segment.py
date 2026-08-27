"""Beat-level and window-level segmentation (PDF 2.4, 2.5).

Beat segmentation slices a fixed window around each annotated R-peak. Two
properties are contractual:

1. The R-peak sits at index ``SAMPLES_BEFORE`` in **every** row. Downstream
   models rely on that alignment, which is why :mod:`ecg.filters` is zero-phase.
2. A beat whose window would run past a record boundary is **dropped and
   counted**, never zero-padded. Padding would invent a flat pre-QRS segment
   that a model will happily learn from.

R-peaks come from the annotations directly, which PDF 2.4 sanctions. An
independent QRS detector is deferred, not silently skipped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class BeatSegments:
    """Segmented beats plus a full account of what did not make it in."""

    X: np.ndarray  # (n_kept, BEAT_LENGTH) float32
    y: np.ndarray  # (n_kept,) int8 -- AAMI class index
    r_peaks: np.ndarray  # (n_kept,) int64 -- original sample index
    symbols: tuple[str, ...] = ()
    n_total: int = 0
    n_kept: int = 0
    n_dropped_boundary: int = 0
    n_excluded_nonbeat: int = 0
    n_unknown: int = 0
    drop_reasons: Counter = field(default_factory=Counter)
    class_counts: Counter = field(default_factory=Counter)

    def reconciles(self) -> bool:
        """Every annotation is accounted for by exactly one outcome."""
        return (
            self.n_kept
            + self.n_dropped_boundary
            + self.n_excluded_nonbeat
            + self.n_unknown
            == self.n_total
        )


def empty_beats() -> BeatSegments:
    """A well-formed empty result -- the zero-annotation case must not raise."""
    return BeatSegments(
        X=np.empty((0, config.BEAT_LENGTH), dtype=np.float32),
        y=np.empty((0,), dtype=np.int8),
        r_peaks=np.empty((0,), dtype=np.int64),
    )


def segment_beats(
    signal: np.ndarray,
    ann_samples,
    mappings,
    before: int = config.SAMPLES_BEFORE,
    after: int = config.SAMPLES_AFTER,
) -> BeatSegments:
    """Slice ``[r - before, r + after)`` around each annotated beat.

    ``mappings`` is a sequence of :class:`ecg.labels.SymbolMapping`, one per
    annotation, in the same order as ``ann_samples``.
    """
    sig = np.asarray(signal, dtype=np.float64)
    samples = np.asarray(ann_samples, dtype=np.int64)

    if samples.size != len(mappings):
        raise ValueError(
            f"annotation/mapping length mismatch: {samples.size} samples vs "
            f"{len(mappings)} mappings"
        )

    result = empty_beats()
    result.n_total = int(samples.size)
    if result.n_total == 0:
        return result

    length = before + after
    n_samples = sig.size

    rows: list[np.ndarray] = []
    labels: list[int] = []
    peaks: list[int] = []
    symbols: list[str] = []

    for sample, mapping in zip(samples, mappings):
        if mapping.outcome == "excluded":
            result.n_excluded_nonbeat += 1
            result.drop_reasons[f"nonbeat:{mapping.symbol}"] += 1
            continue
        if mapping.outcome == "unknown":
            result.n_unknown += 1
            result.drop_reasons[f"unknown:{mapping.symbol}"] += 1
            continue

        start = int(sample) - before
        stop = int(sample) + after

        if start < 0:
            result.n_dropped_boundary += 1
            result.drop_reasons["boundary:start"] += 1
            continue
        if stop > n_samples:
            result.n_dropped_boundary += 1
            result.drop_reasons["boundary:end"] += 1
            continue

        window = sig[start:stop]
        if window.size != length:  # defensive: cannot happen given the guards
            result.n_dropped_boundary += 1
            result.drop_reasons["boundary:short"] += 1
            continue

        rows.append(window)
        labels.append(int(mapping.class_index))
        peaks.append(int(sample))
        symbols.append(mapping.symbol)
        result.class_counts[mapping.aami_class] += 1

    if rows:
        result.X = np.asarray(rows, dtype=np.float32)
        result.y = np.asarray(labels, dtype=np.int8)
        result.r_peaks = np.asarray(peaks, dtype=np.int64)
        result.symbols = tuple(symbols)

    result.n_kept = len(rows)
    return result


def segment_windows(
    signal: np.ndarray,
    window_seconds: float,
    fs: int = config.FS,
    drop_partial: bool = True,
) -> np.ndarray:
    """Split a signal into fixed-duration windows for later rhythm models.

    The trailing partial window is dropped by default. Keeping it would require
    padding, and a short zero-padded tail is not a rhythm sample.
    """
    sig = np.asarray(signal, dtype=np.float32)
    width = int(round(window_seconds * fs))
    if width <= 0:
        raise ValueError(f"window_seconds={window_seconds} yields a width of {width}")

    n_full = sig.size // width
    if not drop_partial and sig.size % width:
        raise ValueError(
            "drop_partial=False is not supported: the trailing window would "
            "need zero-padding, which invents signal that was never recorded"
        )
    if n_full == 0:
        return np.empty((0, width), dtype=np.float32)

    return sig[: n_full * width].reshape(n_full, width).copy()


def suggested_class_weights(class_counts) -> dict[str, float]:
    """Inverse-frequency weights for the classes actually present (PDF 2.7).

    Reported, not applied. Resampling is a *training* decision that belongs next
    to the loss function; baking it into preprocessing would destroy the true
    class distribution and make honest evaluation impossible.
    """
    counts = {k: int(v) for k, v in dict(class_counts).items() if int(v) > 0}
    if not counts:
        return {}
    total = sum(counts.values())
    n_classes = len(counts)
    return {name: total / (n_classes * n) for name, n in counts.items()}
