"""Amplitude normalization (PDF 2.3, 3.9).

Split into ``fit`` and ``apply`` on purpose. Statistics must be computed from
training records only; a single ``normalize(all_data)`` call would leak the
validation and test distributions into training and inflate every downstream
number. Keeping the two halves separate makes that leak hard to write by
accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .errors import InvalidSignalError


@dataclass(frozen=True)
class NormStats:
    """Z-score statistics, fitted once and reused everywhere."""

    mean: float
    std: float

    def to_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "NormStats":
        return cls(mean=float(payload["mean"]), std=float(payload["std"]))


def _check_finite(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.size and not np.all(np.isfinite(out)):
        raise InvalidSignalError("cannot normalize an array containing NaN or inf")
    return out


def fit(arrays, eps: float = config.NORMALIZE_EPS) -> NormStats:
    """Fit z-score statistics over one array or an iterable of arrays.

    Pass **training records only**. Uses a streaming sum of squares so a full
    48-record concatenation never has to exist in memory.
    """
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    total = 0
    acc_sum = 0.0
    acc_sq = 0.0
    for arr in arrays:
        flat = _check_finite(np.ravel(arr))
        total += flat.size
        acc_sum += float(flat.sum())
        acc_sq += float(np.square(flat).sum())

    if total == 0:
        raise ValueError("cannot fit normalization statistics on zero samples")

    mean = acc_sum / total
    variance = max(acc_sq / total - mean * mean, 0.0)
    return NormStats(mean=mean, std=float(np.sqrt(variance)) + eps)


def apply(arr: np.ndarray, stats: NormStats) -> np.ndarray:
    """Apply fitted statistics. Deterministic and idempotent for fixed stats."""
    values = _check_finite(arr)
    return ((values - stats.mean) / stats.std).astype(np.float32)


def zscore(arr: np.ndarray, eps: float = config.NORMALIZE_EPS) -> np.ndarray:
    """Per-array z-score.

    The epsilon guard means a flatline segment -- which occurs in real leads-off
    intervals -- yields zeros instead of NaN. One NaN here would poison every
    batch that segment lands in.
    """
    values = _check_finite(arr)
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 0.0
    return ((values - mean) / (std + eps)).astype(np.float32)
