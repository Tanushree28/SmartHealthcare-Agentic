"""Move a signal and its annotation positions to another sampling rate.

The signal goes through ``scipy.signal.resample_poly``, whose FIR filter is
applied with its delay removed, so sample ``i`` at ``fs_in`` lands at
``i * fs_out / fs_in``. Annotation positions are moved by the same factor and
rounded half up in integer arithmetic: Python's ``round`` rounds halves to
even, and exact halves do occur (180 samples at 360 Hz is 128.5 at 257 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class MovedPositions:
    """Positions at the new rate, and which of the original ones survived."""

    samples: np.ndarray  # int64, positions inside the resampled signal
    keep: np.ndarray  # bool, one per original position

    @property
    def n_dropped(self) -> int:
        return int(self.keep.size - np.count_nonzero(self.keep))


def resample_signal(signal, fs_in: int, fs_out: int) -> np.ndarray:
    """Resample a 1-D signal from ``fs_in`` to ``fs_out`` Hz."""
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(np.asarray(signal, dtype=np.float64), fs_out // g, fs_in // g)


def move_positions(samples, fs_in: int, fs_out: int, n_out: int) -> MovedPositions:
    """Move sample positions to ``fs_out``; drop those outside ``[0, n_out)``."""
    s = np.asarray(samples, dtype=np.int64)
    moved = (2 * s * fs_out + fs_in) // (2 * fs_in)  # round half up
    keep = (moved >= 0) & (moved < n_out)
    return MovedPositions(samples=moved[keep], keep=keep)


def resample_record(signal, samples, fs_in: int, fs_out: int) -> tuple[np.ndarray, MovedPositions]:
    """Resample a signal and move its annotation positions with it."""
    out = resample_signal(signal, fs_in, fs_out)
    return out, move_positions(samples, fs_in, fs_out, out.size)


def round_trip(signal, samples, fs: int, via_fs: int) -> tuple[np.ndarray, MovedPositions]:
    """``fs`` -> ``via_fs`` -> ``fs``, positions rounded onto each grid in turn.

    ``keep`` refers to the original positions, so labels can follow them.
    """
    mid, first = resample_record(signal, samples, fs, via_fs)
    out, second = resample_record(mid, first.samples, via_fs, fs)
    keep = first.keep.copy()
    keep[first.keep] = second.keep
    return out, MovedPositions(samples=second.samples, keep=keep)
