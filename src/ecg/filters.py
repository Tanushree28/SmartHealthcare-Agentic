"""Denoising filters (PDF 2.2).

Everything here is applied with :func:`scipy.signal.filtfilt`, which runs the
filter forward and backward so the net phase response is zero. That is not a
stylistic choice: a causal IIR filter has frequency-dependent group delay, which
would shift the QRS complex away from its annotated sample index and misalign
every beat window downstream by an unknown amount. The alignment guarantee in
:mod:`ecg.segment` rests entirely on this.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from . import config
from .errors import InvalidSignalError, SignalTooShortError


def _validate(sig: np.ndarray) -> np.ndarray:
    arr = np.asarray(sig, dtype=np.float64)
    if arr.ndim != 1:
        raise InvalidSignalError(f"expected a 1-D signal, got shape {arr.shape}")
    if arr.size == 0:
        raise SignalTooShortError("signal is empty")
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise InvalidSignalError(
            f"signal contains {n_bad} non-finite sample(s) (NaN or inf); "
            "filtfilt would propagate these across the entire record"
        )
    return arr


def _apply(b: np.ndarray, a: np.ndarray, sig: np.ndarray, what: str) -> np.ndarray:
    padlen = 3 * max(len(a), len(b))
    if sig.size <= padlen:
        raise SignalTooShortError(
            f"{what} needs more than {padlen} samples (filtfilt padlen), "
            f"got {sig.size}"
        )
    return filtfilt(b, a, sig).astype(np.float64)


def bandpass(
    sig: np.ndarray,
    fs: int = config.FS,
    low: float = config.BANDPASS_LOW_HZ,
    high: float = config.BANDPASS_HIGH_HZ,
    order: int = config.BANDPASS_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass.

    The low edge removes baseline wander (respiration, electrode drift); the
    high edge removes muscle noise while leaving QRS morphology intact.
    """
    arr = _validate(sig)
    nyq = fs / 2.0
    if not 0 < low < high < nyq:
        raise ValueError(
            f"band edges must satisfy 0 < {low} < {high} < Nyquist ({nyq})"
        )
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return _apply(b, a, arr, "bandpass")


def notch(
    sig: np.ndarray,
    fs: int = config.FS,
    freq: float = config.NOTCH_FREQ_HZ,
    quality: float = config.NOTCH_QUALITY,
) -> np.ndarray:
    """Zero-phase IIR notch at the powerline frequency.

    Off by default. Under the default 0.5-40 Hz passband a 60 Hz notch removes
    nothing that the low-pass has not already removed -- running it anyway would
    be a no-op dressed up as denoising. It becomes meaningful only if the
    passband is widened past 50 Hz.
    """
    arr = _validate(sig)
    nyq = fs / 2.0
    if not 0 < freq < nyq:
        raise ValueError(f"notch frequency {freq} must be below Nyquist ({nyq})")
    b, a = iirnotch(freq / nyq, quality)
    return _apply(b, a, arr, "notch")


def denoise(
    sig: np.ndarray,
    fs: int = config.FS,
    low: float = config.BANDPASS_LOW_HZ,
    high: float = config.BANDPASS_HIGH_HZ,
    order: int = config.BANDPASS_ORDER,
    use_notch: bool = config.NOTCH_ENABLED_BY_DEFAULT,
    notch_freq: float = config.NOTCH_FREQ_HZ,
) -> np.ndarray:
    """Full denoising chain: bandpass, then optionally notch."""
    out = bandpass(sig, fs=fs, low=low, high=high, order=order)
    if use_notch:
        out = notch(out, fs=fs, freq=notch_freq)
    return out
