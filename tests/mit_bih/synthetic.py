"""Synthetic ECG signals and real WFDB records for the offline test suite.

Plain functions, importable from test modules (``from .synthetic import ...``).
Fixtures that use them live in ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

from mit_bih import config

# ---------------------------------------------------------------------------
# Signal synthesis
# ---------------------------------------------------------------------------


def make_ecg(
    n_beats: int = 20,
    fs: int = config.FS,
    rr_samples: int = 360,
    first_peak: int = 200,
    qrs_amplitude: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic ECG-like signal with QRS spikes at known indices.

    Returns ``(signal, r_peaks)``. The spike is a narrow symmetric triangle so
    its maximum sits exactly on the annotated sample -- that is what lets the
    zero-phase filtering test assert index preservation.
    """
    n_samples = first_peak + n_beats * rr_samples + first_peak
    rng = np.random.default_rng(seed)
    sig = 0.01 * rng.standard_normal(n_samples)

    r_peaks = first_peak + np.arange(n_beats) * rr_samples

    # Narrow triangular QRS: half-width 4 samples (~11 ms at 360 Hz).
    half = 4
    ramp = np.concatenate([np.arange(half + 1), np.arange(half - 1, -1, -1)])
    kernel = qrs_amplitude * ramp / half

    for peak in r_peaks:
        lo = peak - half
        sig[lo : lo + kernel.size] += kernel

    return sig.astype(np.float64), r_peaks.astype(np.int64)


def add_baseline_wander(sig: np.ndarray, fs: int, amplitude: float, freq: float = 0.05):
    t = np.arange(sig.size) / fs
    return sig + amplitude * np.sin(2 * np.pi * freq * t)


def add_tone(sig: np.ndarray, fs: int, amplitude: float, freq: float):
    t = np.arange(sig.size) / fs
    return sig + amplitude * np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# WFDB record synthesis
# ---------------------------------------------------------------------------


def write_record(
    directory: Path,
    name: str,
    signals: dict[str, np.ndarray],
    fs: int = config.FS,
    ann_samples: np.ndarray | None = None,
    ann_symbols: list[str] | None = None,
) -> Path:
    """Write a real two-channel WFDB record plus ``.atr`` into ``directory``.

    ``signals`` maps lead name -> 1-D array; channel order follows dict order,
    which is what lets a test put V5 in channel 0 and MLII in channel 1.
    """
    directory.mkdir(parents=True, exist_ok=True)
    sig_names = list(signals)
    p_signal = np.column_stack([signals[n] for n in sig_names])

    wfdb.wrsamp(
        record_name=name,
        fs=fs,
        units=["mV"] * len(sig_names),
        sig_name=sig_names,
        p_signal=p_signal,
        fmt=["212"] * len(sig_names),
        adc_gain=[200.0] * len(sig_names),
        baseline=[0] * len(sig_names),
        write_dir=str(directory),
    )

    if ann_samples is not None:
        wfdb.wrann(
            record_name=name,
            extension="atr",
            sample=np.asarray(ann_samples, dtype=np.int64),
            symbol=list(ann_symbols or []),
            fs=fs,
            write_dir=str(directory),
        )

    return directory / name
