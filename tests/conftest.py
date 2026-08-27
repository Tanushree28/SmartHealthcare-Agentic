"""Shared fixtures.

Hard constraint from PlanDoc/PLAN.md: the whole suite runs offline with no real
dataset present. These fixtures synthesize ECG-like signals and write *real*
WFDB records into ``tmp_path``, so the wfdb reading path is genuinely exercised
without a download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg import config  # noqa: E402


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_ecg():
    """A clean 20-beat signal and its true R-peak indices."""
    return make_ecg()


@pytest.fixture
def synthetic_record(tmp_path):
    """Two-channel record (MLII ch0, V5 ch1) with beat and non-beat symbols.

    Annotations: a rhythm marker ``+`` first, then alternating N / V / A beats,
    with a signal-quality marker ``~`` in the middle. This mix is what the
    label-exclusion tests need.
    """
    sig, r_peaks = make_ecg(n_beats=20)
    other = sig * 0.6

    symbols: list[str] = []
    samples: list[int] = []

    # Non-beat rhythm marker before the first beat.
    samples.append(int(r_peaks[0]) - 50)
    symbols.append("+")

    beat_cycle = ["N", "N", "V", "N", "A"]
    for i, peak in enumerate(r_peaks):
        samples.append(int(peak))
        symbols.append(beat_cycle[i % len(beat_cycle)])
        if i == 10:
            samples.append(int(peak) + 60)
            symbols.append("~")  # signal-quality marker

    order = np.argsort(np.asarray(samples, dtype=np.int64), kind="stable")
    samples_arr = np.asarray(samples, dtype=np.int64)[order]
    symbols_sorted = [symbols[i] for i in order]

    directory = tmp_path / "mitdb"
    write_record(
        directory,
        "sim001",
        {"MLII": sig, "V5": other},
        ann_samples=samples_arr,
        ann_symbols=symbols_sorted,
    )

    return {
        "dir": directory,
        "name": "sim001",
        "signal": sig,
        "r_peaks": r_peaks,
        "ann_samples": samples_arr,
        "ann_symbols": symbols_sorted,
        "n_beats": len(r_peaks),
        "n_nonbeat": 2,
    }


@pytest.fixture
def tiny_dataset(tmp_path):
    """Three synthetic records -- enough to exercise splitting and the manifest."""
    directory = tmp_path / "mitdb"
    names = ["901", "902", "903"]

    for i, name in enumerate(names):
        sig, r_peaks = make_ecg(n_beats=12 + i, seed=i)
        symbols = ["N" if j % 3 else "V" for j in range(len(r_peaks))]
        samples = list(map(int, r_peaks))

        # One rhythm marker per record so manifest arithmetic has a non-beat term.
        samples.append(int(r_peaks[0]) - 40)
        symbols.append("+")

        order = np.argsort(np.asarray(samples), kind="stable")
        write_record(
            directory,
            name,
            {"MLII": sig, "V1": sig * 0.5},
            ann_samples=np.asarray(samples)[order],
            ann_symbols=[symbols[k] for k in order],
        )

    return {"dir": directory, "names": names}


@pytest.fixture
def no_network(monkeypatch):
    """Make any socket creation raise, so a test proves it stays offline."""
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during offline test")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    return True
