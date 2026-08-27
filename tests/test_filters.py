"""Group A -- filtering (PlanDoc/PLAN.md)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import add_baseline_wander, add_tone, make_ecg
from ecg import config
from ecg.errors import InvalidSignalError, SignalTooShortError
from ecg.filters import bandpass, denoise, notch


def band_power(sig: np.ndarray, fs: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(sig.size, d=1 / fs)
    mask = (freqs >= lo) & (freqs < hi)
    return float(spec[mask].sum())


def test_a1_bandpass_removes_wander_and_keeps_qrs(synthetic_ecg):
    """A1: drift is suppressed without flattening the QRS complex.

    A filter that removes drift but also crushes the QRS destroys the very
    morphology a classifier needs, so both halves are asserted.
    """
    clean, r_peaks = synthetic_ecg
    qrs_amp = float(clean[r_peaks].mean())
    noisy = add_baseline_wander(clean, config.FS, amplitude=3.0 * qrs_amp)

    filtered = bandpass(noisy, fs=config.FS)

    before = band_power(noisy, config.FS, 0.0, 0.2)
    after = band_power(filtered, config.FS, 0.0, 0.2)
    assert after < before / 1000, f"drift power {after:.3g} vs {before:.3g}"

    # QRS amplitude survives. Measured peak-to-local-baseline, away from edges.
    interior = r_peaks[2:-2]
    recovered = float(filtered[interior].mean() - np.median(filtered))
    assert recovered > 0.7 * qrs_amp, f"QRS flattened: {recovered:.3f} vs {qrs_amp:.3f}"


def test_a2_filtering_is_zero_phase_r_peaks_preserved(synthetic_ecg):
    """A2: the alignment guarantee every downstream beat window depends on.

    This is exactly what breaks if someone later 'optimises' filtfilt into
    lfilter, so it is asserted at +/- 1 sample.
    """
    clean, r_peaks = synthetic_ecg
    noisy = add_baseline_wander(clean, config.FS, amplitude=2.0)

    filtered = bandpass(noisy, fs=config.FS)

    for peak in r_peaks[1:-1]:
        window = filtered[peak - 20 : peak + 21]
        recovered = peak - 20 + int(np.argmax(window))
        assert abs(recovered - peak) <= 1, f"peak {peak} moved to {recovered}"


def test_a3_notch_attenuates_powerline_when_passband_admits_it():
    """A3: proves the notch is real and correctly centred.

    Documents the off-by-default choice rather than covering for broken code.
    """
    clean, _ = make_ecg(n_beats=40)
    fs = config.FS
    noisy = add_tone(clean, fs, amplitude=0.5, freq=config.NOTCH_FREQ_HZ)

    wide = bandpass(noisy, fs=fs, low=0.5, high=100.0)
    notched = notch(wide, fs=fs)

    at_60_before = band_power(wide, fs, 59.0, 61.0)
    at_60_after = band_power(notched, fs, 59.0, 61.0)
    assert at_60_after < at_60_before / 100

    # Neighbouring bands survive.
    near_before = band_power(wide, fs, 20.0, 40.0)
    near_after = band_power(notched, fs, 20.0, 40.0)
    assert near_after > 0.9 * near_before


def test_a3b_notch_is_off_by_default():
    """The default chain must not silently apply a no-op notch."""
    assert config.NOTCH_ENABLED_BY_DEFAULT is False
    sig, _ = make_ecg(n_beats=10)
    assert np.array_equal(denoise(sig), bandpass(sig))


def test_a4_signal_shorter_than_padlen_raises_clear_error():
    """A4: short trailing fragments are realistic; the failure must be diagnosable."""
    with pytest.raises(SignalTooShortError, match="padlen"):
        bandpass(np.zeros(10), fs=config.FS)


def test_a5_nan_input_is_rejected():
    """A5: filtfilt spreads one NaN across the whole array, voiding a record."""
    sig, _ = make_ecg(n_beats=10)
    sig[500] = np.nan
    with pytest.raises(InvalidSignalError, match="non-finite"):
        bandpass(sig, fs=config.FS)


def test_a5b_inf_input_is_rejected():
    sig, _ = make_ecg(n_beats=10)
    sig[500] = np.inf
    with pytest.raises(InvalidSignalError):
        bandpass(sig, fs=config.FS)


def test_a6_invalid_band_edges_rejected():
    sig, _ = make_ecg(n_beats=10)
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass(sig, fs=config.FS, low=0.5, high=500.0)
