"""Group D -- segmentation (PlanDoc/PLAN.md)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_ecg
from ecg import config
from ecg.labels import map_symbols
from ecg.segment import (
    empty_beats,
    segment_beats,
    segment_windows,
    suggested_class_weights,
)


def build(signal, samples, symbols):
    return segment_beats(signal, samples, map_symbols(symbols))


def test_d1_beat_shape_and_r_peak_alignment(synthetic_ecg):
    """D1: the core contract -- the R-peak sits at a fixed offset in every row."""
    sig, r_peaks = synthetic_ecg
    symbols = ["N"] * len(r_peaks)

    out = build(sig, r_peaks, symbols)

    assert out.X.shape == (len(r_peaks), config.BEAT_LENGTH)
    assert out.X.shape[1] == 300
    assert out.y.shape == (len(r_peaks),)
    assert out.r_peaks.shape == (len(r_peaks),)

    # The synthetic QRS maximum lands exactly on the annotated sample, so the
    # per-row argmax must be SAMPLES_BEFORE for every beat.
    assert np.all(out.X.argmax(axis=1) == config.SAMPLES_BEFORE)


def test_d2_boundary_beat_at_start_is_dropped_and_counted():
    """D2: the tempting fix -- zero-padding -- invents a flat pre-QRS segment."""
    sig, r_peaks = make_ecg(n_beats=5)
    samples = np.concatenate([[50], r_peaks])  # 50 < SAMPLES_BEFORE (100)
    symbols = ["N"] * len(samples)

    out = build(sig, samples, symbols)

    assert out.n_kept == len(r_peaks)
    assert out.n_dropped_boundary == 1
    assert out.drop_reasons["boundary:start"] == 1
    assert 50 not in out.r_peaks.tolist()
    assert out.reconciles()


def test_d3_boundary_beat_at_end_is_dropped_and_counted():
    sig, r_peaks = make_ecg(n_beats=5)
    late = sig.size - 10  # needs 200 samples of right context, has 10
    samples = np.concatenate([r_peaks, [late]])
    symbols = ["N"] * len(samples)

    out = build(sig, samples, symbols)

    assert out.n_kept == len(r_peaks)
    assert out.n_dropped_boundary == 1
    assert out.drop_reasons["boundary:end"] == 1
    assert out.reconciles()


def test_d3b_no_zero_padding_ever_occurs():
    """Every kept row must be a verbatim slice of the source signal."""
    sig, r_peaks = make_ecg(n_beats=6)
    out = build(sig, r_peaks, ["N"] * len(r_peaks))

    for row, peak in zip(out.X, out.r_peaks):
        expected = sig[peak - config.SAMPLES_BEFORE : peak + config.SAMPLES_AFTER]
        assert np.allclose(row, expected.astype(np.float32))


def test_d4_duplicate_and_non_monotonic_samples_do_not_corrupt_output():
    """D4: no negative-length or overlapping-index errors; deterministic result."""
    sig, r_peaks = make_ecg(n_beats=6)
    samples = np.array(
        [r_peaks[2], r_peaks[0], r_peaks[0], r_peaks[1]], dtype=np.int64
    )
    symbols = ["N", "V", "V", "N"]

    out = build(sig, samples, symbols)

    assert out.n_kept == 4
    assert out.X.shape == (4, config.BEAT_LENGTH)
    # Order is preserved as given; duplicates produce identical rows, not errors.
    assert np.allclose(out.X[1], out.X[2])
    assert out.reconciles()


def test_d5_window_segmentation_count_and_length():
    """D5: 30 s at 360 Hz with 10 s windows -> 3 windows of 3600 samples."""
    sig = np.arange(30 * config.FS, dtype=np.float32)

    windows = segment_windows(sig, window_seconds=10, fs=config.FS)

    assert windows.shape == (3, 3600)
    assert np.array_equal(windows[0], sig[:3600])


def test_d5b_trailing_partial_window_is_dropped_by_rule():
    sig = np.zeros(25 * config.FS, dtype=np.float32)

    windows = segment_windows(sig, window_seconds=10, fs=config.FS)

    assert windows.shape == (2, 3600)

    with pytest.raises(ValueError, match="zero-padding"):
        segment_windows(sig, window_seconds=10, fs=config.FS, drop_partial=False)


def test_d5c_signal_shorter_than_one_window_yields_empty():
    sig = np.zeros(100, dtype=np.float32)
    assert segment_windows(sig, window_seconds=10, fs=config.FS).shape == (0, 3600)


def test_d6_zero_annotations_yields_empty_arrays_of_correct_dtype():
    """D6: the empty case must reach a manifest row, not raise."""
    sig, _ = make_ecg(n_beats=3)

    out = build(sig, np.array([], dtype=np.int64), [])

    assert out.X.shape == (0, config.BEAT_LENGTH)
    assert out.X.dtype == np.float32
    assert out.y.dtype == np.int8
    assert out.r_peaks.dtype == np.int64
    assert out.n_total == 0
    assert out.reconciles()


def test_non_beat_annotations_are_excluded_and_counted():
    sig, r_peaks = make_ecg(n_beats=5)
    samples = np.concatenate([[r_peaks[0] - 30], r_peaks])
    symbols = ["+"] + ["N"] * len(r_peaks)

    out = build(sig, samples, symbols)

    assert out.n_kept == len(r_peaks)
    assert out.n_excluded_nonbeat == 1
    assert out.drop_reasons["nonbeat:+"] == 1
    assert out.reconciles()


def test_unknown_symbols_are_counted_separately():
    sig, r_peaks = make_ecg(n_beats=4)
    samples = np.concatenate([r_peaks, [r_peaks[-1] + 1]])
    symbols = ["N"] * len(r_peaks) + ["§"]

    out = build(sig, samples, symbols)

    assert out.n_unknown == 1
    assert out.reconciles()


def test_dtypes_of_kept_arrays():
    sig, r_peaks = make_ecg(n_beats=4)
    out = build(sig, r_peaks, ["N", "V", "A", "F"])

    assert out.X.dtype == np.float32
    assert out.y.dtype == np.int8
    assert out.r_peaks.dtype == np.int64
    assert out.y.tolist() == [0, 2, 1, 3]


def test_mapping_length_mismatch_raises():
    sig, r_peaks = make_ecg(n_beats=3)
    with pytest.raises(ValueError, match="length mismatch"):
        segment_beats(sig, r_peaks, map_symbols(["N"]))


def test_empty_beats_helper_reconciles():
    assert empty_beats().reconciles()


def test_suggested_class_weights_favour_rare_classes():
    weights = suggested_class_weights({"N": 900, "V": 100, "F": 0})

    assert set(weights) == {"N", "V"}, "absent classes must not get a weight"
    assert weights["V"] > weights["N"]
    assert suggested_class_weights({}) == {}
