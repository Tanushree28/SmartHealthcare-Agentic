"""Robust rhythm baseline.

``global_rr`` is the mean RR over a whole record, so one arrhythmic stretch
shifts the reference for every beat in that patient. A beat is "premature"
relative to the rhythm *around it*, not to a record-level average that an
episode of bigeminy has dragged downwards.

This replaces the mean with a rolling median over surrounding beats, which a
run of ectopics cannot move nearly as far.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg import features


def test_rolling_median_of_a_steady_rhythm_is_that_rhythm():
    intervals = np.full(50, 0.85)

    baseline = features.rolling_baseline_rr(intervals, half_window=10)

    assert baseline == pytest.approx(0.85)


def test_an_ectopic_run_barely_moves_the_median_but_moves_the_mean():
    """The whole reason this function exists."""
    intervals = np.full(100, 0.85)
    intervals[40:60] = 0.40  # a 20-beat run of very short intervals

    mean_baseline = np.full(100, intervals.mean())
    median_baseline = features.rolling_baseline_rr(intervals, half_window=25)

    # Far from the episode, the median still reports the true rhythm.
    assert median_baseline[5] == pytest.approx(0.85, abs=0.01)
    # The mean is dragged down everywhere, including where nothing happened.
    assert mean_baseline[5] < 0.80


def test_output_length_matches_input():
    intervals = np.linspace(0.7, 1.0, 37)

    assert features.rolling_baseline_rr(intervals).shape == (37,)


def test_edges_use_the_beats_that_exist():
    intervals = np.full(10, 0.9)

    baseline = features.rolling_baseline_rr(intervals, half_window=20)

    assert np.isfinite(baseline).all()
    assert baseline == pytest.approx(0.9)


def test_a_single_interval_is_its_own_baseline():
    assert features.rolling_baseline_rr(np.array([0.8])) == pytest.approx(0.8)


def test_empty_input_returns_empty():
    assert features.rolling_baseline_rr(np.array([])).shape == (0,)


def test_negative_half_window_is_rejected():
    with pytest.raises(ValueError, match="half_window"):
        features.rolling_baseline_rr(np.full(10, 0.8), half_window=-1)


# --- the feature block that uses it -----------------------------------------


def test_robust_rr_features_have_the_documented_columns():
    peaks = (200 + np.arange(40) * 360).astype(np.int64)

    matrix, keep = features.rr_features(peaks, robust=True)

    assert matrix.shape == (38, len(features.RR_FEATURE_NAMES))
    assert keep.size == 38


def test_robust_and_classic_agree_on_a_perfectly_steady_rhythm():
    """With no arrhythmia, median and mean coincide -- so the change is inert
    exactly where it should be."""
    peaks = (200 + np.arange(40) * 360).astype(np.int64)

    classic, _ = features.rr_features(peaks, robust=False)
    robust, _ = features.rr_features(peaks, robust=True)

    np.testing.assert_allclose(classic, robust, atol=1e-5)


def test_robust_differs_when_the_record_contains_an_episode():
    rr = np.full(80, 360)
    rr[30:50] = 150  # an ectopic run
    peaks = np.concatenate([[200], 200 + np.cumsum(rr)]).astype(np.int64)

    classic, _ = features.rr_features(peaks, robust=False)
    robust, _ = features.rr_features(peaks, robust=True)

    # global_rr is column 3; the ratios built from it are columns 4 and 5.
    assert not np.allclose(classic[:, 3], robust[:, 3])


def test_classic_behaviour_is_unchanged_by_default():
    """Phase 1b results must remain reproducible: robust is opt-in."""
    peaks = (200 + np.arange(20) * 300).astype(np.int64)

    default, _ = features.rr_features(peaks)
    explicit, _ = features.rr_features(peaks, robust=False)

    np.testing.assert_array_equal(default, explicit)
