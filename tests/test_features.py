"""Feature extraction for the classical baseline (PDF 2.9)."""

from __future__ import annotations

import numpy as np
import pytest

from ecg import config
from ecg.features import (
    DEFAULT_POOL_FACTOR,
    RR_FEATURE_NAMES,
    extract_record,
    extract_records,
    pool_morphology,
    rr_features,
)


def make_beats(n: int, rr_samples: int = 360, length: int = config.BEAT_LENGTH):
    """n beats at a constant interval, with a distinguishable per-beat value."""
    X = np.tile(np.arange(length, dtype=np.float32), (n, 1))
    X += np.arange(n, dtype=np.float32)[:, None]
    y = np.zeros(n, dtype=np.int8)
    r_peaks = 500 + np.arange(n, dtype=np.int64) * rr_samples
    return X, y, r_peaks


def test_rr_intervals_are_correct_in_seconds():
    """A constant 360-sample spacing at 360 Hz is exactly 1.0 s."""
    _, _, peaks = make_beats(10, rr_samples=360)

    features, keep = rr_features(peaks, fs=360)

    pre = features[:, RR_FEATURE_NAMES.index("pre_rr")]
    post = features[:, RR_FEATURE_NAMES.index("post_rr")]
    assert np.allclose(pre, 1.0)
    assert np.allclose(post, 1.0)
    assert np.allclose(features[:, RR_FEATURE_NAMES.index("global_rr")], 1.0)
    assert np.allclose(features[:, RR_FEATURE_NAMES.index("pre_rr_ratio")], 1.0)


def test_rr_captures_a_premature_beat():
    """The S class is defined by early arrival, so this must show up in pre_rr.

    A beat arriving at half the usual interval is the single most important
    thing the RR features exist to represent.
    """
    peaks = np.array([0, 360, 720, 900, 1260, 1620], dtype=np.int64)

    features, keep = rr_features(peaks, fs=360)

    pre = features[:, RR_FEATURE_NAMES.index("pre_rr")]
    # keep = [1,2,3,4]; the beat at index 3 (sample 900) arrived 0.5 s after 720.
    assert pre[2] == pytest.approx(0.5)
    ratio = features[:, RR_FEATURE_NAMES.index("pre_rr_ratio")]
    assert ratio[2] < 0.7, "a premature beat must stand out against the record mean"


def test_first_and_last_beat_are_dropped_not_imputed():
    """Same rule as Phase 1's boundary beats: drop and count, never invent."""
    X, y, peaks = make_beats(10)

    result = extract_record(X, y, peaks, "901")

    assert result.n_input == 10
    assert result.n_kept == 8
    assert result.n_dropped_edge == 2
    assert result.reconciles()


def test_dropped_beats_are_the_first_and_last_specifically():
    X, y, peaks = make_beats(6)
    y = np.arange(6, dtype=np.int8)  # label == position, so we can identify rows

    result = extract_record(X, y, peaks, "901", use_morphology=False)

    assert result.y.tolist() == [1, 2, 3, 4]


def test_non_monotonic_r_peaks_are_rejected():
    """A negative interval would silently produce nonsense features."""
    peaks = np.array([0, 360, 200, 720], dtype=np.int64)

    with pytest.raises(ValueError, match="strictly increasing"):
        rr_features(peaks)


def test_record_with_fewer_than_three_beats_yields_nothing():
    X, y, peaks = make_beats(2)

    result = extract_record(X, y, peaks, "901")

    assert result.n_kept == 0
    assert result.n_dropped_edge == 2
    assert result.reconciles()


def test_pooling_averages_rather_than_subsamples():
    """Subsampling would alias the QRS complex, the sharpest part of the beat."""
    X = np.zeros((1, 300), dtype=np.float32)
    X[0, :10] = np.array([0, 0, 0, 0, 10, 0, 0, 0, 0, 0], dtype=np.float32)

    pooled = pool_morphology(X, factor=10)

    assert pooled.shape == (1, 30)
    assert pooled[0, 0] == pytest.approx(1.0), "a spike must survive as its mean"


def test_pool_factor_must_divide_the_beat_length():
    X = np.zeros((2, 300), dtype=np.float32)

    with pytest.raises(ValueError, match="not divisible"):
        pool_morphology(X, factor=7)


def test_pool_factor_one_is_a_passthrough():
    X = np.arange(600, dtype=np.float32).reshape(2, 300)
    assert np.array_equal(pool_morphology(X, factor=1), X)


def test_feature_matrix_shape_and_names():
    X, y, peaks = make_beats(12)

    result = extract_record(X, y, peaks, "901")

    expected = len(RR_FEATURE_NAMES) + config.BEAT_LENGTH // DEFAULT_POOL_FACTOR
    assert result.X.shape == (10, expected)
    assert len(result.feature_names) == expected
    assert result.feature_names[: len(RR_FEATURE_NAMES)] == RR_FEATURE_NAMES
    assert result.X.dtype == np.float32


def test_ablation_rr_only():
    X, y, peaks = make_beats(12)

    result = extract_record(X, y, peaks, "901", use_morphology=False)

    assert result.X.shape == (10, len(RR_FEATURE_NAMES))
    assert result.feature_names == RR_FEATURE_NAMES


def test_ablation_morphology_only_keeps_every_beat():
    """Without RR features no beat needs a neighbour, so none is dropped."""
    X, y, peaks = make_beats(12)

    result = extract_record(X, y, peaks, "901", use_rr=False)

    assert result.n_kept == 12
    assert result.n_dropped_edge == 0
    assert result.X.shape == (12, config.BEAT_LENGTH // DEFAULT_POOL_FACTOR)


def test_at_least_one_feature_family_is_required():
    X, y, peaks = make_beats(12)

    with pytest.raises(ValueError, match="at least one"):
        extract_record(X, y, peaks, "901", use_rr=False, use_morphology=False)


def test_mismatched_array_lengths_are_rejected():
    X, y, peaks = make_beats(10)

    with pytest.raises(ValueError, match="disagree"):
        extract_record(X, y[:5], peaks, "901")


def test_global_rr_uses_only_this_record():
    """Cross-record contamination in a feature would be leakage by another name."""
    _, _, slow = make_beats(8, rr_samples=720)
    _, _, fast = make_beats(8, rr_samples=180)

    slow_features, _ = rr_features(slow, fs=360)
    fast_features, _ = rr_features(fast, fs=360)

    idx = RR_FEATURE_NAMES.index("global_rr")
    assert slow_features[:, idx].mean() == pytest.approx(2.0)
    assert fast_features[:, idx].mean() == pytest.approx(0.5)


def test_extract_records_preserves_record_provenance(tmp_path):
    beats = tmp_path / "beats"
    beats.mkdir()
    for name in ("901", "902"):
        X, y, peaks = make_beats(10)
        np.savez(beats / f"{name}.npz", X=X, y=y, r_peaks=peaks)

    result = extract_records(beats, ["901", "902"])

    assert result.n_kept == 16
    assert set(result.record) == {"901", "902"}
    assert result.reconciles()


def test_extract_records_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess_mitbih.py"):
        extract_records(tmp_path, ["999"])
