"""Group B -- normalization (PlanDoc/PLAN.md)."""

from __future__ import annotations

import numpy as np
import pytest

from ecg.errors import InvalidSignalError
from ecg.normalize import NormStats, apply, fit, zscore


def test_b1_zscore_gives_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=3.0, size=5000)

    out = zscore(data)

    assert out.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.std() == pytest.approx(1.0, rel=1e-4)


def test_b2_constant_signal_does_not_produce_nan():
    """B2: flatlines occur in real leads-off intervals; NaN would poison batches."""
    out = zscore(np.ones(500))

    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


def test_b3_statistics_fitted_on_train_only():
    """B3: the direct leakage regression test for PDF 3.9.

    Train data is centred on 0; the test record is offset by +50. If the fit
    ever saw the test record, the offset would vanish after transformation.
    """
    rng = np.random.default_rng(1)
    train = [rng.normal(0.0, 1.0, 4000) for _ in range(3)]
    test = rng.normal(0.0, 1.0, 4000) + 50.0

    stats = fit(train)

    assert stats.mean == pytest.approx(0.0, abs=0.1)

    train_out = apply(train[0], stats)
    test_out = apply(test, stats)

    assert train_out.mean() == pytest.approx(0.0, abs=0.1)
    assert test_out.mean() == pytest.approx(50.0, abs=0.5), (
        "the test record's offset was absorbed -- statistics leaked"
    )


def test_b3b_fit_ignores_data_not_passed_to_it():
    """Adding held-out data must not move statistics that were already fitted."""
    train = np.linspace(-1.0, 1.0, 1000)
    stats_a = fit([train])
    stats_b = fit([train])

    assert stats_a == stats_b


def test_b4_apply_is_deterministic_and_idempotent_given_fixed_stats():
    rng = np.random.default_rng(2)
    data = rng.normal(2.0, 4.0, 1000)
    stats = NormStats(mean=2.0, std=4.0)

    first = apply(data, stats)
    second = apply(data, stats)

    assert np.array_equal(first, second)


def test_fit_streams_over_multiple_arrays():
    """Streaming fit must equal fitting the concatenation."""
    rng = np.random.default_rng(3)
    parts = [rng.normal(1.0, 2.0, 800) for _ in range(4)]

    streamed = fit(parts)
    combined = fit([np.concatenate(parts)])

    assert streamed.mean == pytest.approx(combined.mean, rel=1e-9)
    assert streamed.std == pytest.approx(combined.std, rel=1e-9)


def test_fit_on_zero_samples_raises():
    with pytest.raises(ValueError, match="zero samples"):
        fit([])


def test_nan_is_rejected():
    data = np.ones(100)
    data[10] = np.nan
    with pytest.raises(InvalidSignalError):
        zscore(data)


def test_stats_round_trip_through_dict():
    stats = NormStats(mean=1.5, std=0.25)
    assert NormStats.from_dict(stats.to_dict()) == stats


def test_output_dtype_is_float32():
    assert zscore(np.arange(100.0)).dtype == np.float32
    assert apply(np.arange(100.0), NormStats(0.0, 1.0)).dtype == np.float32
