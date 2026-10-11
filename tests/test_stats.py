"""Paired comparison and bootstrap intervals.

Phase 1d could not rank three of its arms: fold-to-fold variance (std 0.13) was
far larger than the differences between arms (0.007). The paired per-fold
comparison *did* separate them, because it cancels fold difficulty. This module
makes that the default way results are compared.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg import stats


def test_paired_difference_cancels_a_shared_offset():
    """The point of pairing: fold difficulty affects both arms equally."""
    baseline = np.array([0.10, 0.50, 0.30, 0.70, 0.40])
    treatment = baseline + 0.05  # uniformly better, wildly varying folds

    result = stats.paired_comparison(baseline, treatment)

    assert result.mean_difference == pytest.approx(0.05)
    assert result.wins == 5
    assert result.n == 5
    assert result.consistent is True


def test_unpaired_means_would_have_missed_it():
    """A sanity check on the motivation, not on the code."""
    baseline = np.array([0.10, 0.50, 0.30, 0.70, 0.40])
    treatment = baseline + 0.05

    spread = baseline.std(ddof=1)

    assert spread > 0.2  # variance dwarfs the 0.05 effect
    assert stats.paired_comparison(baseline, treatment).mean_difference == pytest.approx(0.05)


def test_a_mixed_result_is_not_consistent():
    result = stats.paired_comparison(
        np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.4, 0.7])
    )

    assert result.wins == 2
    assert result.consistent is False


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        stats.paired_comparison(np.array([0.1, 0.2]), np.array([0.1]))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        stats.paired_comparison(np.array([]), np.array([]))


def test_bootstrap_interval_brackets_the_observed_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=200)

    low, high = stats.bootstrap_ci(values, seed=1)

    assert low < values.mean() < high


def test_a_wider_spread_gives_a_wider_interval():
    rng = np.random.default_rng(0)
    tight = rng.normal(0.5, 0.01, size=100)
    loose = rng.normal(0.5, 0.20, size=100)

    tight_lo, tight_hi = stats.bootstrap_ci(tight, seed=1)
    loose_lo, loose_hi = stats.bootstrap_ci(loose, seed=1)

    assert (loose_hi - loose_lo) > (tight_hi - tight_lo)


def test_bootstrap_is_reproducible_from_the_seed():
    values = np.linspace(0, 1, 50)

    assert stats.bootstrap_ci(values, seed=7) == stats.bootstrap_ci(values, seed=7)


def test_bootstrap_ci_of_a_constant_is_a_point():
    low, high = stats.bootstrap_ci(np.full(20, 0.42), seed=3)

    assert low == pytest.approx(0.42)
    assert high == pytest.approx(0.42)


def test_classification_ci_resamples_beats_not_folds():
    """DS2 is one draw of ~49,646 beats; the interval says how much that draw
    could have differed."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=2000)
    y_pred = y_true.copy()
    y_pred[:200] = 1 - y_pred[:200]  # 10% error

    low, high = stats.metric_ci(y_true, y_pred, metric="s_recall", seed=5, n_boot=200)

    assert 0.0 <= low <= high <= 1.0


def test_paired_comparison_reports_a_readable_summary():
    result = stats.paired_comparison(np.array([0.1, 0.2]), np.array([0.3, 0.5]))

    text = str(result)

    assert "2/2" in text
