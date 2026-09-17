"""Calibration metrics must move when calibration changes, and not otherwise.

The failure mode to avoid is the one recorded as Problem #10 in
``IMPLEMENTATION.md``: a test that passes for the right and the wrong code
alike. So each metric here is checked against a case where its value is known by
construction *and* a case that should move it, rather than only against a
hand-computed number on three rows.

The synthetic calibrated fixture is the load-bearing one: draw ``p`` uniformly,
then draw the label with probability ``p``. By construction the observed rate in
any confidence bin equals the mean confidence of that bin, so a correct ECE goes
to zero as ``n`` grows and an incorrect one does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg import calibration


def _binary_proba(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    return np.column_stack([1.0 - s, s])


def calibrated_fixture(n: int = 200_000, seed: int = 0):
    """Labels drawn at exactly the predicted rate: perfectly calibrated."""
    rng = np.random.default_rng(seed)
    scores = rng.random(n)
    y = (rng.random(n) < scores).astype(np.int64)
    return y, _binary_proba(scores)


class TestBrierAndNLL:
    def test_brier_matches_hand_computation(self):
        y = np.array([1, 0])
        proba = np.array([[0.25, 0.75], [0.6, 0.4]])
        # row 0: (0.25-0)^2 + (0.75-1)^2 = 0.125
        # row 1: (0.6-1)^2  + (0.4-0)^2  = 0.32
        assert calibration.brier_score(y, proba) == pytest.approx(0.2225)

    def test_brier_is_zero_for_perfect_confident_predictions(self):
        y = np.array([0, 1, 1])
        proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        assert calibration.brier_score(y, proba) == pytest.approx(0.0)

    def test_nll_matches_hand_computation(self):
        y = np.array([1, 0])
        proba = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert calibration.negative_log_likelihood(y, proba) == pytest.approx(
            -np.log(0.5)
        )

    def test_nll_survives_a_zero_probability_on_the_true_class(self):
        """float64 softmax reaches exact 0.0; without a floor this is inf."""
        y = np.array([1])
        proba = np.array([[1.0, 0.0]])
        assert np.isfinite(calibration.negative_log_likelihood(y, proba))

    def test_rejects_unnormalised_rows(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            calibration.brier_score(np.array([0]), np.array([[2.0, 3.0]]))


class TestExpectedCalibrationError:
    def test_near_zero_on_a_calibrated_fixture(self):
        y, proba = calibrated_fixture()
        assert calibration.expected_calibration_error(y, proba, n_bins=10) < 0.01

    def test_large_on_an_overconfident_fixture(self):
        """Sharpen a calibrated fixture toward 0/1 and ECE must notice."""
        y, proba = calibrated_fixture()
        sharpened = np.clip((proba[:, 1] - 0.5) * 4.0 + 0.5, 0.0, 1.0)

        before = calibration.expected_calibration_error(y, proba, n_bins=10)
        after = calibration.expected_calibration_error(y, _binary_proba(sharpened), 10)

        assert after > before + 0.1

    def test_rejects_zero_bins(self):
        y, proba = calibrated_fixture(n=100)
        with pytest.raises(ValueError, match="n_bins must be positive"):
            calibration.expected_calibration_error(y, proba, n_bins=0)


class TestPriorDrift:
    def test_ratio_is_one_when_calibrated(self):
        y, proba = calibrated_fixture()
        drift = calibration.prior_drift(y, proba, classes=("A", "B"))
        assert drift[1].ratio == pytest.approx(1.0, abs=0.02)

    def test_ratio_reports_the_inflation_factor(self):
        """The distortion loss weighting is expected to cause, isolated."""
        rng = np.random.default_rng(1)
        n, prevalence = 50_000, 0.02
        y = (rng.random(n) < prevalence).astype(np.int64)

        # A model that says 0.20 for every beat on a class that occurs 2% of
        # the time is overstating it tenfold -- a cap-10 weight, exactly.
        proba = _binary_proba(np.full(n, 0.20))
        drift = calibration.prior_drift(y, proba, classes=("N", "S"))

        assert drift[1].name == "S"
        assert drift[1].ratio == pytest.approx(10.0, rel=0.1)


class TestReliabilityCurve:
    def test_counts_account_for_every_beat(self):
        y, proba = calibrated_fixture(n=5_000)
        curve = calibration.reliability_curve(y, proba, 1, n_bins=10, name="S")
        assert curve.n_binned == y.size

    def test_empty_bins_are_nan_not_zero(self):
        y = np.array([0, 1])
        proba = _binary_proba([0.95, 0.96])  # everything lands in the top bin
        curve = calibration.reliability_curve(y, proba, 1, n_bins=10, name="S")

        assert curve.counts[-1] == 2
        assert np.isnan(curve.observed_rate[0])
        assert curve.n_binned == 2

    def test_gap_is_near_zero_when_calibrated(self):
        y, proba = calibrated_fixture()
        curve = calibration.reliability_curve(y, proba, 1, n_bins=10, name="S")
        assert np.nanmax(np.abs(curve.gap())) < 0.02

    def test_probability_of_one_lands_in_the_last_bin(self):
        y = np.array([1])
        proba = _binary_proba([1.0])
        curve = calibration.reliability_curve(y, proba, 1, n_bins=10, name="S")
        assert curve.counts[-1] == 1


class TestRiskCoverage:
    def test_accuracy_falls_as_coverage_grows_when_confidence_is_informative(self):
        rng = np.random.default_rng(2)
        n = 20_000
        scores = rng.random(n)
        y = (rng.random(n) < scores).astype(np.int64)

        coverage, accuracy = calibration.risk_coverage_curve(y, _binary_proba(scores))

        assert coverage[0] < coverage[-1]
        assert accuracy[0] > accuracy[-1] + 0.05

    def test_accuracy_is_flat_when_confidence_carries_no_information(self):
        """A flat curve is the signal that abstention would buy nothing."""
        rng = np.random.default_rng(3)
        n = 20_000
        y = rng.integers(0, 2, n)
        # Confidence varies but is unrelated to correctness.
        scores = rng.random(n) * 0.4 + 0.3
        coverage, accuracy = calibration.risk_coverage_curve(y, _binary_proba(scores))

        assert abs(accuracy[0] - accuracy[-1]) < 0.05
