"""Per-class decision thresholds, tuned out-of-fold."""

from __future__ import annotations

import numpy as np
import pytest

from ecg import config, evaluate, thresholds


def test_thresholds_of_one_reproduce_argmax():
    proba = np.array([[0.6, 0.3, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0]])
    ones = np.ones(len(config.AAMI_CLASSES))

    np.testing.assert_array_equal(
        thresholds.apply_thresholds(proba, ones), proba.argmax(axis=1)
    )


def test_lowering_a_threshold_makes_that_class_easier_to_predict():
    proba = np.array([[0.6, 0.4, 0.0, 0.0, 0.0]])
    t = np.ones(len(config.AAMI_CLASSES))

    assert thresholds.apply_thresholds(proba, t)[0] == 0

    t[1] = 0.5  # S needs only half the evidence now
    assert thresholds.apply_thresholds(proba, t)[0] == 1


def test_tuning_improves_macro_f1_on_an_imbalanced_set():
    rng = np.random.default_rng(0)
    n = 2000
    y = np.zeros(n, dtype=np.int64)
    y[: int(n * 0.03)] = 1
    rng.shuffle(y)

    # S is genuinely separable but the model is under-confident about it, so
    # argmax misses the whole class -- exactly the real situation.
    proba = np.zeros((n, len(config.AAMI_CLASSES)))
    proba[:, 0] = np.where(y == 1, 0.55, 0.95)
    proba[:, 1] = 1.0 - proba[:, 0]

    before = evaluate.evaluate(y, proba.argmax(axis=1)).macro_f1
    tuned = thresholds.tune_thresholds(y, proba)
    after = evaluate.evaluate(y, thresholds.apply_thresholds(proba, tuned)).macro_f1

    assert after > before


def test_tuning_recovers_a_class_argmax_misses_entirely():
    rng = np.random.default_rng(3)
    n = 1000
    y = np.zeros(n, dtype=np.int64)
    y[:30] = 1
    rng.shuffle(y)

    proba = np.zeros((n, len(config.AAMI_CLASSES)))
    proba[:, 0] = np.where(y == 1, 0.60, 0.97)
    proba[:, 1] = 1.0 - proba[:, 0]

    assert evaluate.evaluate(y, proba.argmax(axis=1)).recall_of("S") == 0.0

    tuned = thresholds.tune_thresholds(y, proba)
    after = evaluate.evaluate(y, thresholds.apply_thresholds(proba, tuned))

    assert after.recall_of("S") > 0.9


def test_tuning_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=500)
    proba = rng.dirichlet(np.ones(5), size=500)

    np.testing.assert_array_equal(
        thresholds.tune_thresholds(y, proba), thresholds.tune_thresholds(y, proba)
    )


def test_absent_class_threshold_stays_at_one():
    """Q is absent from most folds. Tuning must not invent a threshold for it."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, size=400)  # no class 3 or 4
    proba = rng.dirichlet(np.ones(5), size=400)

    tuned = thresholds.tune_thresholds(y, proba)

    assert tuned[3] == 1.0
    assert tuned[4] == 1.0


def test_tuning_never_makes_the_metric_worse():
    """Coordinate ascent starts from all-ones and only accepts improvements."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 5, size=600)
    proba = rng.dirichlet(np.ones(5), size=600)

    before = evaluate.evaluate(y, proba.argmax(axis=1)).macro_f1
    tuned = thresholds.tune_thresholds(y, proba)
    after = evaluate.evaluate(y, thresholds.apply_thresholds(proba, tuned)).macro_f1

    assert after >= before


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        thresholds.tune_thresholds(np.zeros(10, dtype=np.int64), np.zeros((9, 5)))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        thresholds.tune_thresholds(np.zeros(0, dtype=np.int64), np.zeros((0, 5)))


def test_thresholds_must_be_positive():
    proba = np.array([[0.6, 0.4, 0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="positive"):
        thresholds.apply_thresholds(proba, np.array([1.0, 0.0, 1.0, 1.0, 1.0]))


def test_wrong_threshold_count_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        thresholds.apply_thresholds(np.zeros((4, 5)), np.ones(3))


# ---------------------------------------------------------------------------
# Q must not drive tuning.
#
# Caught by the Phase 1d smoke run: tuning on pooled DS1 out-of-fold
# predictions produced a threshold of 0.05 for Q -- fitted to 8 beats. Q is
# present in DS1, so "skip absent classes" was not enough of a guard.
# ---------------------------------------------------------------------------


def make_proba_with_a_tiny_class(n=3000, n_q=8, seed=11):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int64)
    y[:n_q] = 4  # Q
    y[n_q : n_q + 90] = 1  # S
    rng.shuffle(y)

    proba = rng.dirichlet(np.ones(5) * 0.5, size=n)
    # Give Q a faint, noisy signal -- enough for a grid search to latch onto.
    proba[y == 4, 4] += 0.25
    proba /= proba.sum(axis=1, keepdims=True)
    return y, proba


def test_a_class_below_min_support_keeps_its_threshold_at_one():
    y, proba = make_proba_with_a_tiny_class()

    tuned = thresholds.tune_thresholds(y, proba)

    assert tuned[4] == 1.0, f"Q threshold was tuned to {tuned[4]} on 8 beats"


def test_min_support_is_configurable():
    y, proba = make_proba_with_a_tiny_class()

    permissive = thresholds.tune_thresholds(y, proba, min_support=1)

    assert permissive[4] != 1.0  # with the guard off, Q does get tuned


def test_tuning_optimises_the_headline_metric_not_the_all_class_one():
    """Q's F1 must not be what the search is maximising."""
    y, proba = make_proba_with_a_tiny_class()

    tuned = thresholds.tune_thresholds(y, proba)
    scored = evaluate.evaluate(y, thresholds.apply_thresholds(proba, tuned))
    baseline = evaluate.evaluate(y, proba.argmax(axis=1))

    assert scored.macro_f1_of(config.HEADLINE_CLASSES) >= baseline.macro_f1_of(
        config.HEADLINE_CLASSES
    )


def test_objective_classes_can_be_overridden():
    y, proba = make_proba_with_a_tiny_class()

    tuned = thresholds.tune_thresholds(y, proba, objective_classes=("N", "S"))

    assert tuned.shape == (len(config.AAMI_CLASSES),)
