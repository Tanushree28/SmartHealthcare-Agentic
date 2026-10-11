"""Averaging fold models.

Five fold models already exist after any cross-validation run. Averaging their
probabilities usually gains a little accuracy and, more usefully here, cuts the
variance that made Phase 1d's arms impossible to rank.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg import ensemble


def test_averaging_identical_members_changes_nothing():
    proba = np.array([[0.7, 0.2, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0]])

    np.testing.assert_allclose(ensemble.average_proba([proba, proba, proba]), proba)


def test_the_average_still_sums_to_one():
    rng = np.random.default_rng(0)
    members = [rng.dirichlet(np.ones(5), size=20) for _ in range(5)]

    averaged = ensemble.average_proba(members)

    np.testing.assert_allclose(averaged.sum(axis=1), 1.0, atol=1e-9)


def test_averaging_cancels_uncorrelated_member_errors():
    """The reason to ensemble at all."""
    truth = np.zeros(300, dtype=np.int64)
    rng = np.random.default_rng(1)

    members = []
    for m in range(5):
        proba = np.full((300, 5), 0.02)
        proba[:, 0] = 0.92
        # each member is confidently wrong on a different tenth of the beats
        wrong = rng.choice(300, size=30, replace=False)
        proba[wrong, 0] = 0.02
        proba[wrong, 1] = 0.92
        members.append(proba / proba.sum(axis=1, keepdims=True))

    member_acc = np.mean([(p.argmax(axis=1) == truth).mean() for p in members])
    ens_acc = (ensemble.average_proba(members).argmax(axis=1) == truth).mean()

    assert ens_acc > member_acc


def test_weights_are_honoured():
    a = np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (4, 1))
    b = np.tile([0.0, 1.0, 0.0, 0.0, 0.0], (4, 1))

    averaged = ensemble.average_proba([a, b], weights=[3.0, 1.0])

    assert averaged[0, 0] == pytest.approx(0.75)
    assert averaged[0, 1] == pytest.approx(0.25)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        ensemble.average_proba([np.zeros((4, 5)), np.zeros((3, 5))])


def test_no_members_is_rejected():
    with pytest.raises(ValueError, match="no members"):
        ensemble.average_proba([])


def test_wrong_number_of_weights_is_rejected():
    with pytest.raises(ValueError, match="weights"):
        ensemble.average_proba([np.zeros((2, 5))] * 3, weights=[1.0, 1.0])


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        ensemble.average_proba([np.zeros((2, 5))] * 2, weights=[1.0, -1.0])
