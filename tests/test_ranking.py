"""Ranking metrics must separate "badly ordered" from "badly thresholded".

That is the whole reason this module exists, so the tests pin the two
properties that make it readable: a perfect ordering scores 1.0, and an
uninformative one scores the class prevalence rather than 0.0. The second is the
one that matters in practice -- at S's 3.7% prevalence, an average precision of
0.20 sounds like failure and is in fact five times chance.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg import config, ranking


def _proba_from_scores(scores: np.ndarray) -> np.ndarray:
    """Two-column probability matrix where column 1 carries ``scores``."""
    s = np.asarray(scores, dtype=np.float64)
    return np.column_stack([1.0 - s, s])


class TestClassRanking:
    def test_perfect_ordering_scores_one(self):
        y = np.array([0, 0, 0, 1, 1])
        proba = _proba_from_scores([0.1, 0.2, 0.3, 0.9, 0.95])

        result = ranking.class_ranking(y, proba, 1, name="S")

        assert result.average_precision == pytest.approx(1.0)
        assert result.roc_auc == pytest.approx(1.0)
        assert result.support == 2

    def test_uninformative_ordering_scores_prevalence_not_zero(self):
        """The property that makes average precision readable on a rare class."""
        rng = np.random.default_rng(0)
        n, prevalence = 20_000, 0.037

        y = (rng.random(n) < prevalence).astype(np.int64)
        proba = _proba_from_scores(rng.random(n))  # scores independent of y

        result = ranking.class_ranking(y, proba, 1, name="S")

        assert result.average_precision == pytest.approx(y.mean(), abs=0.01)
        assert result.roc_auc == pytest.approx(0.5, abs=0.02)
        assert result.lift_over_chance == pytest.approx(1.0, abs=0.3)

    def test_inverted_ordering_scores_below_chance(self):
        y = np.array([0, 0, 0, 1, 1])
        proba = _proba_from_scores([0.9, 0.8, 0.7, 0.1, 0.05])

        result = ranking.class_ranking(y, proba, 1, name="S")

        assert result.roc_auc == pytest.approx(0.0)
        assert result.average_precision < result.prevalence

    def test_absent_class_is_nan_not_zero(self):
        """Q has 7 beats in DS2 and none in some folds; 0.0 would read as failure."""
        y = np.zeros(10, dtype=np.int64)
        proba = _proba_from_scores(np.full(10, 0.3))

        result = ranking.class_ranking(y, proba, 1, name="Q")

        assert result.support == 0
        assert np.isnan(result.average_precision)
        assert np.isnan(result.roc_auc)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            ranking.class_ranking(np.zeros(5), _proba_from_scores(np.zeros(4)), 1)

    def test_rejects_out_of_range_class(self):
        with pytest.raises(IndexError, match="out of range"):
            ranking.class_ranking(np.zeros(4), _proba_from_scores(np.zeros(4)), 7)


class TestMacroAveragePrecision:
    def test_excludes_q_by_default(self):
        """Q's average precision must not reach the number that ranks models."""
        rankings = tuple(
            ranking.ClassRanking(
                name=name,
                support=10,
                average_precision=0.8 if name != "Q" else 0.0,
                roc_auc=0.9,
                prevalence=0.2,
            )
            for name in config.AAMI_CLASSES
        )

        assert ranking.macro_average_precision(rankings) == pytest.approx(0.8)
        # Averaging all five would drag it to 0.64 -- the failure being prevented.
        assert ranking.macro_average_precision(
            rankings, names=config.AAMI_CLASSES
        ) == pytest.approx(0.64)

    def test_raises_on_missing_class(self):
        rankings = (
            ranking.ClassRanking("N", 10, 0.9, 0.9, 0.5),
        )
        with pytest.raises(KeyError, match="S"):
            ranking.macro_average_precision(rankings)
