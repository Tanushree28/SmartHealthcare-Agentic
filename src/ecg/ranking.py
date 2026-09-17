"""Threshold-free ranking quality, one class against the rest.

Every metric in ``evaluate.py`` is computed after ``argmax`` -- it scores a
*decision rule*, not the model underneath it. That conflates two different
failures. A model that orders beats well but sits at the wrong operating point
and a model that cannot order beats at all both look bad through F1, and only
the first one is cheap to fix.

The distinction decides real questions. Phase 1d ranked focal loss (headline
macro-F1 0.545) above plain cross-entropy (0.379) and carried focal forward. But
class weighting shifts the decision boundary by construction, so some of that
gap is a moved threshold rather than a better model. **Average precision answers
which**, because it integrates over every threshold instead of picking one.

**Why average precision and not ROC-AUC.** S is 3.7% of DS2. ROC-AUC's false
positive rate divides by the 44,196 N beats, so a thousand new false alarms move
it by 0.02 and it reads as near-perfect while the precision a reader would
experience collapses. Average precision divides by the beats *called* S, which
is the quantity that actually degrades. ROC-AUC is still reported, because it is
what the literature quotes, but it is not what ranks anything here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn import metrics as skmetrics

from . import config


@dataclass(frozen=True)
class ClassRanking:
    """How well one class is ordered against every other class."""

    name: str
    support: int  # positives for this class in y_true
    average_precision: float  # area under precision-recall; chance = prevalence
    roc_auc: float  # chance = 0.5
    prevalence: float  # the value average_precision would take at chance

    @property
    def lift_over_chance(self) -> float:
        """Average precision divided by what a random ranker would score.

        Reported because 0.41 means something very different for a class at 3.7%
        prevalence than for one at 89%.
        """
        if self.prevalence <= 0:
            return float("nan")
        return self.average_precision / self.prevalence


def _one_vs_rest(y_true: np.ndarray, proba: np.ndarray, index: int):
    """Binary labels and scores for one class against all others."""
    true = np.asarray(y_true)
    p = np.asarray(proba, dtype=np.float64)

    if p.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {p.shape}")
    if true.shape[0] != p.shape[0]:
        raise ValueError(
            f"shape mismatch: y_true {true.shape} vs proba {p.shape}"
        )
    if not 0 <= index < p.shape[1]:
        raise IndexError(f"class index {index} out of range for {p.shape[1]} columns")

    return (true == index).astype(np.int8), p[:, index]


def class_ranking(
    y_true: np.ndarray,
    proba: np.ndarray,
    index: int,
    name: str | None = None,
) -> ClassRanking:
    """Rank quality for one class.

    A class with no positives, or with nothing but positives, gets ``nan``
    rather than a number: average precision is undefined there, and 0.0 would
    read as "the model failed" when the truth is "the test set cannot say".
    Q has 7 beats in DS2 and none at all in some folds, so this is reached.
    """
    binary, scores = _one_vs_rest(y_true, proba, index)
    support = int(binary.sum())
    n = int(binary.size)
    label = name if name is not None else config.AAMI_CLASSES[index]
    prevalence = support / n if n else float("nan")

    if support == 0 or support == n:
        return ClassRanking(
            name=label,
            support=support,
            average_precision=float("nan"),
            roc_auc=float("nan"),
            prevalence=prevalence,
        )

    return ClassRanking(
        name=label,
        support=support,
        average_precision=float(skmetrics.average_precision_score(binary, scores)),
        roc_auc=float(skmetrics.roc_auc_score(binary, scores)),
        prevalence=prevalence,
    )


def rank_all(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: tuple[str, ...] = config.AAMI_CLASSES,
) -> tuple[ClassRanking, ...]:
    """Rank quality for every class, in ``classes`` order."""
    return tuple(
        class_ranking(y_true, proba, i, name)
        for i, name in enumerate(classes)
    )


def macro_average_precision(
    rankings,
    names: tuple[str, ...] = config.HEADLINE_CLASSES,
) -> float:
    """Mean average precision over the named classes.

    Defaults to ``HEADLINE_CLASSES`` for the same reason the F1 headline does:
    Q's 7 DS2 beats make its average precision noise, and averaging noise into
    the number that ranks models lets it move the ranking.
    """
    wanted = {r.name: r for r in rankings}
    missing = [n for n in names if n not in wanted]
    if missing:
        raise KeyError(f"no ranking for {missing}")

    values = [wanted[n].average_precision for n in names]
    return float(np.mean(values))


def pr_curve(y_true: np.ndarray, proba: np.ndarray, index: int):
    """Precision, recall and thresholds for one class -- for plotting.

    ``precision`` and ``recall`` have one more element than ``thresholds``;
    that final point is the (recall=0, precision=1) endpoint sklearn appends
    and it has no threshold behind it.
    """
    binary, scores = _one_vs_rest(y_true, proba, index)
    precision, recall, thresholds = skmetrics.precision_recall_curve(binary, scores)
    return precision, recall, thresholds
