"""Per-class decision thresholds.

``argmax`` over a softmax is a decision rule, and on an imbalanced problem it is
usually the wrong one: a class that is 2% of the data has to overcome a prior
that argmax takes at face value. Dividing each class's probability by a
per-class threshold before the argmax moves that boundary.

This costs no retraining -- it re-scores probabilities a model has already
produced. Given that a single threshold on ``pre_rr_ratio`` separates S at 0.972
recall while the Phase 1b model's argmax reached 0.271, this is the cheapest
lever in the phase.

**Thresholds are tuned on pooled out-of-fold predictions only.** Tuning them on
the test split would be selecting on DS2, which the spec forbids.
"""

from __future__ import annotations

import numpy as np

from . import config, evaluate

#: Candidate thresholds. Below 1.0 makes a class easier to predict; above,
#: harder. Weighted toward the low end because the useful range for a 2% class
#: is well under 1.
DEFAULT_GRID: tuple[float, ...] = (
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.70,
    1.00,
    1.50,
    2.00,
    3.00,
)


def apply_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Predict by ``argmax(proba / thresholds)``."""
    p = np.asarray(proba, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)

    if p.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {p.shape}")
    if t.shape != (p.shape[1],):
        raise ValueError(
            f"shape mismatch: {p.shape[1]} classes but {t.shape} thresholds"
        )
    if np.any(t <= 0):
        raise ValueError(f"thresholds must be positive, got {t.tolist()}")

    return (p / t).argmax(axis=1).astype(np.int64)


#: A class needs at least this many true examples before a threshold is fitted
#: to it. Q has 8 beats in all of DS1; a threshold fitted to 8 examples is
#: fitted to noise. Caught by the Phase 1d smoke run, which tuned Q to 0.05.
DEFAULT_MIN_SUPPORT: int = 25


def tune_thresholds(
    y_true,
    proba: np.ndarray,
    grid: tuple[float, ...] = DEFAULT_GRID,
    rounds: int = 3,
    classes: tuple[str, ...] = config.AAMI_CLASSES,
    objective_classes: tuple[str, ...] = config.HEADLINE_CLASSES,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> np.ndarray:
    """Coordinate ascent on macro-F1 over per-class thresholds.

    One class at a time, the best grid value is chosen with the others held
    fixed; ``rounds`` passes let earlier choices be revised once later ones have
    moved. Deterministic: the grid is fixed and ties keep the incumbent.

    Starts from all-ones -- plain argmax -- and only accepts strict
    improvements, so the result can never score worse than argmax on the data it
    was tuned on.

    Two guards keep the 8-beat Q class from steering the result:

    * the search maximises macro-F1 over ``objective_classes`` -- N/S/V/F by
      default -- so Q's own F1 is not what is being optimised;
    * a class with fewer than ``min_support`` true examples keeps a threshold of
      1.0, so no threshold is ever fitted to a handful of beats.

    Without the first guard the Phase 1d smoke run tuned Q to 0.05 on 8 beats.
    """
    true = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)

    if p.ndim != 2 or true.shape[0] != p.shape[0]:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs proba {p.shape}")
    if true.size == 0:
        raise ValueError("cannot tune on an empty set")

    if min_support < 1:
        raise ValueError(f"min_support must be >= 1, got {min_support}")

    n_classes = p.shape[1]
    tunable = np.bincount(true, minlength=n_classes) >= min_support

    def score_of(thresholds_: np.ndarray) -> float:
        return evaluate.evaluate(
            true, apply_thresholds(p, thresholds_), classes=classes
        ).macro_f1_of(objective_classes)

    best = np.ones(n_classes, dtype=np.float64)
    best_score = score_of(best)

    for _ in range(rounds):
        improved = False
        for k in range(n_classes):
            if not tunable[k]:
                continue
            for candidate in grid:
                trial = best.copy()
                trial[k] = candidate
                score = score_of(trial)
                if score > best_score:
                    best_score, best, improved = score, trial, True
        if not improved:
            break

    return best
