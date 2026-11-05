"""Comparing results honestly.

Phase 1d produced a comparison it could not read. Fold-to-fold variance had a
standard deviation of 0.13 -- fold 3 scored 0.712 while fold 1 scored 0.352 --
against differences between arms of 0.007. Comparing arm means told us nothing.

The *paired* comparison did work: focal beat plain cross-entropy on S recall in
5 folds out of 5, because both arms saw the identical folds and fold difficulty
cancelled. That is the difference between an unreadable result and a clear one,
and it comes from the analysis, not from more training.

So: compare arms fold-by-fold, report how often one wins rather than by how much
on average, and put an interval on any single-draw number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config

#: Default bootstrap resamples. 2,000 is enough for a stable 95% interval and
#: cheap on arrays of this size.
DEFAULT_N_BOOT: int = 2000

#: Default interval width.
DEFAULT_ALPHA: float = 0.05


@dataclass(frozen=True)
class PairedResult:
    """One arm against another, fold by fold."""

    n: int
    wins: int
    mean_difference: float
    min_difference: float
    max_difference: float
    differences: tuple[float, ...]

    @property
    def consistent(self) -> bool:
        """True when the treatment wins on every fold.

        A consistent win on 5 folds is worth more than a larger mean difference
        that comes from one lucky fold -- the latter is what fold variance
        produces for free.
        """
        return self.wins == self.n

    def to_dict(self) -> dict:
        return {
            "n_folds": self.n,
            "wins": self.wins,
            "consistent": self.consistent,
            "mean_difference": round(self.mean_difference, 4),
            "min_difference": round(self.min_difference, 4),
            "max_difference": round(self.max_difference, 4),
            "differences": [round(d, 4) for d in self.differences],
        }

    def __str__(self) -> str:
        verdict = "consistent" if self.consistent else "mixed"
        return (
            f"{self.wins}/{self.n} folds  "
            f"mean {self.mean_difference:+.3f}  "
            f"range [{self.min_difference:+.3f}, {self.max_difference:+.3f}]  "
            f"({verdict})"
        )


def paired_comparison(baseline, treatment) -> PairedResult:
    """Compare two arms on the folds they share.

    Both inputs must be per-fold scores in the same fold order, from runs that
    differ in exactly one setting. Anything else makes the pairing meaningless.
    """
    base = np.asarray(baseline, dtype=np.float64)
    treat = np.asarray(treatment, dtype=np.float64)

    if base.shape != treat.shape:
        raise ValueError(
            f"paired scores must be the same length: {base.shape} vs {treat.shape}"
        )
    if base.size == 0:
        raise ValueError("cannot compare empty score lists")

    differences = treat - base

    return PairedResult(
        n=int(base.size),
        wins=int((differences > 0).sum()),
        mean_difference=float(differences.mean()),
        min_difference=float(differences.min()),
        max_difference=float(differences.max()),
        differences=tuple(float(d) for d in differences),
    )


def bootstrap_ci(
    values,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = config.PHASE_1D_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``."""
    sample = np.asarray(values, dtype=np.float64)

    if sample.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sample.size, size=(n_boot, sample.size))
    means = sample[draws].mean(axis=1)

    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def metric_ci(
    y_true,
    y_pred,
    metric: str = "macro_f1_headline",
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = 500,
    seed: int = config.PHASE_1D_SEED,
) -> tuple[float, float]:
    """Bootstrap interval for a metric on a single test set.

    DS2 is one draw of roughly 49,646 beats. A point estimate says nothing about
    how much that draw could have differed; this resamples beats to find out.

    Beats are resampled, not patients, so this measures sampling noise within
    the test set -- **not** how the model would do on different patients, which
    is a larger and separate uncertainty.
    """
    from . import evaluate  # local import: evaluate imports sklearn

    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)

    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot bootstrap an empty test set")

    rng = np.random.default_rng(seed)
    scores: list[float] = []

    for _ in range(n_boot):
        index = rng.integers(0, true.size, size=true.size)
        scored = evaluate.evaluate(true[index], pred[index])
        scores.append(_read_metric(scored, metric))

    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def _read_metric(scored, metric: str) -> float:
    if metric == "macro_f1_headline":
        return scored.macro_f1_of(config.HEADLINE_CLASSES)
    if metric == "macro_f1":
        return scored.macro_f1
    if metric == "macro_recall":
        return scored.macro_recall
    if metric.endswith("_recall"):
        name = metric[: -len("_recall")].upper()
        return scored.recall_of(name)
    raise ValueError(f"unknown metric {metric!r}")
