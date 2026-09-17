"""Do the predicted probabilities mean what they say?

Nothing in this project has ever asked. ``evaluate.py`` takes hard labels,
``thresholds.py`` takes ``argmax(proba / t)`` -- the probabilities are produced
and immediately collapsed. That was defensible while the deliverable was a
ranked table of models. It stops being defensible the moment something
downstream *reads* a probability and reasons with it, because a number that is
consumed rather than argmaxed has to be right, not merely well-ordered.

**There is a specific reason to expect these are wrong.** The Phase 1d winner
trains with ``losses.inverse_frequency_weights(cap=10.0)``. Weighting class ``c``
by ``w_c`` teaches the network a tilted posterior, roughly
``p~(c|x) proportional to w_c * p(c|x)``. So ``P(S|beat) = 0.5`` out of that model does not
mean half of such beats are S. ``prior_drift`` measures exactly this: it compares
each class's mean predicted probability against how often the class actually
occurs. Calibrated in aggregate means they match.

**This module measures; it does not fit.** Fitting a correction (temperature or
vector scaling) is the next step and carries a constraint this one does not: it
may only see pooled out-of-fold predictions, never DS2. Keeping measurement free
of that constraint means these functions can be pointed at DS2 without anyone
having to remember whether that was allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config

#: Floor for probabilities before taking a logarithm. float64 softmax output
#: reaches exact 0.0 for a confidently-rejected class, and log(0) is -inf, which
#: would make one beat dominate the whole NLL.
LOG_EPS: float = 1e-12

#: Bin counts for ECE. Two, deliberately. ECE is bin-sensitive -- a single
#: scalar at a single bin count is easy to flatter by choosing the count -- so
#: the honest report gives more than one and shows the curve behind them.
DEFAULT_BIN_COUNTS: tuple[int, ...] = (10, 15)


def _check(y_true: np.ndarray, proba: np.ndarray):
    true = np.asarray(y_true)
    p = np.asarray(proba, dtype=np.float64)

    if p.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {p.shape}")
    if true.shape[0] != p.shape[0]:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs proba {p.shape}")
    if p.shape[0] == 0:
        raise ValueError("no beats to score")

    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        worst = float(np.abs(row_sums - 1.0).max())
        raise ValueError(
            f"rows of proba must sum to 1; worst row is off by {worst:.2e}. "
            f"An unnormalised score array scored as probabilities would make "
            f"every calibration number here meaningless."
        )
    return true, p


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot truth.

    Lower is better; 0.0 is perfect. Unlike NLL it is bounded and does not
    explode on a single confident mistake, which makes it the more stable of the
    two summary numbers when comparing arms.
    """
    true, p = _check(y_true, proba)
    onehot = np.zeros_like(p)
    onehot[np.arange(true.size), true] = 1.0
    return float(np.sum((p - onehot) ** 2, axis=1).mean())


def negative_log_likelihood(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean negative log probability assigned to the true class.

    This is the quantity temperature scaling minimises, so it is the one to
    watch when a correction is fitted later.
    """
    true, p = _check(y_true, proba)
    picked = p[np.arange(true.size), true]
    return float(-np.log(np.maximum(picked, LOG_EPS)).mean())


@dataclass(frozen=True)
class PriorDrift:
    """How far a class's average predicted probability sits from its real rate."""

    name: str
    mean_predicted: float  # average of P(class | beat) over all beats
    observed_rate: float  # how often the class actually occurs
    support: int

    @property
    def ratio(self) -> float:
        """``mean_predicted / observed_rate``. 1.0 is unbiased.

        Above 1.0 means the model systematically overstates this class. A model
        trained with a weight cap of 10 on a 2% class can land near 10.
        """
        if self.observed_rate <= 0:
            return float("nan")
        return self.mean_predicted / self.observed_rate


def prior_drift(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: tuple[str, ...] = config.AAMI_CLASSES,
) -> tuple[PriorDrift, ...]:
    """Mean predicted probability against observed frequency, per class.

    The most direct test of whether loss weighting distorted the output
    distribution, and the cheapest to read: a calibrated model's average
    predicted probability for a class equals how often that class occurs.
    """
    true, p = _check(y_true, proba)
    n = true.size

    return tuple(
        PriorDrift(
            name=name,
            mean_predicted=float(p[:, i].mean()),
            observed_rate=float((true == i).sum() / n),
            support=int((true == i).sum()),
        )
        for i, name in enumerate(classes)
    )


@dataclass(frozen=True)
class ReliabilityCurve:
    """Observed frequency against predicted confidence, in equal-width bins.

    ``counts`` is carried alongside because empty and near-empty bins are the
    usual way a reliability diagram misleads: a bin holding four beats plots
    at 0.0 or 1.0 and looks like catastrophic miscalibration.
    """

    name: str
    edges: np.ndarray  # (n_bins + 1,)
    mean_predicted: np.ndarray  # (n_bins,) nan where the bin is empty
    observed_rate: np.ndarray  # (n_bins,) nan where the bin is empty
    counts: np.ndarray  # (n_bins,)

    @property
    def n_binned(self) -> int:
        return int(self.counts.sum())

    def gap(self) -> np.ndarray:
        """Signed ``observed - predicted`` per bin. Positive = underconfident."""
        return self.observed_rate - self.mean_predicted


def reliability_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    index: int,
    n_bins: int = 10,
    name: str | None = None,
) -> ReliabilityCurve:
    """One-vs-rest reliability for a single class.

    Class-specific rather than top-label, because the question here is whether
    ``P(S|beat)`` can be trusted on its own -- not whether the winning class's
    confidence is trustworthy.
    """
    true, p = _check(y_true, proba)
    if not 0 <= index < p.shape[1]:
        raise IndexError(f"class index {index} out of range for {p.shape[1]} columns")
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    scores = p[:, index]
    hit = (true == index).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed so the last bin includes 1.0 exactly.
    slot = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, n_bins - 1)

    counts = np.bincount(slot, minlength=n_bins).astype(np.int64)
    pred_sum = np.bincount(slot, weights=scores, minlength=n_bins)
    hit_sum = np.bincount(slot, weights=hit, minlength=n_bins)

    filled = counts > 0
    mean_pred = np.full(n_bins, np.nan)
    observed = np.full(n_bins, np.nan)
    mean_pred[filled] = pred_sum[filled] / counts[filled]
    observed[filled] = hit_sum[filled] / counts[filled]

    return ReliabilityCurve(
        name=name if name is not None else config.AAMI_CLASSES[index],
        edges=edges,
        mean_predicted=mean_pred,
        observed_rate=observed,
        counts=counts,
    )


def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Top-label ECE: how far the winning class's confidence is from its accuracy.

    Weighted by bin population, so sparsely-populated extreme bins cannot
    dominate. This is the number most papers quote; ``classwise_calibration_error``
    is the one that matters for a consumer reading a specific class's probability.
    """
    true, p = _check(y_true, proba)
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    slot = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    counts = np.bincount(slot, minlength=n_bins).astype(np.float64)
    conf_sum = np.bincount(slot, weights=confidence, minlength=n_bins)
    acc_sum = np.bincount(slot, weights=correct, minlength=n_bins)

    filled = counts > 0
    gap = np.abs(acc_sum[filled] - conf_sum[filled])
    return float(gap.sum() / true.size)


def classwise_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
    classes: tuple[str, ...] = config.AAMI_CLASSES,
    names: tuple[str, ...] = config.HEADLINE_CLASSES,
) -> float:
    """Mean one-vs-rest ECE over the named classes.

    Averaged over ``HEADLINE_CLASSES`` by default. Q's 7 DS2 beats put every one
    of them in the lowest bin, which yields a flattering near-zero ECE that says
    nothing about the model -- exactly the reason the F1 headline drops Q too.
    """
    true, p = _check(y_true, proba)
    wanted = [classes.index(n) for n in names]

    errors = []
    for i in wanted:
        curve = reliability_curve(true, p, i, n_bins=n_bins, name=classes[i])
        filled = curve.counts > 0
        gap = np.abs(curve.observed_rate[filled] - curve.mean_predicted[filled])
        errors.append(float((gap * curve.counts[filled]).sum() / true.size))

    return float(np.mean(errors))


def risk_coverage_curve(y_true: np.ndarray, proba: np.ndarray, n_points: int = 100):
    """Accuracy against the fraction of beats kept, most-confident first.

    The curve a consumer that can abstain actually cares about: if the agent
    hands the least-confident beats to a human, how good is what it keeps? A
    well-ordered confidence produces accuracy that falls monotonically as
    coverage grows. A flat curve means confidence carries no information and
    abstention buys nothing.

    Returns ``(coverage, accuracy)``, both ascending in coverage.
    """
    true, p = _check(y_true, proba)
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")

    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == true).astype(np.float64)

    order = np.argsort(-confidence, kind="stable")
    running = np.cumsum(correct[order])
    n = true.size

    # Evenly spaced coverage levels; at least one beat kept at every level.
    kept = np.unique(np.clip((np.linspace(0, 1, n_points)[1:] * n).astype(int), 1, n))
    coverage = kept / n
    accuracy = running[kept - 1] / kept
    return coverage, accuracy
