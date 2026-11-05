"""Averaging the predictions of several models.

Any cross-validation run leaves behind one model per fold. Averaging their
probabilities is close to free -- no retraining -- and does two useful things:
it usually improves accuracy a little, and it reduces run-to-run variance.

Variance is the point here. Phase 1d could not rank three of its arms because
fold-to-fold spread (std 0.13) swamped the differences between them (0.007).
Anything that narrows that spread makes the next comparison readable.
"""

from __future__ import annotations

import numpy as np


def average_proba(members, weights=None) -> np.ndarray:
    """Weighted mean of several probability matrices.

    Each member must be ``(n_beats, n_classes)`` for the same beats in the same
    order. Averaging probabilities rather than voting on labels keeps the
    confidence information that per-class thresholding later depends on.
    """
    stack = [np.asarray(m, dtype=np.float64) for m in members]

    if not stack:
        raise ValueError("no members to average")

    shape = stack[0].shape
    if any(m.shape != shape for m in stack):
        raise ValueError(
            f"all members must have the same shape; got "
            f"{[m.shape for m in stack]}"
        )

    if weights is None:
        coefficients = np.ones(len(stack), dtype=np.float64)
    else:
        coefficients = np.asarray(weights, dtype=np.float64)
        if coefficients.shape != (len(stack),):
            raise ValueError(
                f"expected {len(stack)} weights, got {coefficients.shape}"
            )
        if np.any(coefficients < 0):
            raise ValueError(f"negative weight in {coefficients.tolist()}")
        if coefficients.sum() <= 0:
            raise ValueError("weights sum to zero")

    coefficients = coefficients / coefficients.sum()

    return np.tensordot(coefficients, np.stack(stack), axes=(0, 0))
