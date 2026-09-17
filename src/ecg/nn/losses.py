"""Loss functions for the imbalance study.

Arms 0, 1 and 2 of Phase 1d differ only in the object this module builds.
Everything else -- architecture, data, folds, seed, optimizer, epoch budget --
is held identical, which is what makes the comparison attributable.

**Why weights are capped.** ``normalization.json`` for the DS1/DS2 split
suggests a weight of 996.0 for class Q, which has 8 training beats. Applied
literally, those 8 beats would dominate the gradient and destabilise training.
The cap bounds any one class's influence; the cap value itself is selected by
cross-validation, never asserted.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

#: Starting point for the weight cap. Selected properly by cross-validation.
DEFAULT_WEIGHT_CAP: float = 50.0

LOSS_KINDS = ("plain", "weighted", "focal")


def inverse_frequency_weights(
    counts,
    cap: float = DEFAULT_WEIGHT_CAP,
) -> np.ndarray:
    """Per-class weights inversely proportional to frequency, capped.

    A class with no beats gets weight 0.0 rather than infinity -- it cannot
    contribute to the loss, so its weight is arbitrary, and 0.0 is the value
    that keeps the array finite.

    The result is rescaled so the mean weight over present classes is 1.0. That
    keeps the loss on the same scale as the unweighted arm, so arm 0 and arm 1
    can be compared without the learning rate quietly meaning something
    different in each.
    """
    raw = np.asarray(counts, dtype=np.float64)

    if raw.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {raw.shape}")
    if np.any(raw < 0):
        raise ValueError(f"negative class count in {raw.tolist()}")
    if raw.sum() <= 0:
        raise ValueError("no beats in any class")
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    present = raw > 0
    weights = np.zeros_like(raw)

    # total / (n_present * count): the standard "balanced" formula, restricted
    # to classes that are actually present.
    weights[present] = raw.sum() / (present.sum() * raw[present])
    weights = np.minimum(weights, cap)

    mean_present = weights[present].mean()
    if mean_present > 0:
        weights[present] /= mean_present
        # Rescaling can lift a capped weight back above the cap; clip again so
        # the cap is a real bound rather than an intermediate step.
        weights = np.minimum(weights, cap)

    return weights


class FocalLoss(nn.Module):
    """Cross-entropy scaled by ``(1 - p_t) ** gamma``.

    Easy beats -- those the model already gets right with confidence --
    contribute less, so the gradient is dominated by the hard ones. With 89%
    of beats normal, that is most of the dataset.

    At ``gamma=0`` this is exactly cross-entropy. ``test_losses.py`` asserts
    that identity in both the weighted and unweighted case, which is what
    proves the scaling term is right rather than merely plausible.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = float(gamma)
        self.register_buffer(
            "weight", weight if weight is None else weight.to(torch.float32)
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # reduction="none" so the focal term is applied per sample before any
        # averaging -- averaging first would make the term a no-op.
        ce = nn.functional.cross_entropy(
            logits, target, weight=self.weight, reduction="none"
        )

        # p_t recovered from the unweighted log-probability, so the class weight
        # scales the loss without also distorting the focal term.
        log_pt = (
            nn.functional.log_softmax(logits, dim=1)
            .gather(1, target.unsqueeze(1))
            .squeeze(1)
        )
        focal = (1.0 - log_pt.exp()).clamp(min=0.0) ** self.gamma

        loss = focal * ce

        if self.weight is None:
            return loss.mean()

        # Weighted cross-entropy normalises by the sum of the weights, not the
        # sample count. Matching that is what makes the gamma=0 identity hold.
        denominator = self.weight[target].sum()
        if denominator <= 0:
            return loss.sum() * 0.0
        return loss.sum() / denominator


def build_loss(
    kind: str,
    counts,
    cap: float = DEFAULT_WEIGHT_CAP,
    gamma: float = 2.0,
) -> nn.Module:
    """Build the loss for one arm of the study."""
    if kind not in LOSS_KINDS:
        raise ValueError(f"unknown loss {kind!r}; expected one of {list(LOSS_KINDS)}")

    if kind == "plain":
        return nn.CrossEntropyLoss()

    weight = torch.tensor(inverse_frequency_weights(counts, cap), dtype=torch.float32)

    if kind == "weighted":
        return nn.CrossEntropyLoss(weight=weight)
    return FocalLoss(gamma=gamma, weight=weight)
