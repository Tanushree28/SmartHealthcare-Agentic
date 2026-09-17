"""Loss functions and class weighting."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg.nn import losses


# --- weights ---------------------------------------------------------------


def test_rarer_classes_get_larger_weights():
    w = losses.inverse_frequency_weights([100, 10, 50, 5, 0])

    assert w[1] > w[2] > w[0]
    assert w[3] > w[1]


def test_absent_class_gets_zero_weight_not_infinity():
    """Q has 8 training beats and can be absent from a fold entirely."""
    w = losses.inverse_frequency_weights([100, 10, 50, 5, 0])

    assert w[4] == 0.0
    assert np.isfinite(w).all()


def test_the_cap_is_applied():
    w = losses.inverse_frequency_weights([100_000, 10, 10, 10, 8], cap=50.0)

    assert w.max() <= 50.0


def test_the_cap_bounds_how_far_q_can_dominate_n():
    """The reason the cap exists, measured on the real DS1 training counts.

    ``normalization.json`` suggests a raw weight of 996.0 for Q's 8 beats. What
    matters is not that number on its own -- weights are rescaled to mean 1 --
    but how far it lets 8 beats outweigh N's 35,501. Uncapped, the ratio is the
    raw count ratio, ~4,438x, so the gradient is essentially about Q alone.
    """
    real_counts = [35501, 863, 3080, 388, 8]

    uncapped = losses.inverse_frequency_weights(real_counts, cap=float("inf"))
    capped = losses.inverse_frequency_weights(real_counts, cap=50.0)

    uncapped_ratio = uncapped[4] / uncapped[0]
    capped_ratio = capped[4] / capped[0]

    assert uncapped_ratio == pytest.approx(35501 / 8, rel=1e-6)
    assert capped_ratio < uncapped_ratio / 10
    assert capped[4] == pytest.approx(capped.max())


def test_weights_are_rescaled_to_mean_one_over_present_classes():
    w = losses.inverse_frequency_weights([100, 10, 50, 5, 0])

    assert w[w > 0].mean() == pytest.approx(1.0)


def test_all_equal_counts_give_all_equal_weights():
    w = losses.inverse_frequency_weights([10, 10, 10, 10, 10])

    assert w == pytest.approx(np.ones(5))


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        losses.inverse_frequency_weights([10, -1, 10, 10, 10])


def test_all_zero_counts_are_rejected():
    with pytest.raises(ValueError, match="no beats"):
        losses.inverse_frequency_weights([0, 0, 0, 0, 0])


def test_non_positive_cap_is_rejected():
    with pytest.raises(ValueError, match="cap"):
        losses.inverse_frequency_weights([10, 10, 10, 10, 10], cap=0.0)


# --- focal loss ------------------------------------------------------------


def test_focal_loss_at_gamma_zero_equals_cross_entropy():
    """The identity that proves the implementation is really focal loss."""
    torch.manual_seed(0)
    logits = torch.randn(32, 5)
    target = torch.randint(0, 5, (32,))

    focal = losses.FocalLoss(gamma=0.0)(logits, target)
    plain = torch.nn.CrossEntropyLoss()(logits, target)

    torch.testing.assert_close(focal, plain)


def test_weighted_focal_at_gamma_zero_equals_weighted_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(32, 5)
    target = torch.randint(0, 4, (32,))
    w = torch.tensor([1.0, 4.0, 2.0, 8.0, 0.0])

    focal = losses.FocalLoss(gamma=0.0, weight=w)(logits, target)
    plain = torch.nn.CrossEntropyLoss(weight=w)(logits, target)

    torch.testing.assert_close(focal, plain)


def test_focal_loss_down_weights_confident_correct_predictions():
    confident = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])

    plain = torch.nn.CrossEntropyLoss()(confident, target)
    focal = losses.FocalLoss(gamma=2.0)(confident, target)

    assert focal < plain


def test_focal_loss_barely_touches_a_wrong_confident_prediction():
    """The point of focal loss: hard examples keep their weight."""
    wrong = torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])

    plain = torch.nn.CrossEntropyLoss()(wrong, target)
    focal = losses.FocalLoss(gamma=2.0)(wrong, target)

    assert focal == pytest.approx(float(plain), rel=1e-3)


def test_focal_loss_returns_a_scalar():
    loss = losses.FocalLoss(gamma=2.0)(torch.randn(8, 5), torch.randint(0, 5, (8,)))

    assert loss.ndim == 0


def test_negative_gamma_is_rejected():
    with pytest.raises(ValueError, match="gamma"):
        losses.FocalLoss(gamma=-1.0)


# --- factory ---------------------------------------------------------------


def test_build_loss_plain_ignores_the_counts():
    loss_fn = losses.build_loss("plain", counts=[100, 1, 1, 1, 1])

    assert getattr(loss_fn, "weight", None) is None


def test_build_loss_weighted_carries_the_weights():
    loss_fn = losses.build_loss("weighted", counts=[100, 10, 50, 5, 0])

    assert loss_fn.weight is not None
    assert loss_fn.weight[4] == 0.0


def test_build_loss_focal_is_a_focal_loss():
    assert isinstance(losses.build_loss("focal", counts=[10] * 5), losses.FocalLoss)


def test_build_loss_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown loss"):
        losses.build_loss("hinge", counts=[1, 1, 1, 1, 1])
