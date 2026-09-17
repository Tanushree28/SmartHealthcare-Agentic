"""The two-branch network: shape, determinism, and that the RR branch is wired."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ecg import config
from ecg.nn import model as model_module


def make_batch(batch_size=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    beat = torch.randn(batch_size, 1, config.BEAT_LENGTH, generator=generator)
    rr = torch.randn(batch_size, 6, generator=generator)
    return beat, rr


def test_forward_returns_one_logit_per_class():
    net = model_module.HybridBeatNet()
    beat, rr = make_batch()

    assert net(beat, rr).shape == (8, len(config.AAMI_CLASSES))


def test_model_is_deterministic_under_a_fixed_seed():
    beat, rr = make_batch()

    torch.manual_seed(config.PHASE_1D_SEED)
    first = model_module.HybridBeatNet().eval()(beat, rr)
    torch.manual_seed(config.PHASE_1D_SEED)
    second = model_module.HybridBeatNet().eval()(beat, rr)

    torch.testing.assert_close(first, second)


def test_the_rr_branch_actually_changes_the_output():
    """If this fails, the RR branch is decorative and the whole design is void."""
    torch.manual_seed(config.PHASE_1D_SEED)
    net = model_module.HybridBeatNet().eval()
    beat, rr = make_batch()

    assert not torch.allclose(net(beat, rr), net(beat, rr + 5.0))


def test_waveform_only_model_ignores_rr():
    torch.manual_seed(config.PHASE_1D_SEED)
    net = model_module.HybridBeatNet(use_rr=False).eval()
    beat, rr = make_batch()

    torch.testing.assert_close(net(beat, rr), net(beat, rr + 5.0))


def test_waveform_only_model_accepts_rr_of_none():
    net = model_module.HybridBeatNet(use_rr=False).eval()
    beat, _ = make_batch()

    assert net(beat, None).shape == (8, 5)


def test_hybrid_model_rejects_rr_of_none():
    net = model_module.HybridBeatNet(use_rr=True).eval()
    beat, _ = make_batch()

    with pytest.raises(ValueError, match="requires rr"):
        net(beat, None)


def test_parameter_count_is_in_the_designed_range():
    n = model_module.count_parameters(model_module.HybridBeatNet())

    assert 50_000 < n < 200_000, f"expected ~100k parameters, got {n:,}"


def test_waveform_only_model_is_smaller_than_the_hybrid():
    hybrid = model_module.count_parameters(model_module.HybridBeatNet())
    waveform = model_module.count_parameters(model_module.HybridBeatNet(use_rr=False))

    assert waveform < hybrid


def test_batch_size_one_works_in_eval_mode():
    """BatchNorm fails on a single sample in train mode; eval must not."""
    net = model_module.HybridBeatNet().eval()
    beat, rr = make_batch(batch_size=1)

    assert net(beat, rr).shape == (1, 5)


def test_rejects_a_wrong_beat_length():
    net = model_module.HybridBeatNet()

    with pytest.raises(ValueError, match="beat length"):
        net(torch.randn(4, 1, 128), torch.randn(4, 6))


def test_rejects_an_unchannelled_beat():
    net = model_module.HybridBeatNet()

    with pytest.raises(ValueError, match="shape"):
        net(torch.randn(4, config.BEAT_LENGTH), torch.randn(4, 6))


def test_gradients_reach_both_branches():
    """A branch with no gradient is not being trained."""
    net = model_module.HybridBeatNet()
    beat, rr = make_batch()

    net(beat, rr).sum().backward()

    assert net.rr_branch[0].weight.grad is not None
    assert net.rr_branch[0].weight.grad.abs().sum() > 0
    first_conv = net.waveform[0][0]
    assert first_conv.weight.grad is not None
    assert first_conv.weight.grad.abs().sum() > 0
