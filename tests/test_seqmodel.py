"""The rhythm-context model: CNN per beat, BiLSTM across the neighbourhood."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ecg import config
from ecg.nn import seqmodel


def make_batch(batch=4, width=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    beats = torch.randn(batch, width, config.BEAT_LENGTH, generator=g)
    rr = torch.randn(batch, width, 6, generator=g)
    return beats, rr


def test_forward_returns_one_logit_per_class_for_the_centre_beat():
    net = seqmodel.RhythmContextNet()

    assert net(*make_batch()).shape == (4, len(config.AAMI_CLASSES))


def test_it_is_deterministic_under_a_fixed_seed():
    beats, rr = make_batch()

    torch.manual_seed(config.PHASE_1D_SEED)
    first = seqmodel.RhythmContextNet().eval()(beats, rr)
    torch.manual_seed(config.PHASE_1D_SEED)
    second = seqmodel.RhythmContextNet().eval()(beats, rr)

    torch.testing.assert_close(first, second)


def test_neighbouring_beats_change_the_prediction():
    """The whole point: context must actually be used. If this fails, the model
    is a slower version of the Phase 1d one."""
    torch.manual_seed(config.PHASE_1D_SEED)
    net = seqmodel.RhythmContextNet().eval()
    beats, rr = make_batch()

    changed = rr.clone()
    changed[:, 0, :] += 5.0  # disturb only a NEIGHBOUR, never the centre
    changed[:, -1, :] += 5.0

    assert not torch.allclose(net(beats, rr), net(beats, changed))


def test_the_centre_beat_still_matters():
    torch.manual_seed(config.PHASE_1D_SEED)
    net = seqmodel.RhythmContextNet().eval()
    beats, rr = make_batch()

    changed = beats.clone()
    changed[:, 2, :] += 5.0  # the centre beat

    assert not torch.allclose(net(beats, rr), net(changed, rr))


def test_it_accepts_any_odd_window_width():
    net = seqmodel.RhythmContextNet()

    for width in (1, 3, 7):
        assert net(*make_batch(width=width)).shape == (4, 5)


def test_width_one_is_accepted_as_the_no_context_ablation():
    net = seqmodel.RhythmContextNet()

    assert net(*make_batch(width=1)).shape == (4, 5)


def test_an_even_window_is_rejected():
    """Without a single centre beat there is no beat being classified."""
    net = seqmodel.RhythmContextNet()

    with pytest.raises(ValueError, match="odd"):
        net(*make_batch(width=4))


def test_wrong_beat_length_is_rejected():
    net = seqmodel.RhythmContextNet()
    beats = torch.randn(4, 5, 128)
    rr = torch.randn(4, 5, 6)

    with pytest.raises(ValueError, match="beat length"):
        net(beats, rr)


def test_parameter_count_stays_modest():
    n = seqmodel.count_parameters(seqmodel.RhythmContextNet())

    assert 50_000 < n < 500_000, f"{n:,} parameters"


def test_gradients_reach_the_lstm():
    net = seqmodel.RhythmContextNet()

    net(*make_batch()).sum().backward()

    grads = [p.grad for p in net.rhythm.parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0
