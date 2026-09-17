"""Saving and reloading trained weights.

A bare ``state_dict`` is not enough to rebuild this model: ``use_rr`` changes
the architecture, so a waveform-only checkpoint will not load into a hybrid.
The checkpoint therefore carries the settings needed to reconstruct the model,
and ``load_checkpoint`` does the reconstructing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ecg import config
from ecg.nn import checkpoint as ckpt
from ecg.nn import model as model_module


def test_round_trip_reproduces_the_same_predictions(tmp_path):
    torch.manual_seed(config.PHASE_1D_SEED)
    net = model_module.HybridBeatNet().eval()
    beat = torch.randn(4, 1, config.BEAT_LENGTH)
    rr = torch.randn(4, 6)
    before = net(beat, rr)

    path = ckpt.save_checkpoint(
        tmp_path / "arm.pt", net.state_dict(), arm="weighted_ce", use_rr=True
    )
    restored, meta = ckpt.load_checkpoint(path)

    torch.testing.assert_close(restored.eval()(beat, rr), before)
    assert meta["arm"] == "weighted_ce"


def test_waveform_only_checkpoint_rebuilds_a_waveform_only_model(tmp_path):
    net = model_module.HybridBeatNet(use_rr=False)

    path = ckpt.save_checkpoint(
        tmp_path / "wf.pt", net.state_dict(), arm="waveform_only", use_rr=False
    )
    restored, meta = ckpt.load_checkpoint(path)

    assert restored.use_rr is False
    assert meta["use_rr"] is False


def test_checkpoint_records_provenance(tmp_path):
    net = model_module.HybridBeatNet()

    path = ckpt.save_checkpoint(
        tmp_path / "a.pt",
        net.state_dict(),
        arm="focal",
        use_rr=True,
        extra={"device": "cuda", "epochs": 17},
    )
    _, meta = ckpt.load_checkpoint(path)

    assert meta["device"] == "cuda"
    assert meta["epochs"] == 17
    assert "torch_version" in meta
    assert meta["classes"] == list(config.AAMI_CLASSES)


def test_checkpoint_loads_onto_cpu_without_a_gpu(tmp_path):
    """A checkpoint trained on Colab must open on a laptop."""
    net = model_module.HybridBeatNet()
    path = ckpt.save_checkpoint(tmp_path / "a.pt", net.state_dict(), "plain_ce", True)

    restored, _ = ckpt.load_checkpoint(path)

    assert all(p.device.type == "cpu" for p in restored.parameters())


def test_a_mismatched_checkpoint_fails_loudly(tmp_path):
    """Loading waveform-only weights as a hybrid must not silently half-work."""
    hybrid = model_module.HybridBeatNet(use_rr=True)
    path = ckpt.save_checkpoint(
        tmp_path / "bad.pt", hybrid.state_dict(), "x", use_rr=False
    )

    with pytest.raises(RuntimeError):
        ckpt.load_checkpoint(path)
