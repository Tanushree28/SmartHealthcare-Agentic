"""Training loop: smoke, determinism, early stopping, checkpoint round-trip."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg import config, dataset
from ecg.nn import model as model_module
from ecg.nn import train


def make_data(n=200, seed=0, s_fraction=0.2):
    """Small dataset where S is marked by a short pre_rr and nothing else."""
    rng = np.random.default_rng(seed)
    n_s = int(n * s_fraction)

    y = np.zeros(n, dtype=np.int64)
    y[:n_s] = 1
    rng.shuffle(y)

    X = rng.standard_normal((n, config.BEAT_LENGTH)).astype(np.float32)
    rr = np.ones((n, 6), dtype=np.float32) + 0.01 * rng.standard_normal((n, 6))
    rr[y == 1, 0] = 0.5  # premature: short pre_rr
    rr[y == 1, 4] = 0.5  # and a low pre_rr_ratio

    return dataset.BeatData(
        X=X,
        rr=rr.astype(np.float32),
        y=y,
        record=np.full(n, "sim", dtype=object),
        n_input=n,
        n_dropped_edge=0,
    )


def empty_data():
    return dataset.BeatData(
        X=np.zeros((0, config.BEAT_LENGTH), dtype=np.float32),
        rr=np.zeros((0, 6), dtype=np.float32),
        y=np.zeros(0, dtype=np.int64),
        record=np.empty(0, dtype=object),
        n_input=0,
        n_dropped_edge=0,
    )


def test_one_epoch_runs_and_returns_a_result():
    cfg = train.TrainConfig(epochs=1, batch_size=32)

    result = train.train_model(make_data(), make_data(seed=1), "plain", cfg)

    assert result.best_epoch == 1
    assert len(result.history) == 1
    assert 0.0 <= result.best_val_macro_f1 <= 1.0


def test_val_probabilities_have_the_right_shape_and_sum_to_one():
    cfg = train.TrainConfig(epochs=1, batch_size=32)
    val = make_data(n=120, seed=1)

    result = train.train_model(make_data(), val, "plain", cfg)

    assert result.val_proba.shape == (120, len(config.AAMI_CLASSES))
    np.testing.assert_allclose(result.val_proba.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(result.val_y, val.y)


def test_training_is_deterministic_under_a_fixed_seed():
    cfg = train.TrainConfig(epochs=2, batch_size=32, seed=123)

    a = train.train_model(make_data(), make_data(seed=1), "plain", cfg)
    b = train.train_model(make_data(), make_data(seed=1), "plain", cfg)

    np.testing.assert_allclose(a.val_proba, b.val_proba, atol=1e-6)


def test_early_stopping_halts_before_the_epoch_budget():
    cfg = train.TrainConfig(epochs=30, batch_size=32, patience=1)

    result = train.train_model(make_data(), make_data(seed=1), "plain", cfg)

    assert len(result.history) < 30


def test_the_returned_checkpoint_reproduces_the_returned_probabilities():
    cfg = train.TrainConfig(epochs=3, batch_size=32)
    val = make_data(n=120, seed=1)

    result = train.train_model(make_data(), val, "plain", cfg)

    net = model_module.HybridBeatNet(dropout=cfg.dropout)
    net.load_state_dict(result.state_dict)
    reloaded = train.predict_proba(net, val)

    np.testing.assert_allclose(reloaded, result.val_proba, atol=1e-6)


def test_history_records_one_row_per_epoch():
    cfg = train.TrainConfig(epochs=3, batch_size=32, patience=99)

    result = train.train_model(make_data(), make_data(seed=1), "plain", cfg)

    assert [row["epoch"] for row in result.history] == [1, 2, 3]
    assert all("val_macro_f1" in row for row in result.history)


def test_oversampling_changes_the_batches_not_the_validation_set():
    cfg = train.TrainConfig(epochs=1, batch_size=32, oversample=True)
    val = make_data(n=120, seed=1)

    result = train.train_model(make_data(), val, "plain", cfg)

    assert result.val_proba.shape[0] == 120


def test_waveform_only_training_runs():
    cfg = train.TrainConfig(epochs=1, batch_size=32, use_rr=False)

    result = train.train_model(make_data(), make_data(seed=1), "plain", cfg)

    assert result.val_proba.shape[1] == len(config.AAMI_CLASSES)


def test_train_counts_report_the_training_split_only():
    """Phase 1b Problem #6: weights computed over held-out data is leakage."""
    cfg = train.TrainConfig(epochs=1, batch_size=32)
    train_data = make_data(n=200)

    result = train.train_model(train_data, make_data(n=120, seed=1), "plain", cfg)

    assert result.train_counts.sum() == train_data.n_kept


def test_fit_final_returns_a_loadable_state_dict():
    cfg = train.TrainConfig(epochs=50, batch_size=32)

    state = train.fit_final(make_data(), "plain", cfg, epochs=2)

    model_module.HybridBeatNet(dropout=cfg.dropout).load_state_dict(state)


def test_fit_final_rejects_a_zero_epoch_budget():
    with pytest.raises(ValueError, match="epochs"):
        train.fit_final(make_data(), "plain", train.TrainConfig(), epochs=0)


def test_empty_training_data_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        train.train_model(empty_data(), make_data(), "plain", train.TrainConfig(epochs=1))


def test_empty_validation_data_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        train.train_model(make_data(), empty_data(), "plain", train.TrainConfig(epochs=1))


# ---------------------------------------------------------------------------
# Device selection
#
# MPS is ~9x faster than CPU on the real DS1 beats (5.7 s/epoch vs 51 s), which
# is the difference between a 90-minute study and a 15-hour one. It is used only
# if every arm uses it -- CPU and MPS kernels differ enough to shift
# probabilities by ~0.07, so mixing devices across arms would confound the
# comparison. The device is recorded in the run provenance.
# ---------------------------------------------------------------------------


def test_resolve_device_honours_an_explicit_request():
    assert train.resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_falls_back_to_something_real():
    device = train.resolve_device("auto")

    assert device.type in {"cpu", "mps", "cuda"}


def test_cpu_training_is_bit_identical_across_runs():
    cfg = train.TrainConfig(epochs=2, batch_size=32, seed=7, device="cpu")

    a = train.train_model(make_data(n=300), make_data(n=150, seed=1), "weighted", cfg)
    b = train.train_model(make_data(n=300), make_data(n=150, seed=1), "weighted", cfg)

    np.testing.assert_array_equal(a.val_proba, b.val_proba)


def test_checkpoint_comes_back_on_cpu_whatever_the_training_device():
    """A checkpoint must load on a machine with no GPU."""
    cfg = train.TrainConfig(epochs=1, batch_size=32)

    result = train.train_model(make_data(n=300), make_data(n=150, seed=1), "plain", cfg)

    assert all(t.device.type == "cpu" for t in result.state_dict.values())


def test_fit_final_checkpoint_also_comes_back_on_cpu():
    state = train.fit_final(make_data(n=300), "plain", train.TrainConfig(), epochs=1)

    assert all(t.device.type == "cpu" for t in state.values())


def test_predict_proba_moves_the_model_to_the_requested_device():
    """A checkpoint arrives on CPU; scoring it on an accelerator must not
    leave the weights behind. Caught by the Phase 1d smoke run."""
    net = model_module.HybridBeatNet().eval()  # on CPU
    data = make_data(n=64)
    device = train.resolve_device("auto")

    proba = train.predict_proba(net, data, device=device)

    assert proba.shape == (64, len(config.AAMI_CLASSES))
    assert next(net.parameters()).device.type == device.type


def test_predict_proba_defaults_to_the_models_own_device():
    net = model_module.HybridBeatNet().eval()

    proba = train.predict_proba(net, make_data(n=32))

    assert proba.shape == (32, len(config.AAMI_CLASSES))


def test_enable_determinism_sets_the_cudnn_flags():
    """CUDA is not deterministic by default -- measured 4.3e-4 drift on a T4."""
    torch.backends.cudnn.benchmark = True  # the wrong setting, on purpose

    train.enable_determinism()

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_training_still_works_with_determinism_enabled():
    """warn_only=True: an op without a deterministic kernel must not abort a run."""
    cfg = train.TrainConfig(epochs=1, batch_size=32)

    result = train.train_model(make_data(n=200), make_data(n=100, seed=1), "plain", cfg)

    assert result.val_proba.shape[0] == 100
