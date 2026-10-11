"""Training the rhythm-context model."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg import config, dataset, sequence
from ecg.nn import seqtrain
from ecg.nn import train as beat_train


def make_data(n=240, seed=0, s_fraction=0.2, records=("901", "902")):
    rng = np.random.default_rng(seed)
    n_s = int(n * s_fraction)
    y = np.zeros(n, dtype=np.int64)
    y[:n_s] = 1
    rng.shuffle(y)

    X = rng.standard_normal((n, config.BEAT_LENGTH)).astype(np.float32)
    rr = np.ones((n, 6), dtype=np.float32) + 0.01 * rng.standard_normal((n, 6))
    rr[y == 1, 0] = 0.5
    rr[y == 1, 4] = 0.5

    per = n // len(records)
    record = np.array([r for r in records for _ in range(per)], dtype=object)

    return dataset.BeatData(
        X=X, rr=rr.astype(np.float32), y=y, record=record,
        n_input=n, n_dropped_edge=0,
    )


def seq(n=240, seed=0, context=2):
    return sequence.make_sequences(make_data(n=n, seed=seed), context=context)


def test_one_epoch_runs():
    cfg = beat_train.TrainConfig(epochs=1, batch_size=32)

    result = seqtrain.train_sequence_model(seq(), seq(n=120, seed=1), "plain", cfg)

    assert result.best_epoch == 1
    assert result.val_proba.shape == (120, len(config.AAMI_CLASSES))


def test_probabilities_sum_to_one():
    cfg = beat_train.TrainConfig(epochs=1, batch_size=32)

    result = seqtrain.train_sequence_model(seq(), seq(n=120, seed=1), "plain", cfg)

    np.testing.assert_allclose(result.val_proba.sum(axis=1), 1.0, atol=1e-5)


def test_training_is_deterministic():
    cfg = beat_train.TrainConfig(epochs=2, batch_size=32, seed=99)

    a = seqtrain.train_sequence_model(seq(), seq(n=120, seed=1), "weighted", cfg)
    b = seqtrain.train_sequence_model(seq(), seq(n=120, seed=1), "weighted", cfg)

    np.testing.assert_allclose(a.val_proba, b.val_proba, atol=1e-6)


def test_checkpoint_reproduces_the_probabilities():
    cfg = beat_train.TrainConfig(epochs=2, batch_size=32)
    validation = seq(n=120, seed=1)

    result = seqtrain.train_sequence_model(seq(), validation, "plain", cfg)
    reloaded = seqtrain.predict_proba_sequence(
        seqtrain.build_model(cfg, result.state_dict), validation
    )

    np.testing.assert_allclose(reloaded, result.val_proba, atol=1e-6)


def test_labels_returned_are_the_centre_beats():
    validation = seq(n=120, seed=1)
    cfg = beat_train.TrainConfig(epochs=1, batch_size=32)

    result = seqtrain.train_sequence_model(seq(), validation, "plain", cfg)

    np.testing.assert_array_equal(result.val_y, validation.y)


def test_class_weights_come_from_the_training_split_only():
    cfg = beat_train.TrainConfig(epochs=1, batch_size=32)
    training = seq()

    result = seqtrain.train_sequence_model(training, seq(n=120, seed=1), "weighted", cfg)

    assert result.train_counts.sum() == training.n


def test_empty_training_data_is_rejected():
    cfg = beat_train.TrainConfig(epochs=1)
    empty = sequence.make_sequences(
        dataset.BeatData(
            X=np.zeros((0, config.BEAT_LENGTH), dtype=np.float32),
            rr=np.zeros((0, 6), dtype=np.float32),
            y=np.zeros(0, dtype=np.int64),
            record=np.empty(0, dtype=object),
            n_input=0, n_dropped_edge=0,
        )
    )

    with pytest.raises(ValueError, match="empty"):
        seqtrain.train_sequence_model(empty, seq(), "plain", cfg)
