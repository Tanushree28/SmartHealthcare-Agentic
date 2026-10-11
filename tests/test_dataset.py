"""Beat + RR pairing.

Offline: the fixtures write real ``.npz`` files into ``tmp_path`` in exactly the
layout ``pipeline.run`` produces, so the loading path is genuinely exercised
without the real dataset present.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg import config, dataset


def write_beats(directory, record, n_beats, rr_samples=360, labels=None):
    """Write one record's .npz in the layout pipeline.run produces."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(record)) % 2**32)
    X = rng.standard_normal((n_beats, config.BEAT_LENGTH)).astype(np.float32)
    y = (
        np.asarray(labels, dtype=np.int8)
        if labels is not None
        else np.zeros(n_beats, dtype=np.int8)
    )
    r_peaks = (200 + np.arange(n_beats) * rr_samples).astype(np.int64)
    np.savez(directory / f"{record}.npz", X=X, y=y, r_peaks=r_peaks)
    return X, y, r_peaks


def test_load_record_drops_first_and_last_beat(tmp_path):
    write_beats(tmp_path, "901", n_beats=10)

    data = dataset.load_record(tmp_path, "901")

    assert data.n_input == 10
    assert data.n_kept == 8  # both edge beats lack a neighbour
    assert data.n_dropped_edge == 2
    assert data.reconciles()


def test_load_record_keeps_x_rr_and_y_aligned(tmp_path):
    X, y, _ = write_beats(tmp_path, "901", n_beats=10, labels=list(range(5)) * 2)

    data = dataset.load_record(tmp_path, "901")

    assert data.X.shape == (8, config.BEAT_LENGTH)
    assert data.rr.shape == (8, 6)
    assert data.y.shape == (8,)
    # Row i of the output must be input row i+1 -- the first beat was dropped.
    np.testing.assert_array_equal(data.X[0], X[1])
    np.testing.assert_array_equal(data.y, y[1:9])


def test_rr_values_are_physiological(tmp_path):
    # 360 samples between peaks at 360 Hz == exactly 1.0 s.
    write_beats(tmp_path, "901", n_beats=10, rr_samples=360)

    data = dataset.load_record(tmp_path, "901")

    assert data.rr[:, 0] == pytest.approx(1.0)  # pre_rr
    assert data.rr[:, 4] == pytest.approx(1.0)  # pre_rr_ratio


def test_load_records_concatenates_and_tags_the_record(tmp_path):
    write_beats(tmp_path, "901", n_beats=10)
    write_beats(tmp_path, "902", n_beats=6)

    data = dataset.load_records(tmp_path, ["901", "902"])

    assert data.n_kept == 8 + 4
    assert set(data.record.tolist()) == {"901", "902"}
    assert (data.record[:8] == "901").all()


def test_load_records_rejects_a_duplicate_record(tmp_path):
    write_beats(tmp_path, "901", n_beats=10)

    with pytest.raises(ValueError, match="duplicate"):
        dataset.load_records(tmp_path, ["901", "901"])


def test_load_records_rejects_an_empty_record_list(tmp_path):
    with pytest.raises(ValueError, match="no records"):
        dataset.load_records(tmp_path, [])


def test_load_record_rejects_a_beat_length_mismatch(tmp_path):
    np.savez(
        tmp_path / "903.npz",
        X=np.zeros((5, 128), dtype=np.float32),
        y=np.zeros(5, dtype=np.int8),
        r_peaks=(200 + np.arange(5) * 360).astype(np.int64),
    )

    with pytest.raises(ValueError, match="beat length"):
        dataset.load_record(tmp_path, "903")


def test_missing_record_names_the_command_that_produces_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess_mitbih.py"):
        dataset.load_record(tmp_path, "999")


def test_class_counts_covers_every_aami_class(tmp_path):
    write_beats(tmp_path, "901", n_beats=10, labels=[0, 0, 1, 2, 0, 0, 3, 0, 0, 0])

    counts = dataset.class_counts(dataset.load_record(tmp_path, "901"))

    assert counts.shape == (len(config.AAMI_CLASSES),)
    assert counts.sum() == 8
    assert counts[4] == 0  # Q absent, reported as zero not missing


def test_labels_are_widened_to_int64(tmp_path):
    """Saved as int8; torch's cross-entropy wants int64 targets."""
    write_beats(tmp_path, "901", n_beats=10, labels=[0, 1] * 5)

    data = dataset.load_record(tmp_path, "901")

    assert data.y.dtype == np.int64


def test_robust_rr_is_off_by_default(tmp_path):
    """Phase 1b and 1d results must stay reproducible."""
    write_beats(tmp_path, "901", n_beats=30)

    default = dataset.load_record(tmp_path, "901")
    classic = dataset.load_record(tmp_path, "901", robust_rr=False)

    np.testing.assert_array_equal(default.rr, classic.rr)


def test_robust_rr_changes_the_rhythm_reference_on_an_irregular_record(tmp_path):
    directory = tmp_path
    rng = np.random.default_rng(0)
    intervals = np.full(80, 360)
    intervals[30:50] = 150  # an ectopic run
    r_peaks = np.concatenate([[200], 200 + np.cumsum(intervals)]).astype(np.int64)
    n = r_peaks.size
    np.savez(
        directory / "904.npz",
        X=rng.standard_normal((n, config.BEAT_LENGTH)).astype(np.float32),
        y=np.zeros(n, dtype=np.int8),
        r_peaks=r_peaks,
    )

    classic = dataset.load_record(directory, "904", robust_rr=False)
    robust = dataset.load_record(directory, "904", robust_rr=True)

    # column 3 is the rhythm reference the ratios are built from
    assert not np.allclose(classic.rr[:, 3], robust.rr[:, 3])


def test_robust_rr_propagates_through_load_records(tmp_path):
    write_beats(tmp_path, "901", n_beats=30)
    write_beats(tmp_path, "902", n_beats=30)

    both = dataset.load_records(tmp_path, ["901", "902"], robust_rr=True)

    assert both.n_kept == 56
