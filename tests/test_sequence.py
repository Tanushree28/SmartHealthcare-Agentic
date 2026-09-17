"""Beat neighbourhoods for the rhythm-context model.

Phase 1d's model saw one beat at a time. "Premature" is only meaningful against
the rhythm around a beat, so the model had to infer the reference from a single
beat's RR features. 3,015 normal beats were called S as a result.

This builds a window of consecutive beats around each beat, so the model can see
the rhythm it is judging against. Windows never cross a record boundary -- two
different patients' beats must never appear in one window.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg import config, dataset, sequence


def make_data(n_per_record=10, records=("901", "902")):
    rng = np.random.default_rng(0)
    total = n_per_record * len(records)
    return dataset.BeatData(
        X=rng.standard_normal((total, config.BEAT_LENGTH)).astype(np.float32),
        rr=rng.standard_normal((total, 6)).astype(np.float32),
        y=np.arange(total, dtype=np.int64) % 5,
        record=np.array(
            [r for r in records for _ in range(n_per_record)], dtype=object
        ),
        n_input=total,
        n_dropped_edge=0,
    )


def test_context_window_has_the_expected_width():
    seq = sequence.make_sequences(make_data(), context=2)

    assert seq.width == 5
    assert seq.neighbours.shape == (20, 5)


def test_the_centre_column_is_the_beat_itself():
    seq = sequence.make_sequences(make_data(), context=2)

    np.testing.assert_array_equal(seq.neighbours[:, 2], np.arange(20))


def test_labels_are_the_centre_beats_labels():
    data = make_data()

    seq = sequence.make_sequences(data, context=2)

    np.testing.assert_array_equal(seq.y, data.y)


def test_windows_never_cross_a_record_boundary():
    """Two patients' beats in one window would be a leakage bug that looks like
    a modelling result."""
    data = make_data(n_per_record=10, records=("901", "902"))

    seq = sequence.make_sequences(data, context=3)

    for row, centre_record in enumerate(data.record):
        neighbour_records = data.record[seq.neighbours[row]]
        assert set(neighbour_records) == {centre_record}


def test_edges_clamp_rather_than_wrap():
    """The first beat of a record has no predecessor; it repeats itself."""
    data = make_data(n_per_record=10, records=("901",))

    seq = sequence.make_sequences(data, context=2)

    assert seq.neighbours[0].tolist() == [0, 0, 0, 1, 2]
    assert seq.neighbours[9].tolist() == [7, 8, 9, 9, 9]


def test_a_record_shorter_than_the_window_still_works():
    data = make_data(n_per_record=2, records=("901",))

    seq = sequence.make_sequences(data, context=3)

    assert seq.neighbours.shape == (2, 7)
    assert set(seq.neighbours.flatten().tolist()) <= {0, 1}


def test_context_zero_reduces_to_single_beats():
    """The ablation: the sequence model with no context is the Phase 1d model."""
    seq = sequence.make_sequences(make_data(), context=0)

    assert seq.width == 1
    np.testing.assert_array_equal(seq.neighbours[:, 0], np.arange(20))


def test_negative_context_is_rejected():
    with pytest.raises(ValueError, match="context"):
        sequence.make_sequences(make_data(), context=-1)


def test_gather_builds_the_actual_windows():
    data = make_data(n_per_record=6, records=("901",))
    seq = sequence.make_sequences(data, context=1)

    beats, rr = seq.gather(np.array([3]))

    assert beats.shape == (1, 3, config.BEAT_LENGTH)
    assert rr.shape == (1, 3, 6)
    np.testing.assert_array_equal(beats[0, 1], data.X[3])
    np.testing.assert_array_equal(beats[0, 0], data.X[2])


def test_memory_is_not_multiplied_by_the_window_width():
    """Windows are indices into the original beats, not copies -- a 5-wide
    window over 50,955 beats would otherwise be 306 MB."""
    data = make_data(n_per_record=100, records=("901",))

    seq = sequence.make_sequences(data, context=2)

    assert seq.neighbours.nbytes < data.X.nbytes / 10
