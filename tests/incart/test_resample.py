"""Moving a signal and its annotation positions between sampling rates.

INCART is recorded at 257 Hz and the networks only accept 360 Hz. If the
signal and the beat marks move differently, every beat window is off-centre
and the score drops for a reason that has nothing to do with the model.
"""

from __future__ import annotations

import numpy as np

from incart import resample
from tests.mit_bih.synthetic import make_ecg


def test_spike_lands_on_the_moved_position():
    sig, peaks = make_ecg(n_beats=10, fs=257, rr_samples=257)

    out = resample.resample_signal(sig, 257, 360)

    for p in peaks:
        expected = round(int(p) * 360 / 257)
        lo, hi = expected - 10, expected + 11
        assert abs(lo + int(np.argmax(out[lo:hi])) - expected) <= 1
    assert abs(out.size - round(sig.size * 360 / 257)) <= 1


def test_positions_move_and_out_of_range_ones_are_dropped():
    moved = resample.move_positions(np.array([0, 257, 1000]), 257, 360, n_out=1000)

    assert moved.samples.tolist() == [0, 360]  # 1000 -> 1401, past the end
    assert moved.keep.tolist() == [True, True, False]
    assert moved.n_dropped == 1


def test_rounding_is_half_up_not_bankers():
    # 180 * 257 / 360 = 128.5 exactly; banker's rounding would give 128.
    moved = resample.move_positions(np.array([180]), 360, 257, n_out=10**6)

    assert moved.samples.tolist() == [129]


def test_round_trip_keeps_positions_within_one_sample():
    sig, peaks = make_ecg(n_beats=20, fs=360)

    out, moved = resample.round_trip(sig, peaks, 360, 257)

    assert moved.keep.all()
    assert np.abs(moved.samples - peaks).max() <= 1
    assert abs(out.size - sig.size) <= 2


def test_round_trip_keep_mask_refers_to_the_original_positions():
    sig, peaks = make_ecg(n_beats=5, fs=360)
    marks = np.append(peaks, sig.size + 500)  # one mark past the end

    _, moved = resample.round_trip(sig, marks, 360, 257)

    assert moved.keep.tolist() == [True] * 5 + [False]
    assert moved.n_dropped == 1
