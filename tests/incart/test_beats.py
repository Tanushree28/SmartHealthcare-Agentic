"""Beats at the model rate, and the beat-position check."""

from __future__ import annotations

import numpy as np
import pytest

from incart import evaluate as ev
from incart import io
from mit_bih import config as mit_config
from tests.mit_bih.synthetic import make_ecg, write_record


def _ds2_like_record(directory):
    """A one-lead MLII record at 360 Hz, 20 beats alternating N and V."""
    sig, peaks = make_ecg(n_beats=20, fs=360)
    symbols = ["N" if i % 2 == 0 else "V" for i in range(20)]
    write_record(directory, "s100", {"MLII": sig}, fs=360, ann_samples=peaks, ann_symbols=symbols)
    return io.load_labelled_record(directory, "s100", fs=360,
                                   preference=mit_config.LEAD_PREFERENCE)


def test_marks_on_the_peaks_give_zero_offset(bundle):
    sig, peaks = make_ecg(n_beats=20, fs=360)

    offsets = ev.position_offsets(ev.band_passed(sig, bundle), peaks, 360)

    assert offsets.tolist() == [0] * 20


def test_marks_off_the_peaks_give_the_shift(bundle):
    sig, peaks = make_ecg(n_beats=20, fs=360)
    filtered = ev.band_passed(sig, bundle)

    offsets = ev.position_offsets(filtered, peaks - 5, 360)
    summary = ev.summarise_positions(offsets, ev.peak_amplitudes(filtered, peaks - 5, 360), 360)

    assert offsets.tolist() == [5] * 20
    assert summary["n"] == 20
    assert summary["median_samples"] == 5
    assert summary["median_ms"] == pytest.approx(5 / 360 * 1000)
    assert summary["q1_ms"] == summary["q3_ms"] == pytest.approx(5 / 360 * 1000)


def test_peak_amplitude_is_the_largest_value_near_the_mark(bundle):
    sig, peaks = make_ecg(n_beats=10, fs=360, qrs_amplitude=2.0)
    filtered = ev.band_passed(sig, bundle)

    amps = ev.peak_amplitudes(filtered, peaks - 5, 360)

    assert np.allclose(amps, np.abs(filtered[peaks]))


def test_summary_of_no_beats_has_no_numbers():
    summary = ev.summarise_positions(np.zeros(0), np.zeros(0), 360)

    assert summary["n"] == 0 and summary["median_samples"] is None


def test_refined_marks_that_collide_are_dropped_and_counted(bundle):
    sig, peaks = make_ecg(n_beats=10, fs=360)
    marks = np.sort(np.append(peaks, peaks[3] + 4))  # refines onto peaks[3] too
    labels = np.arange(marks.size, dtype=np.int64) % 4

    new, new_labels, n = ev.refined_marks(ev.band_passed(sig, bundle), marks, labels, 360)

    assert n == 1
    assert np.array_equal(new, peaks)
    assert new_labels.tolist() == np.delete(labels, 4).tolist()


def test_beat_counts_reconcile_from_file_to_classified_beats(incart_record, bundle):
    rec = io.load_labelled_record(incart_record["dir"], "I99")

    beats = ev.record_beats(rec, bundle)

    assert beats.reconciles()
    assert beats.n_annotated == 20 and beats.n_dropped_edge == 2
    assert beats.n_out_of_range == 0 and beats.n_collided == 0
    assert beats.X.shape == (18, 300)
    assert np.array_equal(beats.y, incart_record["labels"][1:-1])


def test_incart_marks_land_on_the_peaks_after_resampling(incart_record, bundle):
    rec = io.load_labelled_record(incart_record["dir"], "I99")

    offsets, _ = ev.record_positions(rec, bundle)

    assert offsets.size == 20 and np.abs(offsets).max() <= 1


def test_round_trip_beats_reconcile(tmp_path, bundle):
    rec = _ds2_like_record(tmp_path)

    beats = ev.record_beats(rec, bundle, round_trip_via=257)

    assert beats.reconciles()
    assert beats.n_beats == rec.n_beats - 2
    assert np.array_equal(beats.y, rec.beat_labels[1:-1])


def test_refined_variant_reconciles(incart_record, bundle):
    rec = io.load_labelled_record(incart_record["dir"], "I99")

    beats = ev.record_beats(rec, bundle, refine=True)

    assert beats.reconciles() and beats.n_beats == 18


def test_round_trip_on_a_257_hz_record_is_refused(incart_record, bundle):
    rec = io.load_labelled_record(incart_record["dir"], "I99")

    with pytest.raises(ValueError):
        ev.at_model_rate(rec, bundle, round_trip_via=257)
