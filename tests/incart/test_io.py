"""Loading one record's lead II and its beat labels."""

from __future__ import annotations

import numpy as np
import pytest

from incart import io
from mit_bih.errors import SamplingRateError, UnknownSymbolError

from .synthetic import write_incart_record


def test_lead_ii_is_picked_by_name_from_twelve_leads(incart_record):
    rec = io.load_labelled_record(incart_record["dir"], "I99")

    assert rec.lead == "II" and rec.fs == 257
    # Channel 2 is III; a prefix match would pick the wrong lead.
    assert np.allclose(rec.signal, incart_record["lead_ii"], atol=0.01)
    assert rec.n_beats == 20 and rec.n_excluded == 1
    assert np.array_equal(rec.beat_samples, incart_record["r_peaks"])
    assert np.array_equal(rec.beat_labels, incart_record["labels"])
    assert rec.symbol_counts == {"+": 1, "N": 14, "V": 6}
    assert not rec.has_paced


def test_record_at_another_rate_is_rejected(tmp_path):
    write_incart_record(tmp_path / "x", "I98", fs=360)

    with pytest.raises(SamplingRateError):
        io.load_labelled_record(tmp_path / "x", "I98")


def test_unknown_symbol_stops_the_run():
    with pytest.raises(UnknownSymbolError):
        io.label_beats(np.array([10, 20]), ("N", "Z"))


def test_non_beats_are_excluded_and_paced_beats_are_kept_as_beats():
    samples, labels, n_excluded = io.label_beats(np.array([10, 20, 30]), ("N", "/", "+"))

    assert samples.tolist() == [10, 20]
    assert labels.tolist() == [0, 4]
    assert n_excluded == 1


def test_paced_symbol_marks_the_record(tmp_path):
    write_incart_record(tmp_path / "p", "I97", cycle=("N", "/"))

    assert io.load_labelled_record(tmp_path / "p", "I97").has_paced
