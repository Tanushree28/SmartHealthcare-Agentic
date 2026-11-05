"""Which records are offered, and how a window is cut from one."""

from __future__ import annotations

import pytest

from agentic import ecg_window


def test_offered_records_are_44_tagged_and_paced_ones_left_out():
    recs = {r.name: r for r in ecg_window.offered_records()}
    assert len(recs) == 44
    assert sum(r.tag == "held-out" for r in recs.values()) == 22
    assert sum(r.tag == "trained-on" for r in recs.values()) == 22
    assert not {"102", "104", "107", "217"} & set(recs)
    assert recs["208"].tag == "trained-on" and recs["100"].tag == "held-out"
    assert "201" in recs["202"].note and "202" in recs["201"].note


def test_window_peaks_are_relative_and_truth_holds_beats_only(mitdb_dir):
    win = ecg_window.load_window("100", start_s=2.0, seconds=20.0, record_dir=mitdb_dir)
    assert win.signal.size == 7200 and win.fs == 360 and win.lead == "MLII"
    assert win.r_peaks[0] == 80 and len(win.r_peaks) == 24  # beats 2..25 of the record
    assert len(win.truth) == 24 and win.truth.count("V") == 3 and "+" not in win.truth
    assert win.record.tag == "held-out"


def test_window_past_the_end_is_refused(mitdb_dir):
    with pytest.raises(ecg_window.WindowError, match="34.4 s long"):
        ecg_window.load_window("100", start_s=30.0, seconds=20.0, record_dir=mitdb_dir)


@pytest.mark.parametrize("record, start, seconds, message", [
    ("102", 0, 20, "paced"),
    ("999", 0, 20, "unknown record"),
    ("100", -1, 20, "start"),
    ("100", 0, 0, "seconds"),
])
def test_bad_requests_are_refused(mitdb_dir, record, start, seconds, message):
    with pytest.raises(ecg_window.WindowError, match=message):
        ecg_window.load_window(record, start, seconds, record_dir=mitdb_dir)


def test_too_short_window_is_refused(mitdb_dir):
    with pytest.raises(ecg_window.WindowError, match="at least 2 s"):
        ecg_window.load_window("100", 0, 0.5, record_dir=mitdb_dir)
