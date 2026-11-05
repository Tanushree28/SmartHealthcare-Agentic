"""The heart agent turns beat calls into rhythm findings."""

from __future__ import annotations

import numpy as np

from agentic.heart_agent import BeatCalls, analyze, runs

CLASSES = ("N", "S", "V", "F", "Q")


def calls(seq, rr=0.8, top=0.9, n_dropped=2, source="annotation", fs=360):
    idx = np.array([CLASSES.index(c) for c in seq], dtype=np.int64)
    scores = np.full((len(seq), 5), (1 - top) / 4)
    scores[np.arange(len(seq)), idx] = top
    gaps = np.full(max(len(seq) - 1, 0), round(rr * fs))
    peaks = (100 + np.concatenate([[0], np.cumsum(gaps)])).astype(np.int64)[:len(seq)]
    return BeatCalls(CLASSES, idx, scores, peaks, fs, len(seq) + n_dropped, n_dropped, source)


def by_kind(findings):
    out = {}
    for f in findings:
        out.setdefault(f.kind, []).append(f)
    return out


def test_class_shares_skip_absent_classes():
    k = by_kind(analyze(calls("NNNNNNNNVV")))
    shares = {f.key: f.value for f in k["class_share"]}
    assert shares == {"N": 80.0, "V": 20.0}
    assert k["class_share"][1].statement.startswith("20.0% of beats (2 of 10)")
    assert k["class_share"][1].terms == ("beat:V",)


def test_heart_rate_from_rr():
    hr = by_kind(analyze(calls("NNNNN", rr=0.75)))["heart_rate"][0]
    assert hr.value == 80 and hr.evidence == {"min": 80, "max": 80}


def test_runs_of_three_counted_runs_of_two_ignored():
    assert runs(np.array([0, 0, 2, 2, 2, 0, 0, 2, 2, 0]), 2) == [(2, 4)]
    f = by_kind(analyze(calls("NNVVVNNVVN", rr=0.5)))["v_run"][0]
    assert f.value == 1 and f.evidence["lengths"] == [3] and f.evidence["max_rate_bpm"] == 120
    assert "v_run" not in by_kind(analyze(calls("NNVVNNVVNN")))


def test_low_confidence_and_q_share():
    k = by_kind(analyze(calls("Q" + "N" * 19, top=0.4)))
    assert k["low_confidence_beats"][0].value == 20
    assert k["q_share_high"][0].value == 5.0


def test_one_beat_gives_no_heart_rate():
    k = by_kind(analyze(calls("N")))
    assert "heart_rate" not in k and k["beats_classified"][0].value == 1


def test_no_beats_gives_one_finding():
    findings = analyze(calls(""))
    assert [f.kind for f in findings] == ["beats_classified"]
    assert findings[0].value == 0 and "no beats" in findings[0].statement


def test_ids_and_beat_source():
    findings = analyze(calls("NNNN", source="detected"))
    assert [f.id for f in findings] == [f"H{i}" for i in range(1, len(findings) + 1)]
    assert "detected automatically" in by_kind(findings)["beat_source"][0].statement
