"""The chart agent turns one ICU stay into findings, using only pre-index data."""

from __future__ import annotations

import pytest

from agentic import chart_agent


def run(stay, chart_sources, knowledge):
    findings = chart_agent.analyze(stay, chart_sources, knowledge)
    return findings, {f.kind: [g for g in findings if g.kind == f.kind] for f in findings}


def test_reference_stay(chart_sources, knowledge):
    findings, k = run(101, chart_sources, knowledge)
    assert [f.id for f in findings] == [f"C{i}" for i in range(1, len(findings) + 1)]
    assert k["admission_reason"][0].value == "CONGESTIVE HEART FAILURE"
    labs = {f.key: f for f in k["lab_abnormal"]}
    assert set(labs) == {"creatinine:high", "troponin_t:high"}
    assert labs["creatinine:high"].evidence["hospital_flag"] is True
    assert labs["creatinine:high"].value == 2.0
    assert "vital_out_of_range" not in k
    meds = {f.key: f for f in k["medication"]}
    assert set(meds) == {"antiarrhythmic"} and meds["antiarrhythmic"].evidence["drugs"] == ["Amiodarone"]
    assert "drug:amiodarone" in meds["antiarrhythmic"].terms
    risk = k["risk_score"][0]
    assert 0.0 <= risk.value <= 1.0 and "not validated" in risk.caveat
    assert set(risk.evidence) == {"threshold", "flagged", "reasons", "split"}


def test_stay_with_abnormal_vitals_and_low_potassium(chart_sources, knowledge):
    _, k = run(201, chart_sources, knowledge)
    labs = {f.key for f in k["lab_abnormal"]}
    assert {"lactate:high", "potassium:low"} <= labs
    vitals = {f.key: f.value for f in k["vital_out_of_range"]}
    assert vitals == {"heart_rate:high": 110.0, "resp_rate:high": 28.0}
    assert {f.key for f in k["medication"]} == {"potassium_supplement"}
    assert k["medication_count"][0].value == 2


def test_stay_with_no_measurements(chart_sources, knowledge):
    _, k = run(401, chart_sources, knowledge)
    assert k["demographics"][0].statement.startswith("Age >89")
    assert k["not_measured"][0].value == 18
    assert "lab_abnormal" not in k and "vital_out_of_range" not in k
    assert k["medication_count"][0].value == 0
    assert "risk_score" in k


def test_unknown_stay_raises(chart_sources, knowledge):
    with pytest.raises(KeyError):
        chart_agent.analyze(999999, chart_sources, knowledge)


def test_no_statement_mentions_the_outcome(chart_sources, knowledge):
    for stay in (101, 201, 401):
        for f in chart_agent.analyze(stay, chart_sources, knowledge):
            text = f"{f.statement} {f.caveat} {f.evidence}".lower()
            assert not any(w in text for w in ("label", "expire", "died", "survived",
                                               "discharge"))
