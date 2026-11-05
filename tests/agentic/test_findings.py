"""The shared finding format and the case that carries it."""

from __future__ import annotations

import json

import pytest

from agentic.findings import Case, FindingList, make_banner


def test_ids_are_numbered_per_agent():
    heart, chart = FindingList("heart"), FindingList("chart")
    a = heart.add("class_share", "V", "10.0% V")
    b = heart.add("heart_rate", "", "80 beats/min")
    c = chart.add("demographics", "", "Age 70")
    assert (a.id, b.id, c.id) == ("H1", "H2", "C1")


def test_unknown_agent_is_refused():
    with pytest.raises(ValueError):
        FindingList("llm")


def test_banner_says_simulated_only_when_both_sources_are_present():
    both = make_banner("208", 200001)
    assert both == ("SIMULATED CASE: ECG from MIT-BIH record 208; chart from MIMIC-III Demo ICU "
                    "stay 200001. These are different people. Research demo, not for patient care.")
    assert "SIMULATED" not in make_banner("208", None)
    assert "SIMULATED" not in make_banner(None, 200001)


def test_banner_needs_something():
    with pytest.raises(ValueError):
        make_banner(None, None)


def test_facts_for_llm_never_include_display_only_or_evidence():
    fl = FindingList("heart")
    fl.add("class_share", "V", "20.0% V", value=20.0, unit="%", evidence={"secret": 1},
           caveat="model output")
    case = Case(banner="b", findings=fl.items, display_only={"truth": ["V", "N"], "agree": 1})
    facts = case.facts_for_llm()
    assert facts == [{"id": "H1", "statement": "20.0% V", "value": 20.0, "unit": "%",
                      "caveat": "model output"}]
    text = json.dumps(facts)
    assert "truth" not in text and "secret" not in text and "agree" not in text


def test_to_dict_round_trips_terms_as_a_list():
    f = FindingList("chart").add("lab_abnormal", "potassium:low", "low", terms=("potassium",))
    assert f.to_dict()["terms"] == ["potassium"]
