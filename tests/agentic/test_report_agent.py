"""The LLM writes; the fact-checker decides; the template is the fallback."""

from __future__ import annotations

from agentic import orchestrator, report_agent
from agentic.llm import LlmError, LlmUnavailable

from .fakes import FakeLLM, good_report, make_case

BAD = {"summary": "The patient was diagnosed with VT.", "points": [{"text": "x", "cites": []}],
       "for_clinician_review": {"text": "", "cites": []}}


def test_good_draft_is_shown():
    fake = FakeLLM([good_report()])
    r = report_agent.write_report(make_case(), fake)
    assert r.kind == "llm" and r.attempts == 1 and r.model == "fake-llm"
    assert r.text.startswith("SIMULATED CASE") and "fact-checked" in r.text
    assert "[H1, H2]" in r.text and "Limitations" in r.text


def test_one_retry_with_the_problems_listed():
    fake = FakeLLM([BAD, good_report()])
    r = report_agent.write_report(make_case(), fake)
    assert r.kind == "llm" and r.attempts == 2
    retry_prompt = fake.calls[1]["messages"][-1]["content"]
    assert "failed the fact-check" in retry_prompt and "diagnos" in retry_prompt


def test_two_failures_fall_back_to_the_template():
    r = report_agent.write_report(make_case(), FakeLLM([BAD, BAD]))
    assert r.kind == "template" and r.attempts == 2 and r.problems
    assert "failed the fact-check 2 times" in r.note and "[R3]" in r.text


def test_wrong_shape_twice_falls_back():
    shape = {"summary": 3, "points": "H1", "for_clinician_review": None}
    r = report_agent.write_report(make_case(), FakeLLM([shape, shape]))
    assert r.kind == "template" and {p.code for p in r.problems} == {"schema"}


def test_llm_unavailable_falls_back():
    r = report_agent.write_report(make_case(), FakeLLM([LlmUnavailable("start it")]))
    assert r.kind == "template" and r.note.startswith("LLM unavailable")


def test_invalid_json_falls_back():
    r = report_agent.write_report(make_case(), FakeLLM([LlmError("not valid JSON")]))
    assert r.kind == "template" and "not valid JSON" in r.note


def test_no_llm_means_template():
    r = report_agent.write_report(make_case(), None)
    assert r.kind == "template" and r.note == "LLM off"


def test_prompts_never_carry_truth_or_outcome(knowledge, tiny_classifier, chart_sources, mitdb_dir):
    fake = FakeLLM([BAD], cycle=True)
    deps = orchestrator.Deps(knowledge=knowledge, classifier=tiny_classifier, chart=chart_sources,
                             record_dir=mitdb_dir, write_report=report_agent.make_writer(fake))
    req = orchestrator.CaseRequest(record="100", start_s=2.0, seconds=20.0, stay=201)
    list(orchestrator.run_case(req, deps))
    text = fake.prompt_text().lower()
    assert fake.calls
    for word in ("truth", "agree", "display", "label", "hospital_expire_flag", "deathtime",
                 "expired", "died", "survived", "discharge"):
        assert word not in text, word


def test_simulated_case_tells_the_llm_they_are_two_people():
    import json

    from agentic.findings import Case

    facts = json.loads(report_agent.facts_message(make_case()))
    assert "different people" in facts["note"] and "one patient" in facts["note"]
    ecg_only = Case(banner="b", record="208", findings=make_case().findings[:2])
    assert "note" not in json.loads(report_agent.facts_message(ecg_only))
