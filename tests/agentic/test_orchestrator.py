"""One case through every stage, with a tiny ECG ensemble and synthetic data."""

from __future__ import annotations

import json

from agentic import orchestrator, template_report
from agentic.findings import Case, FindingList


def deps(knowledge, tiny_classifier=None, chart_sources=None, mitdb_dir=None):
    return orchestrator.Deps(knowledge=knowledge, classifier=tiny_classifier, chart=chart_sources,
                             record_dir=mitdb_dir)


def test_full_case_emits_nine_stages_in_order(knowledge, tiny_classifier, chart_sources, mitdb_dir):
    req = orchestrator.CaseRequest(record="100", start_s=2.0, seconds=20.0, stay=101)
    events = list(orchestrator.run_case(req, deps(knowledge, tiny_classifier, chart_sources,
                                                  mitdb_dir)))
    assert [e.stage for e in events] == list(range(1, 10))
    assert not any(e.skipped for e in events)
    beats = events[3].payload
    assert beats["n_classified"] + beats["n_dropped"] == 24
    shown = events[4].payload["display_only"]
    assert shown["of"] == beats["n_classified"] and len(shown["truth"]) == beats["n_classified"]
    case, report = events[8].payload["case"], events[8].payload["report"]
    assert case.banner.startswith("SIMULATED CASE: ECG from MIT-BIH record 100")
    assert report.kind == "template" and report.text.startswith(case.banner)
    assert {f.agent for f in case.findings} == {"heart", "chart"}
    assert 24.0 in case.context_numbers and 101.0 in case.context_numbers


def test_truth_never_reaches_the_llm_facts(knowledge, tiny_classifier, chart_sources, mitdb_dir):
    req = orchestrator.CaseRequest(record="100", start_s=2.0, seconds=20.0, stay=101)
    case = list(orchestrator.run_case(req, deps(knowledge, tiny_classifier, chart_sources,
                                                mitdb_dir)))[-1].payload["case"]
    facts = json.dumps(case.facts_for_llm())
    assert "truth" not in facts and "agree" not in facts and "display" not in facts


def test_ecg_only_skips_the_chart_stage(knowledge, tiny_classifier, mitdb_dir):
    events = list(orchestrator.run_case(orchestrator.CaseRequest(record="100", start_s=2.0),
                                        deps(knowledge, tiny_classifier, None, mitdb_dir)))
    assert events[6].skipped and events[6].payload["skipped"] == "no ICU stay chosen"
    assert "SIMULATED" not in events[8].payload["case"].banner


def test_chart_only_skips_the_ecg_stages(knowledge, chart_sources):
    events = list(orchestrator.run_case(orchestrator.CaseRequest(stay=201),
                                        deps(knowledge, None, chart_sources)))
    assert [e.skipped for e in events[:6]] == [True] * 6
    assert events[8].payload["case"].stay == 201


def test_missing_chart_sources_skip_with_the_reason(knowledge):
    d = deps(knowledge)
    d.unavailable["chart"] = "MIMIC outputs not available"
    events = list(orchestrator.run_case(orchestrator.CaseRequest(stay=201), d))
    assert events[6].payload["skipped"] == "MIMIC outputs not available"


def test_template_lists_findings_rules_and_limitations():
    fl = FindingList("heart")
    fl.add("class_share", "V", "20.0% V", caveat="model output")
    case = Case(banner="BANNER", record="208", stay=1, findings=fl.items)
    text = template_report.render(case)
    assert text.startswith("BANNER") and "[H1] 20.0% V  (model output)" in text
    assert "Rules matched\n  none" in text
    assert template_report.SIMULATED_LIMITATION in text
    assert template_report.SIMULATED_LIMITATION not in template_report.render(
        Case(banner="B", record="208", findings=fl.items))
