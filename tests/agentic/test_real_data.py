"""On the real MIT-BIH records and the MIMIC-III Demo (skipped by default)."""

from __future__ import annotations

import pytest

from agentic import chart_agent, orchestrator, report_agent, verifier
from agentic.llm import OllamaClient

from .fakes import make_case


def _real_deps(need_ecg=True):
    from mimic_iii import config as mimic_config
    from mit_bih import config as ecg_config

    if not ecg_config.mitdb_dir().is_dir():
        pytest.skip("MIT-BIH not on disk")
    if not (mimic_config.output_dir("demo") / "model_meta.json").exists():
        pytest.skip("MIMIC-III Demo outputs not built")
    deps = orchestrator.default_deps(need_ecg=need_ecg, need_chart=True)
    if deps.unavailable:
        pytest.skip(str(deps.unavailable))
    return deps


@pytest.mark.requires_data
def test_real_case_runs_end_to_end_with_the_template():
    deps = _real_deps()
    stay = int(deps.chart.cohort["icustay_id"].sort_values().iloc[0])
    req = orchestrator.CaseRequest(record="208", seconds=20.0, stay=stay)
    events = list(orchestrator.run_case(req, deps))
    assert [e.stage for e in events] == list(range(1, 10))
    assert events[4].payload["display_only"]["of"] > 15
    case, report = events[8].payload["case"], events[8].payload["report"]
    assert report.kind == "template" and case.banner.startswith("SIMULATED CASE")
    assert any(f.kind == "class_share" for f in case.findings)


@pytest.mark.requires_data
def test_every_demo_stay_gives_chart_findings():
    deps = _real_deps(need_ecg=False)
    for stay in deps.chart.cohort["icustay_id"]:
        findings = chart_agent.analyze(int(stay), deps.chart, deps.knowledge)
        assert findings[0].kind == "demographics"
        assert any(f.kind == "risk_score" for f in findings)


@pytest.mark.requires_ollama
def test_real_llm_report_is_fact_checked_or_falls_back():
    client = OllamaClient()
    if not client.status().ok:
        pytest.skip(client.status().message)
    case = make_case()
    r = report_agent.write_report(case, client)
    assert r.kind in ("llm", "template")
    if r.kind == "llm":
        assert verifier.check_report(r.draft, case) == []
