"""Stage events become strict JSON the page can read."""

from __future__ import annotations

import json

from agentic import orchestrator
from agentic.findings import ReportResult
from agentic.verifier import Problem
from frontend import serialize


def test_full_case_serializes_to_strict_json(deps):
    req = orchestrator.CaseRequest(record="100", start_s=2.0, seconds=20.0, stay=101)
    events = [serialize.event(e) for e in orchestrator.run_case(req, deps)]
    for e in events:
        json.dumps(e, allow_nan=False)
    assert [e["stage"] for e in events] == list(range(1, 10))
    assert events[0]["record"] == {"name": "100", "tag": "held-out", "note": ""}
    assert len(events[0]["signal"]) == 7200 and isinstance(events[0]["signal"][0], float)
    assert set(events[4]["display_only"]) == {"truth", "agree", "of"}
    assert events[8]["simulated"] is True and events[8]["report"]["kind"] == "template"
    assert events[5]["findings"][0]["id"] == "H1"


def test_skipped_stage_keeps_only_the_reason():
    ev = orchestrator.StageEvent(7, "chart_agent", {"skipped": "no ICU stay chosen"})
    assert serialize.event(ev) == {"stage": 7, "name": "chart_agent",
                                   "skipped": "no ICU stay chosen"}


def test_report_problems_become_text(deps):
    req = orchestrator.CaseRequest(stay=101)
    case = list(orchestrator.run_case(req, deps))[-1].payload["case"]
    report = ReportResult("template", "text", model="m", attempts=2,
                          problems=(Problem("dose", "point 1: '40 mg' is a dose"),), note="n")
    out = serialize.event(orchestrator.StageEvent(9, "report", {"case": case, "report": report}))
    assert out["report"]["problems"] == ["point 1: '40 mg' is a dose"]
    assert out["simulated"] is False
