"""The page's logic, without HTTP: outcome lookup, options, runs, chat."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontend.outcome import OutcomeLookup

SRC_AGENTIC = Path(__file__).resolve().parents[2] / "src" / "agentic"


def test_outcome_lookup_reads_the_label(chart_env):
    lookup = OutcomeLookup(chart_env["out"])
    assert lookup.get(201)["outcome"] == "died in hospital"
    assert lookup.get(101)["outcome"] == "survived to hospital discharge"
    assert "never" in lookup.get(101)["note"]


def test_unknown_stay_outcome_raises(chart_env):
    with pytest.raises(KeyError):
        OutcomeLookup(chart_env["out"]).get(999999)


def test_agentic_never_imports_the_outcome_module():
    for path in SRC_AGENTIC.glob("*.py"):
        text = path.read_text()
        assert "frontend" not in text and "OutcomeLookup" not in text, path.name


# --- App logic ---------------------------------------------------------------

from agentic import orchestrator  # noqa: E402
from frontend.app import App, parse_run  # noqa: E402

from ..agentic.fakes import FakeLLM  # noqa: E402


def q(**kw):
    return {k: [str(v)] for k, v in kw.items()}


def test_options_lists_records_stays_and_llm(app):
    o = app.options()
    recs = {r["name"]: r for r in o["records"]}
    assert len(recs) == 44 and recs["100"]["tag"] == "held-out"
    assert recs["100"]["seconds"] == 34.4 and recs["208"]["seconds"] is None  # not on disk here
    stays = {s["id"]: s for s in o["stays"]}
    assert stays[101]["reason"] == "CONGESTIVE HEART FAILURE"
    assert stays[101]["split"] in {"train", "val", "test"} and stays[401]["age"] == ">89"
    assert o["llm"]["ok"] is False and "off" in o["llm"]["message"]
    assert o["limits"] == {"min_seconds": 2.0, "max_seconds": 120.0}
    assert o["outcome_available"] is True


@pytest.mark.parametrize("query, message", [
    (q(record="100", seconds="abc"), "numbers"),
    (q(record="100", start="nan"), "numbers"),
    (q(record="100", seconds="500"), "at most 120"),
    (q(record="100", peaks="x"), "peaks"),
    (q(stay="abc"), "stay must be a number"),
    ({}, "choose"),
])
def test_parse_run_rejects_bad_values(query, message):
    with pytest.raises(ValueError, match=message):
        parse_run(query)


def test_parse_run_reads_a_full_query():
    req = parse_run(q(record="100", start="2", seconds="20", stay="101", peaks="detected"))
    assert req == orchestrator.CaseRequest(record="100", start_s=2.0, seconds=20.0, stay=101,
                                           detect_peaks=True)


def test_run_streams_nine_stages_then_done(app):
    events = list(app.run(q(record="100", start="2", seconds="20", stay="101")))
    assert [k for k, _ in events] == ["stage"] * 9 + ["done"]
    case_id = events[-1][1]["case_id"]
    assert events[8][1]["case_id"] == case_id and case_id in app.cases


def test_run_reports_errors_as_events(app):
    past_end = list(app.run(q(record="100", start="30", seconds="20")))
    assert past_end == [("error", {"message": "the window ends at 50.0 s but record 100 is "
                                              "34.4 s long"})]
    unknown = list(app.run(q(stay="999999")))
    assert unknown[-1][0] == "error" and "not in the cohort" in unknown[-1][1]["message"]
    assert list(app.run(q(record="100", seconds="abc")))[0][0] == "error"


def _finding_id(app, case_id, kind, key):
    return next(f.id for f in app.cases[case_id].case.findings if f.kind == kind and f.key == key)


def test_chat_uses_the_stored_case_and_keeps_history(deps, chart_env):
    from frontend.outcome import OutcomeLookup

    fake = FakeLLM([{}, {}], model="fake")  # the report: two rejected drafts -> template
    app = App(deps, llm=fake, outcome=OutcomeLookup(chart_env["out"]))
    case_id = list(app.run(q(stay="101")))[-1][1]["case_id"]
    creat = _finding_id(app, case_id, "lab_abnormal", "creatinine:high")
    trop = _finding_id(app, case_id, "lab_abnormal", "troponin_t:high")
    fake._replies = iter([
        {"answerable": True, "answer": "Creatinine was 2 mg/dL.", "cites": [creat]},
        {"answerable": True, "answer": "Troponin T was 0.5 ng/mL.", "cites": [trop]}])
    first = app.chat(case_id, "What was creatinine?")
    second = app.chat(case_id, "And troponin?")
    assert first["verified"] and not first["refused"] and first["cites"] == [creat]
    assert second["answer"] == "Troponin T was 0.5 ng/mL."
    last_prompt = [m["content"] for m in fake.calls[-1]["messages"]]
    assert "What was creatinine?" in last_prompt


def test_chat_unknown_case_raises(app):
    with pytest.raises(KeyError):
        app.chat("no-such-case", "hello")


def test_old_cases_are_forgotten(deps, chart_env):
    small = App(deps, max_cases=2)
    ids = [list(small.run(q(stay="101")))[-1][1]["case_id"] for _ in range(3)]
    assert ids[0] not in small.cases and ids[1] in small.cases and ids[2] in small.cases


def test_outcome_never_reaches_the_chat(deps, chart_env):
    from frontend.outcome import OutcomeLookup

    fake = FakeLLM([{}, {}, {"answerable": False, "answer": "", "cites": []}], model="fake")
    app = App(deps, llm=fake, outcome=OutcomeLookup(chart_env["out"]))
    case_id = list(app.run(q(stay="201")))[-1][1]["case_id"]
    assert app.outcome(201)["outcome"] == "died in hospital"
    app.chat(case_id, "What was lactate?")
    text = fake.prompt_text().lower()
    assert "died" not in text and "survived" not in text and "hospital discharge" not in text
