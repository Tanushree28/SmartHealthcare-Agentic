"""The bake-off scores models on the fact-check, rule coverage and time."""

from __future__ import annotations

from agentic import bakeoff
from agentic.llm import LlmUnavailable

from .fakes import FakeLLM, good_report, make_case

BAD = {"summary": "x", "points": [{"text": "V waves", "cites": ["H1"]}],
       "for_clinician_review": {"text": "", "cites": []}}


def test_scores_and_winner():
    cases = [("case 1", make_case()), ("case 2", make_case())]
    good = FakeLLM([good_report()], model="good", cycle=True)
    flaky = FakeLLM([BAD, good_report()], model="flaky", cycle=True)
    bad = FakeLLM([BAD], model="bad", cycle=True)
    rows = bakeoff.summarize(bakeoff.run_trials(cases, [good, flaky, bad]))
    by = {r["model"]: r for r in rows}
    assert by["good"]["pass_first_try"] == 1.0 and by["good"]["coverage"] == 1.0
    assert by["flaky"]["pass_first_try"] == 0.0 and by["flaky"]["pass_after_retry"] == 1.0
    assert by["bad"]["pass_after_retry"] == 0.0 and by["bad"]["problems"]["wrong_term"] == 2
    assert by["bad"]["coverage"] == 0.0
    assert bakeoff.pick_winner(rows) in {"good", "flaky"}


def test_ties_break_on_coverage_then_speed():
    rows = [{"model": "a", "pass_after_retry": 1.0, "coverage": 0.5, "median_seconds": 1.0},
            {"model": "b", "pass_after_retry": 1.0, "coverage": 1.0, "median_seconds": 9.0},
            {"model": "c", "pass_after_retry": 1.0, "coverage": 1.0, "median_seconds": 2.0}]
    assert bakeoff.pick_winner(rows) == "c"


def test_unreachable_model_is_scored_as_an_llm_error():
    rows = bakeoff.summarize(bakeoff.run_trials(
        [("case 1", make_case())], [FakeLLM([LlmUnavailable("down")], model="off")]))
    assert rows[0]["pass_after_retry"] == 0.0 and rows[0]["problems"] == {"llm_error": 1}


def test_markdown_holds_aggregates_only():
    case = make_case()
    rows = bakeoff.summarize(bakeoff.run_trials(
        [("case 1", case)], [FakeLLM([good_report()], model="good", cycle=True)]))
    text = bakeoff.render_markdown(rows, "good", n_cases=1, seed=7, run_date="2026-10-15")
    assert "| good |" in text and "Winner: **good**" in text
    assert str(case.stay) not in text and "3.1" not in text and "SIMULATED" not in text
