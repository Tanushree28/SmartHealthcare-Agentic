"""The fact-checker accepts faithful drafts and names every unsupported claim."""

from __future__ import annotations

import pytest

from agentic import verifier
from agentic.findings import Finding

from .fakes import good_report, make_case


def codes(problems):
    return [p.code for p in problems]


def test_faithful_draft_passes():
    assert verifier.check_report(good_report(), make_case()) == []


def edit(point_text, cites=("C1",)):
    d = good_report()
    d["points"][1] = {"text": point_text, "cites": list(cites)}
    return d


@pytest.mark.parametrize("text, cites, code", [
    ("Potassium was low at 2.8 mEq/L.", ["C1"], "unsupported_number"),
    ("Potassium was low at 3.1 mEq/L.", ["H1"], "unsupported_number"),
    ("Potassium was low at 3.1 mEq/L.", ["H9"], "unknown_id"),
    ("Potassium was low.", [], "no_citation"),
    ("Several V waves were seen.", ["H1"], "wrong_term"),
    ("The patient is diagnosed with VT.", ["H2"], "diagnosis_language"),
    ("Potassium should be increased.", ["C1"], "treatment_language"),
    ("Give 40 mEq of potassium.", ["C1"], "dose"),
    ("The patient died later.", ["C1"], "outcome_language"),
    ("The risk score was 41%.", ["C3"], "unsupported_number"),
    ("Lactate was high.", ["C1"], "unsupported_term"),
    ("Metoprolol was given.", ["C2"], "unsupported_term"),
])
def test_each_problem_is_caught(text, cites, code):
    assert code in codes(verifier.check_report(edit(text, cites), make_case()))


def test_rounding_to_the_shown_precision_is_allowed():
    assert verifier.check_report(edit("About 18% of beats were V beats.", ["H1"]),
                                 make_case()) == []


def test_numbers_from_the_matched_findings_of_a_cited_rule_are_allowed():
    d = good_report()
    d["for_clinician_review"] = {"text": "Potassium of 3.1 mEq/L with V beats is worth checking.",
                                 "cites": ["R3"]}
    assert verifier.check_report(d, make_case()) == []


def test_context_numbers_are_allowed():
    d = good_report()
    d["summary"] = "In the first 24 hours of ICU stay 200001, potassium was 3.1 mEq/L."
    assert verifier.check_report(d, make_case()) == []


def test_units_with_a_slash_are_not_doses():
    assert verifier.check_report(edit("Potassium was 3.1 mEq/L.", ["C1"]), make_case()) == []


@pytest.mark.parametrize("draft", [
    {"summary": 3, "points": [], "for_clinician_review": {"text": "", "cites": []}},
    {"summary": "s", "points": [{"text": "t", "cites": "H1"}],
     "for_clinician_review": {"text": "", "cites": []}},
    {"summary": "s", "points": [{"text": "t", "cites": ["H1"]}]},
    ["not", "a", "dict"],
])
def test_schema_problems_are_reported(draft):
    assert "schema" in codes(verifier.check_report(draft, make_case()))


def test_empty_review_is_allowed():
    d = good_report()
    d["for_clinician_review"] = {"text": "", "cites": []}
    assert verifier.check_report(d, make_case()) == []


GLOSSARY = [Finding(id="G3", agent="glossary", kind="glossary", key="V beat",
                    statement="V beat: an early beat that starts in the ventricles.",
                    terms=("beat:V",))]


def test_chat_answer_checks():
    case = make_case()
    ok = {"answerable": True, "answer": "A V beat starts in the ventricles.", "cites": ["G3"]}
    assert verifier.check_chat(ok, case, GLOSSARY) == []
    assert verifier.check_chat({"answerable": False, "answer": "", "cites": []}, case, GLOSSARY) == []
    bad = {"answerable": True, "answer": "Potassium was 2.0.", "cites": ["C1"]}
    assert "unsupported_number" in codes(verifier.check_chat(bad, case, GLOSSARY))
    assert "schema" in codes(verifier.check_chat({"answer": "x"}, case, GLOSSARY))
