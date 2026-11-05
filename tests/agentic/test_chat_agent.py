"""The chat answers from the case and the glossary, and refuses the rest."""

from __future__ import annotations

import pytest

from agentic import glossary
from agentic.chat_agent import ChatAgent, REFUSE_SCOPE
from agentic.llm import LlmUnavailable

from .fakes import FakeLLM, make_case


@pytest.fixture(scope="module")
def entries():
    return glossary.load_glossary()


def test_glossary_loads_with_sources(entries):
    assert [g.id for g in entries][:3] == ["G1", "G2", "G3"]
    assert all(g.agent == "glossary" and g.evidence["source"] for g in entries)


def test_glossary_refuses_an_entry_without_a_source(tmp_path):
    path = tmp_path / "g.toml"
    path.write_text('[[entry]]\nid = "G1"\nterm = "x"\ndefinition = "y"\nsource = ""\n')
    with pytest.raises(glossary.GlossaryError, match="source"):
        glossary.load_glossary(path)


@pytest.mark.parametrize("question, reason", [
    ("What dose of potassium should we give?", "treatment"),
    ("Does this patient have heart failure?", "diagnosis"),
    ("Did the patient survive?", "outcome"),
])
def test_blocked_questions_never_reach_the_llm(entries, question, reason):
    fake = FakeLLM([])
    answer = ChatAgent(make_case(), entries, fake).ask(question)
    assert answer.refused and answer.reason == reason and fake.calls == []


def test_glossary_question_is_answered(entries):
    fake = FakeLLM([{"answerable": True, "answer": "A V beat is an early beat that starts in "
                                                   "the ventricles.", "cites": ["G3"]}])
    answer = ChatAgent(make_case(), entries, fake).ask("What is a V beat?")
    assert answer.verified and not answer.refused and answer.cites == ("G3",)


def test_case_question_is_answered_from_findings(entries):
    fake = FakeLLM([{"answerable": True, "answer": "There was 1 run, the longest 4 beats.",
                     "cites": ["H2"]}])
    answer = ChatAgent(make_case(), entries, fake).ask("How many runs of V beats?")
    assert answer.verified and answer.text.startswith("There was 1 run")


def test_out_of_scope_is_refused(entries):
    fake = FakeLLM([{"answerable": False, "answer": "", "cites": []}])
    answer = ChatAgent(make_case(), entries, fake).ask("What is the capital of France?")
    assert answer.refused and answer.text == REFUSE_SCOPE


def test_two_failed_checks_are_refused(entries):
    bad = {"answerable": True, "answer": "Potassium was 2.0.", "cites": ["C1"]}
    answer = ChatAgent(make_case(), entries, FakeLLM([bad, bad])).ask("What was potassium?")
    assert answer.refused and "fact-check" in answer.reason


def test_llm_error_refuses(entries):
    answer = ChatAgent(make_case(), entries, FakeLLM([LlmUnavailable("down")])).ask("Why?")
    assert answer.refused and "down" in answer.reason


def test_no_llm_refuses(entries):
    answer = ChatAgent(make_case(), entries, None).ask("What is a V beat?")
    assert answer.refused and "ollama serve" in answer.text


def test_history_is_sent(entries):
    fake = FakeLLM([{"answerable": True, "answer": "Potassium was 3.1 mEq/L.", "cites": ["C1"]}])
    ChatAgent(make_case(), entries, fake).ask("And potassium?", [("How many runs?", "1 run.")])
    contents = [m["content"] for m in fake.calls[0]["messages"]]
    assert "How many runs?" in contents and "1 run." in contents


def test_empty_question_is_refused(entries):
    assert ChatAgent(make_case(), entries, FakeLLM([])).ask("   ").refused


def test_chat_prompt_says_a_simulated_case_is_two_people(entries):
    fake = FakeLLM([{"answerable": False, "answer": "", "cites": []}])
    ChatAgent(make_case(), entries, fake).ask("Tell me about the ECG.")
    assert "Never describe them as one patient" in fake.calls[0]["messages"][0]["content"]
