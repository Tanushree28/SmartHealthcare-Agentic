"""Chat agent: answers questions about the case on screen, and nothing else.

Treatment, diagnosis and outcome questions are refused by fixed patterns
before any LLM call. Everything else goes to the local LLM with the case's
facts and the glossary only; the answer is shown only if
verifier.check_chat passes (one retry), and otherwise a fixed refusal is shown.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import verifier
from .findings import SIMULATED_NOTE, Case, Finding
from .llm import LlmError

CHAT_SCHEMA = {"type": "object",
               "properties": {"answerable": {"type": "boolean"}, "answer": {"type": "string"},
                              "cites": {"type": "array", "items": {"type": "string"}}},
               "required": ["answerable", "answer", "cites"]}

CHAT_PROMPT = """You answer questions about one case for a research demo, using ONLY the facts and glossary entries below.
- If the question cannot be answered from them, reply {"answerable": false, "answer": "", "cites": []}.
- Otherwise answer in 1-3 plain sentences and list every fact or glossary ID you used in "cites".
- Copy numbers exactly as they appear. Do not diagnose, suggest treatments or doses, or talk about the patient's outcome.
- Beat types are "N beats", "S beats", "V beats", "F beats" and "Q beats", never "waves".
Reply with JSON only.

Facts and glossary:
"""

REFUSE_SCOPE = "I can only answer from this case's findings and the glossary."
REFUSE_TREATMENT = "I can't suggest treatments or doses. This is a research demo, not medical advice."
REFUSE_DIAGNOSIS = ("I can't diagnose. I can say what the findings show and which rules "
                    "matched.")
REFUSE_OUTCOME = "I don't have the patient's outcome; the AI is never given it."
NEEDS_LLM = "The chat needs the local LLM. Start Ollama with `ollama serve`."
MAX_QUESTION = 500
MAX_TURNS = 6

BLOCKED = (
    (re.compile(r"\b(?:dose|dosage|dosing|prescrib\w*|treat\w*|therapy|what (?:drug|medicine) "
                r"should|should (?:we|i|they|you) (?:give|start|stop|increase|decrease))\b", re.I),
     REFUSE_TREATMENT, "treatment"),
    (re.compile(r"\b(?:diagnos\w*|does (?:the|this) patient have|is (?:it|this) "
                r"(?:vt|an? mi|a heart attack|heart failure))\b", re.I),
     REFUSE_DIAGNOSIS, "diagnosis"),
    (re.compile(r"\b(?:did|will|does|would) (?:the |this )?(?:patient|person|he|she|they) "
                r"(?:die|survive|live|make it)\b|\boutcome\b|\bdischarged?\b|\bsurviv\w*|\bdied\b",
                re.I),
     REFUSE_OUTCOME, "outcome"),
)


@dataclass(frozen=True)
class ChatAnswer:
    text: str
    cites: tuple[str, ...] = ()
    verified: bool = False
    refused: bool = False
    reason: str = ""


class ChatAgent:
    def __init__(self, case: Case, glossary: list[Finding], llm) -> None:
        self.case = case
        self.glossary = glossary
        self.llm = llm
        context = {"case": case.banner,
                   **({"note": SIMULATED_NOTE} if case.simulated else {}),
                   "facts": case.facts_for_llm(),
                   "glossary": [{"id": g.id, "statement": g.statement} for g in glossary]}
        self.system = CHAT_PROMPT + json.dumps(context, indent=1, ensure_ascii=False)

    def ask(self, question: str, history=()) -> ChatAnswer:
        q = question.strip()
        if not q:
            return ChatAnswer(REFUSE_SCOPE, refused=True, reason="empty question")
        if len(q) > MAX_QUESTION:
            return ChatAnswer(f"Please ask in under {MAX_QUESTION} characters.", refused=True,
                              reason="too long")
        for pattern, reply, reason in BLOCKED:
            if pattern.search(q):
                return ChatAnswer(reply, verified=True, refused=True, reason=reason)
        if self.llm is None:
            return ChatAnswer(NEEDS_LLM, refused=True, reason="LLM off")

        messages = [{"role": "system", "content": self.system}]
        for past_q, past_a in list(history)[-MAX_TURNS:]:
            messages += [{"role": "user", "content": past_q},
                         {"role": "assistant", "content": past_a}]
        messages.append({"role": "user", "content": q})

        problems = []
        for _ in range(2):
            try:
                draft = self.llm.chat_json(messages, CHAT_SCHEMA)
            except LlmError as exc:
                return ChatAnswer(NEEDS_LLM, refused=True, reason=str(exc))
            problems = verifier.check_chat(draft, self.case, self.glossary)
            if not problems:
                if not draft["answerable"]:
                    return ChatAnswer(REFUSE_SCOPE, verified=True, refused=True,
                                      reason="outside the case and the glossary")
                return ChatAnswer(draft["answer"].strip(), tuple(draft["cites"]), verified=True)
            messages += [
                {"role": "assistant", "content": json.dumps(draft, ensure_ascii=False)},
                {"role": "user", "content": "That answer failed the fact-check:\n"
                 + "\n".join(f"- {p.detail}" for p in problems)
                 + "\nAnswer again, fixing every problem. Same JSON format."}]
        return ChatAnswer(REFUSE_SCOPE, refused=True,
                          reason="failed the fact-check twice: "
                                 + "; ".join(p.detail for p in problems))
