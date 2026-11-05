"""Report agent: the local LLM words the report; the fact-checker decides.

The LLM gets the banner and the facts (ID, statement, value, unit, caveat) and
nothing else. Its draft is shown only if verifier.check_report finds no
problem. One retry is allowed, with the problems listed; after that the
template report is shown, with the reason. The banner and the limitations are
always added by code, never left to the LLM.
"""

from __future__ import annotations

import json
import time

from . import template_report, verifier
from .findings import SIMULATED_NOTE, Case, ReportResult
from .llm import LlmError

_POINT = {"type": "object",
          "properties": {"text": {"type": "string"},
                         "cites": {"type": "array", "items": {"type": "string"}}},
          "required": ["text", "cites"]}
REPORT_SCHEMA = {"type": "object",
                 "properties": {"summary": {"type": "string"},
                                "points": {"type": "array", "items": _POINT},
                                "for_clinician_review": _POINT},
                 "required": ["summary", "points", "for_clinician_review"]}

SYSTEM_PROMPT = """You turn a list of facts about one case into a short plain-language summary for a research demo.

Rules:
- Use ONLY the facts given. Add nothing from your own knowledge.
- Every point must cite the IDs of the facts it uses, for example ["H2", "R1"].
- Copy numbers exactly as they appear in the facts you cite. Do not convert them (0.41 stays 0.41, not 41%).
- Do not diagnose. Say "consistent with" or "worth checking", never "has", "diagnosed" or "confirms".
- Do not suggest treatments, drugs to give, or doses.
- Do not say anything about the patient's outcome.
- Beat types are "N beats", "S beats", "V beats", "F beats" and "Q beats". Never write "V waves" or "S waves".
- summary: 2-3 sentences. points: 3-6 items. for_clinician_review: one sentence citing the matched rules (R...), or empty text and no cites if no rule matched.
Reply with JSON only."""


def facts_message(case: Case) -> str:
    payload = {"case": case.banner}
    if case.simulated:
        payload["note"] = SIMULATED_NOTE
    payload["facts"] = case.facts_for_llm()
    return json.dumps(payload, indent=1, ensure_ascii=False)


def render(draft: dict, case: Case, model: str) -> str:
    lines = [case.banner, "",
             f"AI-written ({model}), fact-checked by the app: every point traces to the cited "
             f"findings.", "", draft["summary"].strip(), ""]
    lines += [f"  - {p['text'].strip()} [{', '.join(p['cites'])}]" for p in draft["points"]]
    review = draft["for_clinician_review"]
    if review["text"].strip():
        lines += ["", f"For clinician review: {review['text'].strip()} "
                      f"[{', '.join(review['cites'])}]"]
    lines += ["", "Limitations", *(f"  - {x}" for x in template_report.limitations(case))]
    return "\n".join(lines)


def write_report(case: Case, llm, attempts: int = 2) -> ReportResult:
    if llm is None:
        return template_report.template_report(case, "LLM off")
    start = time.monotonic()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": facts_message(case)}]
    draft, problems = None, []
    for attempt in range(1, attempts + 1):
        try:
            draft = llm.chat_json(messages, REPORT_SCHEMA)
        except LlmError as exc:
            return ReportResult("template", template_report.render(case), model=llm.model,
                                seconds=time.monotonic() - start, attempts=attempt,
                                note=f"LLM unavailable: {exc}")
        problems = verifier.check_report(draft, case)
        if not problems:
            return ReportResult("llm", render(draft, case, llm.model), model=llm.model,
                                seconds=time.monotonic() - start, attempts=attempt, draft=draft)
        messages += [
            {"role": "assistant", "content": json.dumps(draft, ensure_ascii=False)},
            {"role": "user", "content": "Your draft failed the fact-check:\n"
             + "\n".join(f"- {p.detail}" for p in problems)
             + "\nWrite it again, fixing every problem. Same JSON format."}]
    return ReportResult("template", template_report.render(case), model=llm.model,
                        seconds=time.monotonic() - start, attempts=attempts,
                        problems=tuple(problems), draft=draft if isinstance(draft, dict) else None,
                        note=f"the AI draft failed the fact-check {attempts} times")


def make_writer(llm):
    return lambda case: write_report(case, llm)
