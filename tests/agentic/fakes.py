"""Test doubles: a hand-made case, a draft that passes the fact-check, and a fake LLM."""

from __future__ import annotations

import copy
import itertools

from agentic.findings import Case, Finding, FindingList
from agentic.llm import LlmStatus


def make_case() -> Case:
    heart = FindingList("heart")
    heart.add("class_share", "V", "18.4% of beats (5 of 27) were classified V (ventricular ectopic)",
              value=18.4, unit="%", evidence={"count": 5, "total": 27}, terms=("beat:V",))
    heart.add("v_run", "V", "1 run(s) of 3 or more consecutive V beats (longest 4 beats, "
              "fastest 124 beats/min)", value=1, unit="runs",
              evidence={"lengths": [4], "rates_bpm": [124], "max_rate_bpm": 124},
              terms=("beat:V", "run"))
    chart = FindingList("chart")
    chart.add("lab_abnormal", "potassium:low", "Last potassium before the 24-hour mark: 3.1 mEq/L "
              "(low; outside the review range 3.5 to 5)", value=3.1, unit="mEq/L",
              terms=("potassium",))
    chart.add("medication", "antiarrhythmic", "Antiarrhythmic started before the 24-hour mark: "
              "Amiodarone", value=1, unit="drugs", evidence={"drugs": ["Amiodarone"]},
              terms=("medication", "antiarrhythmic", "drug:amiodarone"))
    chart.add("risk_score", "", "Demo risk model score 0.41 for in-hospital death (threshold "
              "0.70): below the threshold. Biggest pushes: +lactate_max, -age, +heart_rate_mean",
              value=0.41, evidence={"threshold": 0.7, "flagged": False},
              terms=("risk_score", "lactate", "heart_rate"), caveat="not validated")
    rule = Finding(id="R3", agent="rule", kind="rule", key="R3",
                   statement="Ventricular ectopy together with low potassium: a combination "
                             "worth checking.",
                   evidence={"matched": ["H1", "C1"], "title": "t", "source": "s", "status": "d"},
                   terms=("beat:V", "potassium"))
    return Case(banner="SIMULATED CASE: ECG from MIT-BIH record 208; chart from MIMIC-III Demo "
                       "ICU stay 200001. These are different people. Research demo, not for "
                       "patient care.",
                record="208", stay=200001, findings=heart.items + chart.items, rules=[rule],
                context_numbers=(24.0, 208.0, 0.0, 20.0, 200001.0))


GOOD_REPORT = {
    "summary": "This simulated case shows frequent V beats and a low potassium of 3.1 mEq/L.",
    "points": [
        {"text": "18.4% of beats were V beats, with a run of 4 at 124 beats/min.",
         "cites": ["H1", "H2"]},
        {"text": "Potassium was low at 3.1 mEq/L.", "cites": ["C1"]},
        {"text": "Amiodarone had been started before the 24-hour mark.", "cites": ["C2"]},
    ],
    "for_clinician_review": {"text": "Ventricular ectopy together with low potassium is worth "
                                     "checking.", "cites": ["R3"]},
}


def good_report() -> dict:
    return copy.deepcopy(GOOD_REPORT)


class FakeLLM:
    """Returns scripted replies (dicts, or exceptions to raise) and records every prompt."""

    def __init__(self, replies, model: str = "fake-llm", cycle: bool = False) -> None:
        self._replies = itertools.cycle(replies) if cycle else iter(replies)
        self.model = model
        self.calls: list[dict] = []

    def chat_json(self, messages, schema) -> dict:
        self.calls.append({"messages": [dict(m) for m in messages], "schema": schema})
        reply = next(self._replies)
        if isinstance(reply, Exception):
            raise reply
        return copy.deepcopy(reply)

    def status(self) -> LlmStatus:
        return LlmStatus(True, self.model, "ready")

    def prompt_text(self) -> str:
        return "\n".join(m["content"] for c in self.calls for m in c["messages"])
