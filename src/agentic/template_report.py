"""The plain report: every finding and rule match in a fixed layout.

Always correct and always available. It is what the reader gets when the LLM
is off, missing, slow, or its draft fails the fact-check.
"""

from __future__ import annotations

from .findings import Case, ReportResult

LIMITATIONS = (
    "Research demo. Not a diagnosis and not for patient care.",
    "Beat classes come from the ECG model (scores uncalibrated), not from a cardiologist.",
    "Rules and review ranges are drafts awaiting professor review.",
    "Chart findings use only data from before the 24-hour mark of the ICU stay.",
)
SIMULATED_LIMITATION = ("The ECG and the chart come from different people (simulated case); "
                        "nothing here describes one real patient.")


def limitations(case: Case) -> list[str]:
    out = list(LIMITATIONS)
    if case.record is not None and case.stay is not None:
        out.append(SIMULATED_LIMITATION)
    return out


def render(case: Case) -> str:
    lines = [case.banner, ""]
    for title, agent in (("Heartbeat (ECG) findings", "heart"),
                         ("Chart findings (before the 24-hour mark)", "chart")):
        items = [f for f in case.findings if f.agent == agent]
        if not items:
            continue
        lines.append(title)
        lines += [f"  [{f.id}] {f.statement}" + (f"  ({f.caveat})" if f.caveat else "")
                  for f in items]
        lines.append("")
    lines.append("Rules matched")
    if case.rules:
        for r in case.rules:
            ev = r.evidence
            lines.append(f"  [{r.id}] {r.statement} Based on {', '.join(ev['matched'])}. "
                         f"Source: {ev['source']}. Status: {ev['status']}.")
    else:
        lines.append("  none")
    lines += ["", "Limitations", *(f"  - {x}" for x in limitations(case))]
    return "\n".join(lines)


def template_report(case: Case, note: str = "LLM off") -> ReportResult:
    return ReportResult(kind="template", text=render(case), note=note)
