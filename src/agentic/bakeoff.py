"""LLM bake-off: the same cases for each candidate model, scored by the fact-checker.

The decision rule is fixed before the run (PlanDoc/agentic_with_fe.md §13):
highest pass rate after the retry; ties go to higher rule coverage (the share
of matched rules the report cites); then to the faster model.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass

from . import report_agent
from .findings import Case


@dataclass(frozen=True)
class Trial:
    model: str
    case_label: str
    kind: str  # "llm" | "template"
    attempts: int
    seconds: float
    problem_codes: tuple[str, ...]
    coverage: float | None  # None when no rule matched


def coverage(draft: dict | None, case: Case) -> float | None:
    rule_ids = {r.id for r in case.rules}
    if not rule_ids:
        return None
    if not draft:
        return 0.0
    items = [*draft.get("points", []), draft.get("for_clinician_review", {})]
    cited = {c for item in items if isinstance(item, dict) for c in item.get("cites", [])}
    return len(rule_ids & cited) / len(rule_ids)


def run_trials(cases: list[tuple[str, Case]], clients) -> list[Trial]:
    trials = []
    for client in clients:
        for label, case in cases:
            r = report_agent.write_report(case, client)
            codes = tuple(p.code for p in r.problems)
            if r.note.startswith("LLM unavailable"):
                codes = ("llm_error",)
            trials.append(Trial(client.model, label, r.kind, r.attempts, r.seconds, codes,
                                coverage(r.draft if r.kind == "llm" else None, case)))
    return trials


def summarize(trials: list[Trial]) -> list[dict]:
    rows = []
    for model in dict.fromkeys(t.model for t in trials):
        ts = [t for t in trials if t.model == model]
        n = len(ts)
        cov = [t.coverage for t in ts if t.coverage is not None]
        rows.append({
            "model": model, "cases": n,
            "pass_first_try": sum(t.kind == "llm" and t.attempts == 1 for t in ts) / n,
            "pass_after_retry": sum(t.kind == "llm" for t in ts) / n,
            "coverage": statistics.mean(cov) if cov else None,
            "median_seconds": statistics.median(t.seconds for t in ts),
            "max_seconds": max(t.seconds for t in ts),
            "problems": dict(Counter(c for t in ts for c in t.problem_codes)),
        })
    return rows


def pick_winner(rows: list[dict]) -> str:
    best = max(rows, key=lambda r: (r["pass_after_retry"], r["coverage"] or 0.0,
                                    -r["median_seconds"]))
    return best["model"]


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def render_markdown(rows: list[dict], winner: str, n_cases: int, seed: int,
                    run_date: str) -> str:
    lines = [
        "# LLM bake-off for the report agent", "",
        f"Run {run_date}. {n_cases} simulated cases (held-out MIT-BIH windows paired with "
        f"MIMIC-III Demo ICU stays; different people), seed {seed}. Aggregates only: no "
        f"patient values are in this file.", "",
        "Decision rule, fixed before the run: highest pass rate after the retry; ties to "
        "higher rule coverage; then the faster model.", "",
        "| model | cases | pass first try | pass after retry | rule coverage | median s | max s "
        "| problems in rejected drafts |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        problems = ", ".join(f"{k} {v}" for k, v in sorted(r["problems"].items())) or "none"
        lines.append(f"| {r['model']} | {r['cases']} | {_pct(r['pass_first_try'])} | "
                     f"{_pct(r['pass_after_retry'])} | {_pct(r['coverage'])} | "
                     f"{r['median_seconds']:.1f} | {r['max_seconds']:.1f} | {problems} |")
    lines += ["", f"Winner: **{winner}**", ""]
    return "\n".join(lines)
