"""Evaluate the knowledge-file rules against a case's findings.

A rule matches when every ``all`` condition and at least one ``any`` condition
(if there are any) is met by some finding. The match is itself a finding with
the rule's ID, so the report and the chat can cite it like any other fact.
"""

from __future__ import annotations

import operator

from .findings import Finding
from .knowledge import Condition, Rule

_OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt,
        "==": operator.eq}


def _key_matches(cond_key: str | None, key: str) -> bool:
    if cond_key is None:
        return True
    if cond_key.endswith(":*"):
        return key.startswith(cond_key[:-1])
    return key == cond_key


def matching(cond: Condition, findings: list[Finding]) -> list[Finding]:
    """The findings that satisfy one condition."""
    hits = []
    for f in findings:
        if f.kind != cond.kind or not _key_matches(cond.key, f.key):
            continue
        if cond.op == "exists":
            hits.append(f)
            continue
        actual = f.value if cond.field == "value" else f.evidence.get(cond.field)
        if actual is None or isinstance(actual, str):
            continue
        if _OPS[cond.op](actual, cond.value):
            hits.append(f)
    return hits


def evaluate(rules: tuple[Rule, ...] | list[Rule], findings: list[Finding]) -> list[Finding]:
    matches = []
    for rule in rules:
        all_hits = [matching(c, findings) for c in rule.all]
        if not all(all_hits):
            continue
        any_hits = [matching(c, findings) for c in rule.any]
        if rule.any and not any(any_hits):
            continue
        used = {f.id: f for hits in all_hits + any_hits for f in hits}
        matched = [f.id for f in findings if f.id in used]
        terms = tuple(sorted({t for f in used.values() for t in f.terms}))
        matches.append(Finding(
            id=rule.id, agent="rule", kind="rule", key=rule.id, statement=rule.says,
            evidence={"matched": matched, "title": rule.title, "source": rule.source,
                      "status": rule.status},
            terms=terms))
    return matches
