"""The knowledge file: review ranges, drug groups and rules, loaded and checked.

Everything clinical that is a judgement call -- what counts as low potassium,
which drugs are antiarrhythmics, what "frequent" ectopy means -- lives in
``configs/agent_rules.toml``, not in code, so a reviewer can read and change it
in one place. All of it is a draft until the professor reviews it; the loader
refuses a rule with no source and a condition naming a finding kind no agent
produces (a typo would otherwise make a rule silently never fire).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from common import paths

DEFAULT_PATH = paths.PROJECT_ROOT / "configs" / "agent_rules.toml"

#: Finding kinds the heart and chart agents produce.
KNOWN_KINDS = frozenset({
    "beats_classified", "class_share", "heart_rate", "v_run", "s_run",
    "low_confidence_beats", "q_share_high", "beat_source",
    "demographics", "admission_reason", "lab_abnormal", "vital_out_of_range",
    "not_measured", "medication", "medication_count", "risk_score",
})
OPS = (">=", ">", "<=", "<", "==", "exists")
RULE_ID = re.compile(r"^R\d+$")


class KnowledgeError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    kind: str
    key: str | None = None  # exact key, or "prefix:*" for any key with that prefix
    op: str = "exists"
    value: float | bool | None = None
    field: str = "value"  # "value", or the name of an evidence entry


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    says: str
    source: str
    status: str
    all: tuple[Condition, ...] = ()
    any: tuple[Condition, ...] = ()


@dataclass(frozen=True)
class Knowledge:
    status: str
    vital_ranges: dict[str, tuple[float, float]]
    lab_ranges: dict[str, tuple[float, float]]
    drug_groups: dict[str, tuple[str, ...]]
    rules: tuple[Rule, ...]


def _condition(raw: dict, where: str) -> Condition:
    unknown = set(raw) - {"kind", "key", "op", "value", "field"}
    if unknown:
        raise KnowledgeError(f"{where}: unknown condition fields {sorted(unknown)}")
    kind = raw.get("kind")
    if kind not in KNOWN_KINDS:
        raise KnowledgeError(f"{where}: unknown finding kind {kind!r}")
    op = raw.get("op", "exists")
    if op not in OPS:
        raise KnowledgeError(f"{where}: unknown op {op!r}")
    if op != "exists" and "value" not in raw:
        raise KnowledgeError(f"{where}: op {op!r} needs a value")
    return Condition(kind=kind, key=raw.get("key"), op=op, value=raw.get("value"),
                     field=raw.get("field", "value"))


def _ranges(raw: dict, section: str) -> dict[str, tuple[float, float]]:
    out = {}
    for name, pair in raw.get(section, {}).items():
        if not (isinstance(pair, list) and len(pair) == 2 and pair[0] < pair[1]):
            raise KnowledgeError(f"[{section}] {name}: expected [low, high] with low < high, "
                                 f"got {pair!r}")
        out[name] = (float(pair[0]), float(pair[1]))
    return out


def parse(raw: dict) -> Knowledge:
    status = str(raw.get("status", ""))
    rules, seen = [], set()
    for i, r in enumerate(raw.get("rule", [])):
        rid = str(r.get("id", ""))
        where = f"rule #{i + 1} ({rid or '?'})"
        if not RULE_ID.match(rid):
            raise KnowledgeError(f"{where}: id must look like R1, R2, ...")
        if rid in seen:
            raise KnowledgeError(f"{where}: duplicate id")
        seen.add(rid)
        for name in ("title", "says", "source"):
            if not str(r.get(name, "")).strip():
                hint = "; every rule needs a source" if name == "source" else ""
                raise KnowledgeError(f"{where}: missing {name}{hint}")
        conds_all = tuple(_condition(c, where) for c in r.get("all", []))
        conds_any = tuple(_condition(c, where) for c in r.get("any", []))
        if not conds_all and not conds_any:
            raise KnowledgeError(f"{where}: no conditions")
        rules.append(Rule(id=rid, title=r["title"], says=r["says"], source=r["source"],
                          status=str(r.get("status", status)), all=conds_all, any=conds_any))
    groups = {g: tuple(str(d).lower() for d in names)
              for g, names in raw.get("drug_groups", {}).items()}
    return Knowledge(status=status, vital_ranges=_ranges(raw, "vital_ranges"),
                     lab_ranges=_ranges(raw, "lab_ranges"), drug_groups=groups,
                     rules=tuple(rules))


def load_knowledge(path: Path | str = DEFAULT_PATH) -> Knowledge:
    with open(path, "rb") as handle:
        return parse(tomllib.load(handle))
