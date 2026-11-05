"""The knowledge file loads, refuses bad rules, and its rules fire on the right findings."""

from __future__ import annotations

import copy
import tomllib

import pytest

from agentic import knowledge, rules
from agentic.findings import Finding


def F(fid, kind, key="", value=None, terms=(), **evidence):
    return Finding(id=fid, agent="heart" if fid[0] == "H" else "chart", kind=kind, key=key,
                   statement=f"{kind} {key} {value}", value=value, evidence=evidence, terms=terms)


@pytest.fixture(scope="module")
def raw():
    with open(knowledge.DEFAULT_PATH, "rb") as h:
        return tomllib.load(h)


def test_real_file_loads_and_every_rule_has_a_source(raw):
    k = knowledge.parse(raw)
    assert len(k.rules) == 12
    assert all(r.source.strip() and "review" in r.status for r in k.rules)
    assert k.lab_ranges["potassium"] == (3.5, 5.0)
    assert "amiodarone" in k.drug_groups["antiarrhythmic"]


@pytest.mark.parametrize("change, message", [
    (lambda r: r["rule"][0].update(source=""), "every rule needs a source"),
    (lambda r: r["rule"][0].update(id="X1"), "id must look like"),
    (lambda r: r["rule"][1].update(id="R1"), "duplicate id"),
    (lambda r: r["rule"][0]["all"][0].update(kind="nonsense"), "unknown finding kind"),
    (lambda r: r["rule"][0]["all"][0].update(op="~"), "unknown op"),
    (lambda r: r["rule"][0].update({"all": [], "any": []}), "no conditions"),
    (lambda r: r["lab_ranges"].update(potassium=[5.0, 3.5]), "low < high"),
])
def test_bad_knowledge_is_refused(raw, change, message):
    broken = copy.deepcopy(raw)
    change(broken)
    with pytest.raises(knowledge.KnowledgeError, match=message):
        knowledge.parse(broken)


@pytest.fixture(scope="module")
def k(raw):
    return knowledge.parse(raw)


def ids(matches):
    return [m.id for m in matches]


def test_r1_fires_at_ten_percent_not_below(k):
    assert "R1" in ids(rules.evaluate(k.rules, [F("H1", "class_share", "V", 10.0)]))
    assert "R1" not in ids(rules.evaluate(k.rules, [F("H1", "class_share", "V", 9.9)]))


def test_r2_uses_the_run_rate_from_evidence(k):
    fast = F("H2", "v_run", "V", 1, max_rate_bpm=101)
    slow = F("H2", "v_run", "V", 1, max_rate_bpm=100)
    assert "R2" in ids(rules.evaluate(k.rules, [fast]))
    assert "R2" not in ids(rules.evaluate(k.rules, [slow]))


def test_r3_needs_both_ectopy_and_low_potassium(k):
    v = F("H1", "class_share", "V", 15.0, terms=("beat:V",))
    low_k = F("C1", "lab_abnormal", "potassium:low", 3.1, terms=("potassium",))
    match = [m for m in rules.evaluate(k.rules, [v, low_k]) if m.id == "R3"]
    assert match and match[0].evidence["matched"] == ["H1", "C1"]
    assert set(match[0].terms) == {"beat:V", "potassium"}
    assert "R3" not in ids(rules.evaluate(k.rules, [v]))
    assert "R3" not in ids(rules.evaluate(k.rules, [low_k]))


def test_r8_matches_any_troponin_key(k):
    assert "R8" in ids(rules.evaluate(k.rules, [F("C1", "lab_abnormal", "troponin_i:flagged", 0.03)]))
    assert "R8" not in ids(rules.evaluate(k.rules, [F("C1", "lab_abnormal", "potassium:low", 3.0)]))


def test_r12_reads_a_boolean_from_evidence(k):
    assert "R12" in ids(rules.evaluate(k.rules, [F("C1", "risk_score", "", 0.8, flagged=True)]))
    assert "R12" not in ids(rules.evaluate(k.rules, [F("C1", "risk_score", "", 0.2, flagged=False)]))


def test_no_findings_no_matches(k):
    assert rules.evaluate(k.rules, []) == []
