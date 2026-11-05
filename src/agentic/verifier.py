"""The fact-checker: an LLM text is shown only if every claim traces to a finding.

Deterministic rules, no second LLM:

1. the draft has the expected JSON shape;
2. every point cites at least one ID, and every ID exists in this case;
3. every number in a point appears in the facts it cites (or in the facts a
   cited rule matched), at the precision written -- "18%" may stand for 18.4;
4. every clinical word (beat class, lab, vital, drug group, drug) is backed by
   a cited fact's ``terms``;
5. no wrong beat terms ("V waves"), diagnosis or certainty language, treatment
   advice, doses, or talk of the patient's outcome.

Numbers and words in the summary are checked against all of the case's facts.
"""

from __future__ import annotations

import functools
import numbers
import re
from dataclasses import dataclass

from .findings import Case, Finding


@dataclass(frozen=True)
class Problem:
    code: str
    detail: str


NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")

#: Clinical phrases and the finding terms that may back them.
VOCAB: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in {
    "ventricular": {"beat:V"}, "v beat": {"beat:V"}, "v beats": {"beat:V"},
    "pvc": {"beat:V"}, "pvcs": {"beat:V"},
    "supraventricular": {"beat:S"}, "s beat": {"beat:S"}, "s beats": {"beat:S"},
    "atrial premature": {"beat:S"},
    "fusion": {"beat:F"}, "unclassifiable": {"beat:Q"}, "q beat": {"beat:Q"},
    "q beats": {"beat:Q"},
    "heart rate": {"heart_rate"}, "pulse": {"heart_rate"},
    "tachycardia": {"heart_rate", "run"}, "bradycardia": {"heart_rate"},
    "nsvt": {"run"}, "non-sustained": {"run"},
    "systolic": {"sbp"}, "diastolic": {"dbp"}, "mean arterial": {"mbp"}, "map": {"mbp"},
    "blood pressure": {"sbp", "dbp", "mbp"},
    "respiratory rate": {"resp_rate"}, "breathing rate": {"resp_rate"},
    "oxygen saturation": {"spo2"}, "spo2": {"spo2"},
    "temperature": {"temp_c"}, "fever": {"temp_c"},
    "troponin": {"troponin_t", "troponin_i"}, "ck-mb": {"ck_mb"},
    "creatinine": {"creatinine"}, "potassium": {"potassium", "potassium_supplement"},
    "sodium": {"sodium"}, "bnp": {"bnp"}, "natriuretic": {"bnp"}, "lactate": {"lactate"},
    "glucose": {"glucose"}, "hemoglobin": {"hemoglobin"}, "haemoglobin": {"hemoglobin"},
    "white blood": {"wbc"}, "wbc": {"wbc"},
    "antiarrhythmic": {"antiarrhythmic"}, "beta-blocker": {"beta_blocker"},
    "beta blocker": {"beta_blocker"}, "anticoagulant": {"anticoagulant"},
    "diuretic": {"diuretic"}, "vasopressor": {"vasopressor_inotrope"},
    "inotrope": {"vasopressor_inotrope"},
    "risk score": {"risk_score"},
}.items()}

BANNED = (
    ("wrong_term", re.compile(r"\b(?:[vsnfqp])[- ]waves?\b", re.I),
     "is not a term for beats; say 'V beats' or 'S beats'"),
    ("diagnosis_language",
     re.compile(r"\b(?:diagnos\w*|confirm\w*|definitely|certainly|proves?|suffer\w*)\b", re.I),
     "sounds like a diagnosis; describe what the findings show instead"),
    ("treatment_language",
     re.compile(r"\b(?:administer\w*|prescrib\w*|titrat\w*|should (?:be )?"
                r"(?:started|stopped|given|increased|decreased|start|stop|give|increase|decrease))\b",
                re.I),
     "is treatment advice"),
    ("dose", re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|units?|meq|mmol)\b(?!\s*/)", re.I),
     "is a dose"),
    ("outcome_language",
     re.compile(r"\b(?:died|dies|passed away|survived|survives|expired|was discharged|"
                r"went home)\b", re.I),
     "talks about the patient's outcome, which the AI is never given"),
)


@functools.lru_cache(maxsize=1)
def _vocab() -> tuple[tuple[str, frozenset[str], re.Pattern], ...]:
    from .knowledge import load_knowledge

    words = dict(VOCAB)
    for patterns in load_knowledge().drug_groups.values():
        for p in patterns:
            words[p] = frozenset({f"drug:{p}"})
    return tuple((phrase, keys, re.compile(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])"))
                 for phrase, keys in words.items())


def _collect_numbers(obj, out: set[float]) -> None:
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, numbers.Real):
        out.add(float(obj))
    elif isinstance(obj, str):
        out.update(float(n) for n in NUMBER.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def finding_numbers(f: Finding, by_id: dict[str, Finding]) -> set[float]:
    out: set[float] = set()
    for part in (f.value, f.statement, f.evidence, f.caveat):
        _collect_numbers(part, out)
    if f.agent == "rule":
        for mid in f.evidence.get("matched", []):
            if mid in by_id and by_id[mid].agent != "rule":
                out |= finding_numbers(by_id[mid], by_id)
    return out


def _supported(text: str, allowed: set[float]) -> bool:
    value = float(text)
    decimals = len(text.split(".")[1]) if "." in text else 0
    tol = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(value - a) <= tol for a in allowed)


def check_text(text: str, cited: list[Finding], by_id: dict[str, Finding],
               context: set[float], where: str) -> list[Problem]:
    problems = []
    allowed = set(context)
    terms: set[str] = set()
    for f in cited:
        allowed |= finding_numbers(f, by_id)
        terms |= set(f.terms)
    for n in NUMBER.findall(text):
        if not _supported(n, allowed):
            problems.append(Problem("unsupported_number",
                                    f"{where}: the number {n} is not in the facts it cites"))
    low = text.lower()
    for phrase, keys, pattern in _vocab():
        if pattern.search(low) and not keys & terms:
            problems.append(Problem("unsupported_term",
                                    f"{where}: '{phrase}' is not in the facts it cites"))
    for code, pattern, why in BANNED:
        m = pattern.search(text)
        if m:
            problems.append(Problem(code, f"{where}: '{m.group(0)}' {why}"))
    return problems


def _is_point(x) -> bool:
    return (isinstance(x, dict) and isinstance(x.get("text"), str)
            and isinstance(x.get("cites"), list) and all(isinstance(c, str) for c in x["cites"]))


def _report_shape(d) -> list[Problem]:
    if not isinstance(d, dict):
        return [Problem("schema", "the draft is not a JSON object")]
    out = []
    if not isinstance(d.get("summary"), str) or not d["summary"].strip():
        out.append(Problem("schema", "'summary' must be a non-empty string"))
    pts = d.get("points")
    if not isinstance(pts, list) or not pts or not all(_is_point(p) for p in pts):
        out.append(Problem("schema", "'points' must be a non-empty list of {text, cites[]}"))
    if not _is_point(d.get("for_clinician_review")):
        out.append(Problem("schema", "'for_clinician_review' must be {text, cites[]}"))
    return out


def _check_item(where: str, text: str, cites: list[str], by_id, context) -> list[Problem]:
    if not cites:
        return [Problem("no_citation", f"{where} cites no facts")]
    unknown = [c for c in cites if c not in by_id]
    if unknown:
        return [Problem("unknown_id", f"{where} cites {unknown}, which are not facts here")]
    return check_text(text, [by_id[c] for c in cites], by_id, context, where)


def check_report(draft, case: Case) -> list[Problem]:
    problems = _report_shape(draft)
    if problems:
        return problems
    by_id = case.by_id()
    context = set(case.context_numbers)
    problems += check_text(draft["summary"], case.all_findings(), by_id, context, "summary")
    for i, point in enumerate(draft["points"], 1):
        problems += _check_item(f"point {i}", point["text"], point["cites"], by_id, context)
    review = draft["for_clinician_review"]
    if review["text"].strip():
        problems += _check_item("for_clinician_review", review["text"], review["cites"],
                                by_id, context)
    return problems


def check_chat(draft, case: Case, glossary: list[Finding]) -> list[Problem]:
    if not (isinstance(draft, dict) and isinstance(draft.get("answerable"), bool)
            and isinstance(draft.get("answer"), str) and _is_point(
                {"text": draft.get("answer"), "cites": draft.get("cites")})):
        return [Problem("schema", "the answer must be {answerable: bool, answer: str, "
                                  "cites: [str]}")]
    if not draft["answerable"]:
        return []
    by_id = {**case.by_id(), **{g.id: g for g in glossary}}
    return _check_item("answer", draft["answer"], draft["cites"], by_id,
                       set(case.context_numbers))
