# Agents End to End (Part A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the heart agent, the chart agent, the knowledge rules, the template report, the local-LLM report with its fact-checker, the grounded chat agent and the LLM bake-off. The whole thing must run end to end from the terminal.

**Architecture:**
- **Agents:** each agent turns a trained model's output into `Finding` objects. The heart agent reads the MIT-BIH beat calls and the chart agent reads the MIMIC-III first 24 h. A finding is an ID, a code-written statement, values and evidence.
- **Rules:** rules from a TOML knowledge file match findings.
- **Orchestrator:** runs the stages in order (record window → beats → heart agent → chart agent → rules → report) and yields one event per stage. The CLI prints the events; the Part B web app will stream them.
- **The LLM:** a local Ollama model only words the report. A deterministic verifier must accept every claim before it is shown; otherwise the template report is shown.

**Tech Stack:**
- Python 3.13 in `venv/`.
- Libraries already installed: numpy, pandas, scikit-learn, torch, wfdb.
- Standard library: `tomllib`, `urllib`, `json`, `re`, `http.server` (in tests).
- Ollama 0.18.1 on `localhost:11434`, with `llama3.1:8b` and `qwen2.5` already pulled.
- pytest.

**Spec:** `PlanDoc/agentic_with_fe.md`, Part A (§1–§13). Errors, tests and permissions are in §15–§19.

## Global Constraints

- **No new dependencies.** Use only the standard library and packages already in `venv/`. Never `pip install`.
- **Running tests:** use `venv/bin/python -m pytest …` from the project root. `pytest.ini` already sets `pythonpath = src`.
- **Local LLM only.**
  - The LLM is reached only at `http://localhost:11434` or `http://127.0.0.1:11434`. The client refuses any other address before sending anything, and ignores proxy environment variables.
  - Never run `ollama pull`.
- **No commits.** `PERMISSIONS.md` says every commit needs the user's OK, so each task ends with a **Checkpoint** step instead of a commit.
- **Banner text, verbatim**, when both an ECG and a stay are present: `SIMULATED CASE: ECG from MIT-BIH record {record}; chart from MIMIC-III Demo ICU stay {stay}. These are different people. Research demo, not for patient care.`
- **No leakage into prompts.** The patient outcome (`label`, `hospital_expire_flag`, `deathtime`, discharge fields) and the cardiologist's beat labels never reach a finding, a rule or a prompt. Beat labels are display-only.
- **Medications:** use only `PRESCRIPTIONS` rows whose `startdate` is a day **before** the index day (`startdate < index_time.normalize()`).
- **No row-level clinical data in the repository.** Saved reports go under `<data root>/processed/agentic/`. Only aggregate numbers go into `PlanDoc/`.
- **Existing code stays as it is.** Do not modify `src/mit_bih`, `src/mimic_iii` or `src/multimodal`. The existing 633 tests must still pass.
- **Knowledge and glossary are drafts.** Every rule and glossary entry carries `status = "draft - needs professor review"` and a non-empty source.
- **Statements are written by code**, with numbers at their displayed precision. The LLM never writes a finding.

## Review Focus

1. **A window that runs past the end of the record**, or starts in its last seconds, must fail with a plain error naming the record length, never a traceback. Test: Task 3, `test_window_past_the_end_is_refused`.
2. **An ICU stay with no vitals or labs at all** (synthetic stay 401) must still give demographics, a "not measured" finding, medications and a risk score, with no crash. Test: Task 6, `test_stay_with_no_measurements`.
3. **LLM JSON with the wrong shape** (a numeric `summary`, `cites` as a string, missing keys) must be caught as a schema problem and lead to the template report, never a crash. Tests: Task 9, `test_schema_problems_are_reported`; Task 10, `test_wrong_shape_twice_falls_back`.
4. **Ollama down, or the model not pulled,** must lead to the template report with the reason shown, and the chat must refuse politely. Tests: Task 8, `test_status_when_nothing_listens` and `test_status_when_model_missing`; Task 10, `test_llm_unavailable_falls_back`; Task 11, `test_llm_error_refuses`.
5. **A window with fewer than 2 classified beats** must give no heart-rate finding and no division by zero. Test: Task 4, `test_one_beat_gives_no_heart_rate`.

---

## File map

| File | Responsibility |
|---|---|
| `src/agentic/__init__.py` | package docstring |
| `src/agentic/findings.py` | `Finding`, `FindingList`, `Case`, `ReportResult`, `make_banner` |
| `configs/agent_rules.toml` | review ranges, drug groups, 12 draft rules |
| `src/agentic/knowledge.py` | load and validate `agent_rules.toml` |
| `src/agentic/rules.py` | evaluate rules against findings |
| `src/agentic/ecg_window.py` | offered records with held-out/trained-on tags; load a record window with display-only truth |
| `src/agentic/heart_agent.py` | `BeatCalls` → heart findings |
| `src/agentic/chart_data.py` | load everything the chart agent reads, once, with the label dropped |
| `src/agentic/chart_agent.py` | one ICU stay → chart findings |
| `src/agentic/template_report.py` | the plain report |
| `src/agentic/orchestrator.py` | stages 1–9 as events; `default_deps` |
| `src/agentic/llm.py` | Ollama client, localhost-only |
| `src/agentic/verifier.py` | the fact-checker |
| `src/agentic/report_agent.py` | LLM report with one retry and fallback |
| `configs/glossary.toml`, `src/agentic/glossary.py` | reviewed definitions G1… |
| `src/agentic/chat_agent.py` | grounded question answering |
| `src/agentic/bakeoff.py` | score models on the same cases |
| `scripts/agentic/report.py` | CLI: list, run a case, ask questions |
| `scripts/agentic/compare_llms.py` | CLI: the bake-off |
| `tests/agentic/…` | tests; `fakes.py` holds `FakeLLM` and a hand-made case |
| modified: `pytest.ini`, `tests/test_scripts.py`, `README.md`, `COMMANDS.md`, `PlanDoc/IMPLEMENTATION.md`, `PlanDoc/agentic_with_fe.md` | marker, `--help` tests, docs |

---

### Task 1: Finding format, case, banner, test marker

**Files:**
- Create: `src/agentic/__init__.py`, `src/agentic/findings.py`
- Create: `tests/agentic/__init__.py`, `tests/agentic/test_findings.py`
- Modify: `pytest.ini`

**Interfaces:**
- Produces:
  - `Finding(id, agent, kind, key, statement, value=None, unit="", evidence={}, terms=(), caveat="")`, frozen, with `.to_dict()`.
  - `FindingList(agent).add(kind, key, statement, value=None, unit="", evidence=None, terms=(), caveat="") -> Finding`. IDs are `H1…` for `"heart"` and `C1…` for `"chart"`.
  - `make_banner(record: str|None, stay: int|None, source="demo") -> str`.
  - `Case(banner, record=None, stay=None, findings=[], rules=[], context_numbers=(), display_only={})` with `.all_findings()`, `.by_id()` and `.facts_for_llm() -> list[dict]`.
  - `ReportResult(kind, text, model=None, seconds=0.0, attempts=0, problems=(), note="", draft=None)`.

- [ ] **Step 1: Write the failing tests** in `tests/agentic/test_findings.py` (plus an empty `tests/agentic/__init__.py`):

```python
"""The shared finding format and the case that carries it."""

from __future__ import annotations

import json

import pytest

from agentic.findings import Case, FindingList, make_banner


def test_ids_are_numbered_per_agent():
    heart, chart = FindingList("heart"), FindingList("chart")
    a = heart.add("class_share", "V", "10.0% V")
    b = heart.add("heart_rate", "", "80 beats/min")
    c = chart.add("demographics", "", "Age 70")
    assert (a.id, b.id, c.id) == ("H1", "H2", "C1")


def test_unknown_agent_is_refused():
    with pytest.raises(ValueError):
        FindingList("llm")


def test_banner_says_simulated_only_when_both_sources_are_present():
    both = make_banner("208", 200001)
    assert both == ("SIMULATED CASE: ECG from MIT-BIH record 208; chart from MIMIC-III Demo ICU "
                    "stay 200001. These are different people. Research demo, not for patient care.")
    assert "SIMULATED" not in make_banner("208", None)
    assert "SIMULATED" not in make_banner(None, 200001)


def test_banner_needs_something():
    with pytest.raises(ValueError):
        make_banner(None, None)


def test_facts_for_llm_never_include_display_only_or_evidence():
    fl = FindingList("heart")
    fl.add("class_share", "V", "20.0% V", value=20.0, unit="%", evidence={"secret": 1},
           caveat="model output")
    case = Case(banner="b", findings=fl.items, display_only={"truth": ["V", "N"], "agree": 1})
    facts = case.facts_for_llm()
    assert facts == [{"id": "H1", "statement": "20.0% V", "value": 20.0, "unit": "%",
                      "caveat": "model output"}]
    text = json.dumps(facts)
    assert "truth" not in text and "secret" not in text and "agree" not in text


def test_to_dict_round_trips_terms_as_a_list():
    f = FindingList("chart").add("lab_abnormal", "potassium:low", "low", terms=("potassium",))
    assert f.to_dict()["terms"] == ["potassium"]
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_findings.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic'`.

- [ ] **Step 3: Implement.** `src/agentic/__init__.py`:

```python
"""Agentic pipeline (Part A): agents turn model outputs into findings, rules
connect them, a local LLM words the report, and a fact-checker decides
whether anyone sees the LLM's text. See PlanDoc/agentic_with_fe.md."""
```

`src/agentic/findings.py`:

```python
"""The one format every agent speaks.

A :class:`Finding` is one fact with its evidence. The heart agent, the chart
agent and the rules produce findings; the report, the chat and the
fact-checker read them. ``statement`` is always written by code from the
values -- never by the LLM -- so every number a reader sees traces back to a
model output or a chart value.

A :class:`Case` is everything known about one (possibly simulated) patient.
``display_only`` holds what a person may look at but no LLM may see: the
cardiologist's beat labels. :meth:`Case.facts_for_llm` is the only way facts
reach a prompt, and it never reads ``display_only`` or ``evidence``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PREFIX = {"heart": "H", "chart": "C"}


@dataclass(frozen=True)
class Finding:
    id: str
    agent: str  # "heart" | "chart" | "rule" | "glossary"
    kind: str
    key: str
    statement: str
    value: float | int | str | None = None
    unit: str = ""
    evidence: dict = field(default_factory=dict)
    #: Vocabulary keys this finding can back (see verifier.VOCAB), e.g. "beat:V".
    terms: tuple[str, ...] = ()
    caveat: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        out["terms"] = list(self.terms)
        return out


class FindingList:
    """Collects one agent's findings and numbers them H1, H2, ... or C1, C2, ..."""

    def __init__(self, agent: str) -> None:
        if agent not in PREFIX:
            raise ValueError(f"no ID prefix for agent {agent!r}")
        self.agent = agent
        self.items: list[Finding] = []

    def add(self, kind: str, key: str, statement: str, value=None, unit: str = "",
            evidence: dict | None = None, terms=(), caveat: str = "") -> Finding:
        finding = Finding(id=f"{PREFIX[self.agent]}{len(self.items) + 1}", agent=self.agent,
                          kind=kind, key=key, statement=statement, value=value, unit=unit,
                          evidence=dict(evidence or {}), terms=tuple(terms), caveat=caveat)
        self.items.append(finding)
        return finding


def make_banner(record: str | None, stay: int | None, source: str = "demo") -> str:
    data = "MIMIC-III Demo" if source == "demo" else "MIMIC-III"
    if record is not None and stay is not None:
        return (f"SIMULATED CASE: ECG from MIT-BIH record {record}; chart from {data} ICU stay "
                f"{stay}. These are different people. Research demo, not for patient care.")
    if record is not None:
        return f"ECG from MIT-BIH record {record}. Research demo, not for patient care."
    if stay is not None:
        return f"Chart from {data} ICU stay {stay}. Research demo, not for patient care."
    raise ValueError("a case needs an ECG record, an ICU stay, or both")


@dataclass
class Case:
    banner: str
    record: str | None = None
    stay: int | None = None
    findings: list[Finding] = field(default_factory=list)
    rules: list[Finding] = field(default_factory=list)
    #: Numbers a text may use besides the findings: record, stay, window, 24 h.
    context_numbers: tuple[float, ...] = ()
    #: For people only -- never given to any LLM.
    display_only: dict = field(default_factory=dict)

    def all_findings(self) -> list[Finding]:
        return [*self.findings, *self.rules]

    def by_id(self) -> dict[str, Finding]:
        return {f.id: f for f in self.all_findings()}

    def facts_for_llm(self) -> list[dict]:
        """What a prompt may contain: ID, statement, value, unit, caveat. Nothing else."""
        return [{"id": f.id, "statement": f.statement, "value": f.value, "unit": f.unit,
                 **({"caveat": f.caveat} if f.caveat else {})} for f in self.all_findings()]


@dataclass(frozen=True)
class ReportResult:
    kind: str  # "llm" | "template"
    text: str
    model: str | None = None
    seconds: float = 0.0
    attempts: int = 0
    problems: tuple = ()  # verifier.Problem items from the last rejected draft
    note: str = ""
    draft: dict | None = None
```

In `pytest.ini`, replace the `addopts` line and add the marker. The block becomes:

```ini
addopts = -m "not requires_data and not slow and not requires_ollama"
markers =
    requires_data: needs a real dataset on disk (skipped by default)
    slow: trains real networks; minutes not seconds (skipped by default).
        Run before accepting any change to losses.py, train.py or the model:
        pytest tests/mit_bih/test_imbalance_regression.py -m slow
    requires_ollama: needs the local Ollama server with llama3.1:8b pulled (skipped by default):
        pytest tests/agentic -m requires_ollama
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_findings.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Checkpoint.** Do not commit (the user's OK is needed). Run `git status --short src/agentic tests/agentic pytest.ini` and confirm only these files are new or changed.

---

### Task 2: Knowledge file and rules engine

**Files:**
- Create: `configs/agent_rules.toml`, `src/agentic/knowledge.py`, `src/agentic/rules.py`
- Test: `tests/agentic/test_knowledge_rules.py`

**Interfaces:**
- Consumes: `Finding` (Task 1).
- Produces:
  - `knowledge.load_knowledge(path=DEFAULT_PATH) -> Knowledge`
  - `knowledge.parse(raw: dict) -> Knowledge`
  - `Knowledge(status, vital_ranges: dict[str, (lo, hi)], lab_ranges, drug_groups: dict[str, tuple[str,...]], rules: tuple[Rule,...])`
  - `Rule(id, title, says, source, status, all, any)`
  - `Condition(kind, key=None, op="exists", value=None, field="value")`
  - `KnowledgeError(ValueError)` and `KNOWN_KINDS`
  - `rules.evaluate(rules, findings) -> list[Finding]`: each match has `id=rule.id`, `agent="rule"`, `kind="rule"`, `evidence={"matched": [ids], "title", "source", "status"}`, and `terms` set to the union of the matched findings' terms.

- [ ] **Step 1: Write the failing tests** `tests/agentic/test_knowledge_rules.py`:

```python
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
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_knowledge_rules.py -q`
Expected: FAIL (`agentic.knowledge` not found).

- [ ] **Step 3: Write `configs/agent_rules.toml`:**

```toml
# Knowledge for the agent pipeline: review ranges, drug groups and rules.
#
# DRAFT. Every range, threshold and rule below is a starting point written for
# the demo and needs review by the professor. Read by src/agentic/knowledge.py,
# which refuses a rule without a source.
#
# Ranges are adult reference ranges used only to FLAG values and to give an
# abnormal lab its direction (low/high). They are not treatment targets.

status = "draft - needs professor review"

[vital_ranges]
heart_rate = [60, 100]   # beats/min
sbp = [90, 140]          # mmHg
dbp = [60, 90]           # mmHg
mbp = [65, 110]          # mmHg
resp_rate = [12, 20]     # breaths/min
spo2 = [94, 100]         # %
temp_c = [36.0, 38.0]    # degC

[lab_ranges]
potassium = [3.5, 5.0]       # mEq/L
sodium = [135, 145]          # mEq/L
lactate = [0.5, 2.0]         # mmol/L
creatinine = [0.5, 1.2]      # mg/dL
glucose = [70, 180]          # mg/dL
hemoglobin = [12.0, 17.5]    # g/dL
wbc = [4.0, 11.0]            # K/uL
bnp = [0, 100]               # pg/mL
troponin_t = [0, 0.01]       # ng/mL
troponin_i = [0, 0.04]       # ng/mL
ck_mb = [0, 5]               # ng/mL

# A drug belongs to a group when its PRESCRIPTIONS name contains one of these
# lower-case words. "heparin" also matches line flushes; acceptable for a demo.
[drug_groups]
antiarrhythmic = ["amiodarone", "lidocaine", "procainamide", "sotalol", "dofetilide", "flecainide", "propafenone", "mexiletine"]
beta_blocker = ["metoprolol", "carvedilol", "atenolol", "esmolol", "labetalol", "propranolol", "bisoprolol"]
anticoagulant = ["heparin", "warfarin", "enoxaparin", "apixaban", "rivaroxaban", "dabigatran", "bivalirudin", "argatroban"]
diuretic = ["furosemide", "bumetanide", "torsemide", "hydrochlorothiazide", "spironolactone", "metolazone"]
potassium_supplement = ["potassium chloride", "potassium phosphate", "kcl"]
digoxin = ["digoxin"]
vasopressor_inotrope = ["norepinephrine", "epinephrine", "dopamine", "dobutamine", "vasopressin", "phenylephrine", "milrinone"]

[[rule]]
id = "R1"
title = "Frequent ventricular ectopy"
all = [{ kind = "class_share", key = "V", op = ">=", value = 10 }]
says = "Frequent ventricular ectopic (V) beats in this ECG window."
source = "Project draft threshold (V beats at least 10% of the window); threshold to review"

[[rule]]
id = "R2"
title = "Run consistent with non-sustained VT"
all = [{ kind = "v_run", op = ">", value = 100, field = "max_rate_bpm" }]
says = "A run of 3 or more V beats faster than 100/min, consistent with the definition of non-sustained VT; needs clinician review."
source = "2017 AHA/ACC/HRS guideline for ventricular arrhythmias and the prevention of sudden cardiac death (NSVT definition); to confirm"

[[rule]]
id = "R3"
title = "Ventricular ectopy with low potassium"
all = [{ kind = "lab_abnormal", key = "potassium:low" }]
any = [
  { kind = "class_share", key = "V", op = ">=", value = 10 },
  { kind = "v_run" },
]
says = "Ventricular ectopy together with low potassium: a combination worth checking."
source = "Low potassium is a recognised contributor to ventricular arrhythmias; source to confirm"

[[rule]]
id = "R4"
title = "Fast heart rate on the ECG"
all = [{ kind = "heart_rate", op = ">", value = 100 }]
says = "Heart rate above 100/min in this ECG window (tachycardia)."
source = "Standard adult definition of tachycardia (above 100 beats/min); to confirm"

[[rule]]
id = "R5"
title = "Slow heart rate on the ECG"
all = [{ kind = "heart_rate", op = "<", value = 60 }]
says = "Heart rate below 60/min in this ECG window (bradycardia)."
source = "Standard adult definition of bradycardia (below 60 beats/min); to confirm"

[[rule]]
id = "R6"
title = "Frequent supraventricular ectopy"
any = [
  { kind = "class_share", key = "S", op = ">=", value = 10 },
  { kind = "s_run" },
]
says = "Frequent supraventricular ectopic (S) beats in this ECG window."
source = "Project draft threshold (S beats at least 10% of the window, or a run); threshold to review"

[[rule]]
id = "R7"
title = "Raised lactate"
all = [{ kind = "lab_abnormal", key = "lactate:high" }]
says = "Raised lactate before the 24-hour mark."
source = "Sepsis-3 (Singer et al., JAMA 2016) uses lactate above 2 mmol/L; to confirm"

[[rule]]
id = "R8"
title = "Troponin above the reference"
any = [
  { kind = "lab_abnormal", key = "troponin_t:*" },
  { kind = "lab_abnormal", key = "troponin_i:*" },
]
says = "Troponin above the reference: a marker of heart-muscle injury."
source = "Fourth Universal Definition of Myocardial Infarction (Thygesen et al., 2018); to confirm"

[[rule]]
id = "R9"
title = "Low mean arterial pressure"
all = [{ kind = "vital_out_of_range", key = "mbp:low" }]
says = "Mean arterial pressure fell below 65 mmHg at some point before the 24-hour mark."
source = "Surviving Sepsis Campaign initial MAP target of 65 mmHg; to confirm"

[[rule]]
id = "R10"
title = "Ectopy on an antiarrhythmic"
all = [{ kind = "medication", key = "antiarrhythmic" }]
any = [
  { kind = "class_share", key = "V", op = ">=", value = 10 },
  { kind = "v_run" },
]
says = "Ventricular ectopy although an antiarrhythmic had already been started."
source = "Project draft; to confirm"

[[rule]]
id = "R11"
title = "Many unclassifiable beats"
all = [{ kind = "q_share_high" }]
says = "Many beats were unclassifiable; ECG quality may limit these findings."
source = "Project rule (model limitation)"

[[rule]]
id = "R12"
title = "Demo risk score above threshold"
all = [{ kind = "risk_score", field = "flagged", op = "==", value = true }]
says = "The Demo risk model scored this stay above its threshold (not validated)."
source = "Project rule (MIMIC-III Demo model)"
```

- [ ] **Step 4: Write `src/agentic/knowledge.py`:**

```python
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
```

- [ ] **Step 5: Write `src/agentic/rules.py`:**

```python
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
```

- [ ] **Step 6: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_knowledge_rules.py -q`
Expected: `14 passed`. (The parametrized test counts 7.)

- [ ] **Step 7: Checkpoint.** No commit. Run `git status --short configs src/agentic tests/agentic`.

---

### Task 3: ECG record window

**Files:**
- Create: `src/agentic/ecg_window.py`, `tests/agentic/conftest.py`, `tests/agentic/test_ecg_window.py`

**Interfaces:**
- Consumes:
  - `mit_bih.io.load_record(dir, name) -> LoadedRecord(signal, lead, fs, n_samples, duration_s)`
  - `mit_bih.io.load_annotations(dir, name) -> LoadedAnnotations(samples, symbols)`
  - `mit_bih.labels.map_symbols(symbols)`, which returns items with `.is_beat` and `.aami_class`
  - `mit_bih.splits.DS1_RECORDS` and `DS2_RECORDS`
  - `mit_bih.config.ALL_RECORDS`, `PACED_RECORDS` and `mitdb_dir()`
- Produces:
  - `RecordInfo(name, tag, note)`
  - `offered_records() -> list[RecordInfo]`
  - `WindowError(ValueError)`
  - `EcgWindow(record: RecordInfo, lead, fs, start_s, signal, r_peaks, truth)`. Here `r_peaks` is relative to the window and `truth` is display only.
  - `load_window(record, start_s, seconds, record_dir=None) -> EcgWindow`
- Test fixtures produced in `tests/agentic/conftest.py`, reused by later tasks:
  - `knowledge`, `tiny_bundle`, `tiny_classifier`
  - `mitdb_dir`: a synthetic record named `"100"` with 40 beats 300 samples apart; beats 10–12 are annotated V, the rest N.
  - `chart_env`, `chart_sources`: added in Task 5.

- [ ] **Step 1: Write the conftest** `tests/agentic/conftest.py` (the chart fixtures are added in Task 5):

```python
"""Fixtures for the agent tests: the knowledge file, a tiny ECG ensemble, a
synthetic MIT-BIH record and (Task 5) a synthetic MIMIC-III database with a
trained model. No real patient data."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agentic.knowledge import load_knowledge  # noqa: E402
from common import provenance  # noqa: E402
from mit_bih import config as ecg_config  # noqa: E402
from mit_bih import inference as ecg_inference  # noqa: E402
from mit_bih.nn import checkpoint  # noqa: E402
from mit_bih.nn.model import HybridBeatNet  # noqa: E402

from ..mit_bih.synthetic import make_ecg, write_record  # noqa: E402

#: Beats of the synthetic record "100" annotated V; every other beat is N.
V_BEATS = (10, 11, 12)


@pytest.fixture(scope="session")
def knowledge():
    return load_knowledge()


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory):
    """Two randomly initialised hybrid networks behind a real bundle file."""
    d = tmp_path_factory.mktemp("ecg")
    members = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        p = checkpoint.save_checkpoint(d / f"m{seed}.pt", HybridBeatNet().state_dict(),
                                       arm="test", use_rr=True)
        members.append({"path": str(p), "sha256": provenance.sha256_file(p)})
    bundle = ecg_inference.InferenceBundle(
        fs=ecg_config.FS, bandpass_low_hz=0.5, bandpass_high_hz=40.0, bandpass_order=3,
        notch=False, samples_before=100, samples_after=200, norm_mean=0.0, norm_std=0.4,
        rr_robust=False, classes=tuple(ecg_config.AAMI_CLASSES), ensemble="test",
        members=tuple(members))
    return ecg_inference.save_bundle(d / "bundle.json", bundle)


@pytest.fixture(scope="session")
def tiny_classifier(tiny_bundle):
    return ecg_inference.EcgClassifier.from_bundle(tiny_bundle)


@pytest.fixture(scope="session")
def mitdb_dir(tmp_path_factory):
    """Record "100" (a held-out name): 40 beats 300 samples apart (34.4 s)."""
    sig, peaks = make_ecg(n_beats=40, rr_samples=300)
    symbols = ["+"] + ["V" if i in V_BEATS else "N" for i in range(len(peaks))]
    samples = np.concatenate([[int(peaks[0]) - 50], peaks])
    directory = tmp_path_factory.mktemp("mitdb")
    write_record(directory, "100", {"MLII": sig, "V5": sig * 0.6},
                 ann_samples=samples, ann_symbols=symbols)
    return directory
```

- [ ] **Step 2: Write the failing tests** `tests/agentic/test_ecg_window.py`:

```python
"""Which records are offered, and how a window is cut from one."""

from __future__ import annotations

import pytest

from agentic import ecg_window


def test_offered_records_are_44_tagged_and_paced_ones_left_out():
    recs = {r.name: r for r in ecg_window.offered_records()}
    assert len(recs) == 44
    assert sum(r.tag == "held-out" for r in recs.values()) == 22
    assert sum(r.tag == "trained-on" for r in recs.values()) == 22
    assert not {"102", "104", "107", "217"} & set(recs)
    assert recs["208"].tag == "trained-on" and recs["100"].tag == "held-out"
    assert "201" in recs["202"].note and "202" in recs["201"].note


def test_window_peaks_are_relative_and_truth_holds_beats_only(mitdb_dir):
    win = ecg_window.load_window("100", start_s=2.0, seconds=20.0, record_dir=mitdb_dir)
    assert win.signal.size == 7200 and win.fs == 360 and win.lead == "MLII"
    assert win.r_peaks[0] == 80 and len(win.r_peaks) == 24  # beats 2..25 of the record
    assert len(win.truth) == 24 and win.truth.count("V") == 3 and "+" not in win.truth
    assert win.record.tag == "held-out"


def test_window_past_the_end_is_refused(mitdb_dir):
    with pytest.raises(ecg_window.WindowError, match="34.4 s long"):
        ecg_window.load_window("100", start_s=30.0, seconds=20.0, record_dir=mitdb_dir)


@pytest.mark.parametrize("record, start, seconds, message", [
    ("102", 0, 20, "paced"),
    ("999", 0, 20, "unknown record"),
    ("100", -1, 20, "start"),
    ("100", 0, 0, "seconds"),
])
def test_bad_requests_are_refused(mitdb_dir, record, start, seconds, message):
    with pytest.raises(ecg_window.WindowError, match=message):
        ecg_window.load_window(record, start, seconds, record_dir=mitdb_dir)
```

- [ ] **Step 3: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_ecg_window.py -q`
Expected: FAIL (`agentic.ecg_window` not found).

- [ ] **Step 4: Implement** `src/agentic/ecg_window.py`:

```python
"""Which MIT-BIH records the demo offers, and cutting a window from one.

Every offered record is tagged. **Held-out** records (DS2) were never used to
train the ECG model, so results on them are an honest view. **Trained-on**
records (DS1) were, so results on them look better than they are. Records 201
and 202 are the same person, which the notes say. The four paced records are
not offered: the model was never meant for paced rhythms.

The window's beat positions are the cardiologist's annotations, as in
training. The annotation *labels* are kept as ``truth`` for display only; the
orchestrator never gives them to an agent or to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mit_bih import config as ecg_config
from mit_bih import io as ecg_io
from mit_bih import labels as ecg_labels
from mit_bih import splits

NOTES = {"201": "same person as record 202 (held-out)",
         "202": "same person as record 201 (trained-on)"}


class WindowError(ValueError):
    pass


@dataclass(frozen=True)
class RecordInfo:
    name: str
    tag: str  # "held-out" | "trained-on"
    note: str = ""


def offered_records() -> list[RecordInfo]:
    out = []
    for name in ecg_config.ALL_RECORDS:
        if name in ecg_config.PACED_RECORDS:
            continue
        tag = "held-out" if name in splits.DS2_RECORDS else "trained-on"
        out.append(RecordInfo(name, tag, NOTES.get(name, "")))
    return out


@dataclass(frozen=True)
class EcgWindow:
    record: RecordInfo
    lead: str
    fs: int
    start_s: float
    signal: np.ndarray  # the window only
    r_peaks: np.ndarray  # annotated beat positions, relative to the window start
    truth: tuple[str, ...]  # AAMI class per r_peak -- DISPLAY ONLY


def load_window(record: str, start_s: float, seconds: float,
                record_dir: Path | str | None = None) -> EcgWindow:
    record = str(record)
    info = {r.name: r for r in offered_records()}.get(record)
    if info is None:
        if record in ecg_config.PACED_RECORDS:
            raise WindowError(f"record {record} is paced; paced records are outside what the "
                              f"model was built for")
        raise WindowError(f"unknown record {record!r}; see --list")
    if start_s < 0:
        raise WindowError(f"start must be 0 s or later, got {start_s:g}")
    if seconds <= 0:
        raise WindowError(f"seconds must be positive, got {seconds:g}")

    directory = Path(record_dir or ecg_config.mitdb_dir()).expanduser()
    rec = ecg_io.load_record(directory, record)
    lo = int(round(start_s * rec.fs))
    hi = lo + int(round(seconds * rec.fs))
    if hi > rec.n_samples:
        raise WindowError(f"the window ends at {hi / rec.fs:.1f} s but record {record} is "
                          f"{rec.duration_s:.1f} s long")

    ann = ecg_io.load_annotations(directory, record)
    maps = ecg_labels.map_symbols(ann.symbols)
    keep = [i for i, m in enumerate(maps) if m.is_beat and lo <= ann.samples[i] < hi]
    peaks = (ann.samples[keep] - lo).astype(np.int64)
    truth = tuple(maps[i].aami_class for i in keep)
    return EcgWindow(info, rec.lead, rec.fs, float(start_s), rec.signal[lo:hi].copy(),
                     peaks, truth)
```

- [ ] **Step 5: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_ecg_window.py -q`
Expected: `7 passed`.

- [ ] **Step 6: Checkpoint.** No commit.

---

### Task 4: Heart agent

**Files:**
- Create: `src/agentic/heart_agent.py`, `tests/agentic/test_heart_agent.py`

**Interfaces:**
- Consumes: `FindingList` (Task 1); `mit_bih.inference.EcgPrediction` (`.scores`, `.r_peaks`, `.classes`, `.predicted`, and `.prepared` with `.n_input`, `.n_dropped_boundary`, `.n_dropped_edge`).
- Produces:
  - `BeatCalls(classes, predicted, scores, r_peaks, fs, n_input, n_dropped, source)`, plus `BeatCalls.from_prediction(pred, fs, source)`.
  - `runs(predicted, target, min_len=3) -> list[tuple[int,int]]`.
  - `analyze(calls) -> list[Finding]`. The kinds are those in §8 of the spec, and every key is a class letter or `""`.

- [ ] **Step 1: Write the failing tests** `tests/agentic/test_heart_agent.py`:

```python
"""The heart agent turns beat calls into rhythm findings."""

from __future__ import annotations

import numpy as np

from agentic.heart_agent import BeatCalls, analyze, runs

CLASSES = ("N", "S", "V", "F", "Q")


def calls(seq, rr=0.8, top=0.9, n_dropped=2, source="annotation", fs=360):
    idx = np.array([CLASSES.index(c) for c in seq], dtype=np.int64)
    scores = np.full((len(seq), 5), (1 - top) / 4)
    scores[np.arange(len(seq)), idx] = top
    gaps = np.full(max(len(seq) - 1, 0), round(rr * fs))
    peaks = (100 + np.concatenate([[0], np.cumsum(gaps)])).astype(np.int64)[:len(seq)]
    return BeatCalls(CLASSES, idx, scores, peaks, fs, len(seq) + n_dropped, n_dropped, source)


def by_kind(findings):
    out = {}
    for f in findings:
        out.setdefault(f.kind, []).append(f)
    return out


def test_class_shares_skip_absent_classes():
    k = by_kind(analyze(calls("NNNNNNNNVV")))
    shares = {f.key: f.value for f in k["class_share"]}
    assert shares == {"N": 80.0, "V": 20.0}
    assert k["class_share"][1].statement.startswith("20.0% of beats (2 of 10)")
    assert k["class_share"][1].terms == ("beat:V",)


def test_heart_rate_from_rr():
    hr = by_kind(analyze(calls("NNNNN", rr=0.75)))["heart_rate"][0]
    assert hr.value == 80 and hr.evidence == {"min": 80, "max": 80}


def test_runs_of_three_counted_runs_of_two_ignored():
    assert runs(np.array([0, 0, 2, 2, 2, 0, 0, 2, 2, 0]), 2) == [(2, 4)]
    f = by_kind(analyze(calls("NNVVVNNVVN", rr=0.5)))["v_run"][0]
    assert f.value == 1 and f.evidence["lengths"] == [3] and f.evidence["max_rate_bpm"] == 120
    assert "v_run" not in by_kind(analyze(calls("NNVVNNVVNN")))


def test_low_confidence_and_q_share():
    k = by_kind(analyze(calls("Q" + "N" * 19, top=0.4)))
    assert k["low_confidence_beats"][0].value == 20
    assert k["q_share_high"][0].value == 5.0


def test_one_beat_gives_no_heart_rate():
    k = by_kind(analyze(calls("N")))
    assert "heart_rate" not in k and k["beats_classified"][0].value == 1


def test_no_beats_gives_one_finding():
    findings = analyze(calls(""))
    assert [f.kind for f in findings] == ["beats_classified"]
    assert findings[0].value == 0 and "no beats" in findings[0].statement


def test_ids_and_beat_source():
    findings = analyze(calls("NNNN", source="detected"))
    assert [f.id for f in findings] == [f"H{i}" for i in range(1, len(findings) + 1)]
    assert "detected automatically" in by_kind(findings)["beat_source"][0].statement
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_heart_agent.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/agentic/heart_agent.py`:

```python
"""Heart agent: the ECG model's beat-by-beat calls, turned into rhythm findings.

Input is :class:`BeatCalls` -- plain arrays -- so this module needs no torch
and can be tested with hand-made calls. Beats are dropped only at the window
edges (no full window, or no RR neighbour), so the classified beats are
consecutive and a "run" is simply consecutive calls of one class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .findings import Finding, FindingList

CLASS_NAMES = {"N": "normal", "S": "supraventricular ectopic", "V": "ventricular ectopic",
               "F": "fusion", "Q": "unclassifiable"}
MIN_RUN = 3
LOW_CONFIDENCE = 0.5
Q_SHARE_FLAG = 5.0  # percent
MODEL_CAVEAT = "classified by the ECG model (scores uncalibrated), not by a cardiologist"


@dataclass(frozen=True)
class BeatCalls:
    classes: tuple[str, ...]
    predicted: np.ndarray  # (n,) class index
    scores: np.ndarray  # (n, k)
    r_peaks: np.ndarray  # (n,) sample index
    fs: int
    n_input: int
    n_dropped: int
    source: str  # "annotation" | "detected"

    @classmethod
    def from_prediction(cls, pred, fs: int, source: str) -> "BeatCalls":
        p = pred.prepared
        return cls(tuple(pred.classes), np.asarray(pred.predicted, dtype=np.int64),
                   np.asarray(pred.scores), np.asarray(pred.r_peaks, dtype=np.int64), int(fs),
                   int(p.n_input), int(p.n_dropped_boundary + p.n_dropped_edge), source)


def runs(predicted: np.ndarray, target: int, min_len: int = MIN_RUN) -> list[tuple[int, int]]:
    """Inclusive (first, last) index of each run of ``target`` at least ``min_len`` long."""
    out, start = [], None
    for i, p in enumerate([*np.asarray(predicted).tolist(), -1]):
        if p == target and start is None:
            start = i
        elif p != target and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def _rate(peaks: np.ndarray, fs: int) -> int:
    return int(round(60.0 / (np.diff(peaks).mean() / fs)))


def analyze(calls: BeatCalls) -> list[Finding]:
    fl = FindingList("heart")
    n = int(calls.predicted.size)
    if n == 0:
        fl.add("beats_classified", "", "no beats could be classified in this window", value=0,
               unit="beats", evidence={"n_input": calls.n_input, "n_dropped": calls.n_dropped},
               terms=("beat",))
        return fl.items

    fl.add("beats_classified", "",
           f"{n} beats classified ({calls.n_dropped} at the window edges could not be)",
           value=n, unit="beats", evidence={"n_input": calls.n_input, "n_dropped": calls.n_dropped},
           terms=("beat",))

    counts = np.bincount(calls.predicted, minlength=len(calls.classes))
    for i, c in enumerate(calls.classes):
        if counts[i] == 0:
            continue
        share = round(100.0 * counts[i] / n, 1)
        fl.add("class_share", c,
               f"{share:.1f}% of beats ({int(counts[i])} of {n}) were classified {c} "
               f"({CLASS_NAMES.get(c, c)})",
               value=share, unit="%", evidence={"count": int(counts[i]), "total": n},
               terms=(f"beat:{c}",), caveat=MODEL_CAVEAT)

    if n >= 2:
        rr = np.diff(calls.r_peaks) / calls.fs
        hr, lo, hi = (int(round(60.0 / x)) for x in (rr.mean(), rr.max(), rr.min()))
        fl.add("heart_rate", "",
               f"average heart rate {hr} beats/min in this window (range {lo} to {hi})",
               value=hr, unit="beats/min", evidence={"min": lo, "max": hi}, terms=("heart_rate",))

    for c, kind in (("V", "v_run"), ("S", "s_run")):
        if c not in calls.classes:
            continue
        found = runs(calls.predicted, calls.classes.index(c))
        if not found:
            continue
        lengths = [b - a + 1 for a, b in found]
        rates = [_rate(calls.r_peaks[a:b + 1], calls.fs) for a, b in found]
        fl.add(kind, c,
               f"{len(found)} run(s) of {MIN_RUN} or more consecutive {c} beats "
               f"(longest {max(lengths)} beats, fastest {max(rates)} beats/min)",
               value=len(found), unit="runs",
               evidence={"lengths": lengths, "rates_bpm": rates, "max_rate_bpm": max(rates),
                         "beat_indices": [[int(a), int(b)] for a, b in found]},
               terms=(f"beat:{c}", "run"), caveat=MODEL_CAVEAT)

    low = int((calls.scores.max(axis=1) < LOW_CONFIDENCE).sum())
    if low:
        fl.add("low_confidence_beats", "",
               f"{low} of {n} beats had a top score below {LOW_CONFIDENCE} (the model was unsure)",
               value=low, unit="beats", terms=("beat",), caveat=MODEL_CAVEAT)

    if "Q" in calls.classes:
        q = round(100.0 * counts[calls.classes.index("Q")] / n, 1)
        if q >= Q_SHARE_FLAG:
            fl.add("q_share_high", "Q",
                   f"{q:.1f}% of beats were unclassifiable (Q); signal quality may limit these "
                   f"findings", value=q, unit="%", terms=("beat:Q",))

    fl.add("beat_source", "",
           "beat positions came from the cardiologist's annotations"
           if calls.source == "annotation" else "beat positions were detected automatically (XQRS)")
    return fl.items
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_heart_agent.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Checkpoint.** No commit.

---

### Task 5: Chart data loader

**Files:**
- Create: `src/agentic/chart_data.py`, `tests/agentic/test_chart_data.py`
- Modify: `tests/agentic/conftest.py` (append the `chart_env` and `chart_sources` fixtures)

**Interfaces:**
- Consumes:
  - `mimic_iii.train.load_model(out_dir) -> (model, meta)`
  - `mimic_iii.features.read_feature_table(path)`
  - `mimic_iii.dataset.read_split(path)`, `dataset.SPLIT_NAMES` and `PatientSplit.subjects(name)`
  - `mimic_iii.ingestion.find_table(root, name)`, which raises `SchemaError` when the table is missing
  - `mimic_iii.config.itemid_to_variable("lab")`, `config.VARIABLES`, `config.load_task_spec()` and `config.LABEL_COLUMN`
- Produces:
  - `ChartSources(table, cohort, admission_reason: dict[int,str], medications: DataFrame[hadm_id, drug, startdate], last_labs: DataFrame[icustay_id, variable, value, flag], model, meta, split_of: dict[int,str], source)`
  - `load_chart_sources(out_dir, raw_dir, source="demo") -> ChartSources`
  - `read_admission_reasons(raw_dir, hadm_ids)`, `read_medications(raw_dir, cohort)` and `read_last_labs(raw_dir, cohort, spec=None, chunksize=1_000_000)`

- [ ] **Step 1: Append the chart fixtures to `tests/agentic/conftest.py`:**

```python
import pandas as pd  # noqa: E402

from agentic.chart_data import load_chart_sources  # noqa: E402
from mimic_iii import dataset, features, train  # noqa: E402

from ..mimic_iii.conftest import load_cohort  # noqa: E402
from ..mimic_iii.synthetic import build_tables, write_database  # noqa: E402


@pytest.fixture(scope="session")
def chart_env(tmp_path_factory):
    """The synthetic MIMIC-III database plus what the agents need on top of it:
    admission reasons, prescriptions around the index day, hospital lab flags,
    and a low potassium for stay 201. Then cohort, features, split and model,
    written the way scripts/mimic_iii/*.py write them."""
    tables = build_tables()
    adm = tables["ADMISSIONS"]
    adm["diagnosis"] = "CONGESTIVE HEART FAILURE"
    adm.loc[adm["subject_id"] == 2, "diagnosis"] = "CHEST PAIN"

    labs = tables["LABEVENTS"]
    labs.loc[(labs["subject_id"] == 1) & (labs["itemid"] == 50912) & (labs["valuenum"] == 2.0),
             "flag"] = "abnormal"
    extra = pd.DataFrame([{"row_id": len(labs) + 1, "subject_id": 2, "hadm_id": 21,
                           "itemid": 50971, "charttime": "2150-01-01 12:00:00", "value": "3.1",
                           "valuenum": 3.1, "valueuom": "mEq/L", "flag": "abnormal"}])
    labs = pd.concat([labs, extra], ignore_index=True)
    labs["hadm_id"] = labs["hadm_id"].astype("Int64")
    tables["LABEVENTS"] = labs

    # Stays 101 and 201 start 2150-01-01 08:00, so the index day is 2150-01-02.
    rx = [(1, 11, 101, "2150-01-01 00:00:00", "Amiodarone"),          # day before: in
          (1, 11, 101, "2150-01-02 00:00:00", "Metoprolol"),          # index day: out
          (1, 11, 101, "2150-01-03 00:00:00", "Warfarin"),            # after: out
          (2, 21, 201, "2149-12-31 00:00:00", "Potassium Chloride"),  # in
          (2, 21, 201, "2149-12-31 00:00:00", "Acetaminophen")]       # in
    tables["PRESCRIPTIONS"] = pd.DataFrame(
        [{"row_id": i + 1, "subject_id": s, "hadm_id": h, "icustay_id": icu, "startdate": d,
          "enddate": "2150-01-09 00:00:00", "drug": name}
         for i, (s, h, icu, d, name) in enumerate(rx)])

    raw = write_database(tmp_path_factory.mktemp("mimic") / "db", tables)
    result = load_cohort(raw)
    events, _ = features.extract_events(raw, result.cohort)
    table = features.build_feature_table(result.cohort, events)
    split = dataset.patient_split(table)
    out = tmp_path_factory.mktemp("mimic_out")
    result.cohort.to_csv(out / "cohort.csv.gz", index=False)
    table.to_csv(out / "features.csv.gz", index=False)
    dataset.write_split(out / "splits.json", split, table)
    parts = dataset.split_frame(table, split)
    model, sel = train.select_and_fit(parts["train"], parts["val"])
    train.save_model(out, model, {
        "task": {"label": "ADMISSIONS.hospital_expire_flag", "index_time": "ICUSTAYS.intime + 24 h"},
        "data_source": "demo", "is_demo": True, "banner": "PIPELINE VERIFICATION ONLY",
        "threshold": sel["threshold"], "selection": sel})
    return {"raw": raw, "out": out, "table": table}


@pytest.fixture(scope="session")
def chart_sources(chart_env):
    return load_chart_sources(chart_env["out"], chart_env["raw"])
```

(Move the two `pandas` and `from …` imports up beside the existing imports at the top of the file; they are shown here only to make this step self-contained.)

- [ ] **Step 2: Write the failing tests** `tests/agentic/test_chart_data.py`:

```python
"""What the chart agent reads, and what it must never read."""

from __future__ import annotations

import pandas as pd

from agentic import chart_data


def test_outcome_is_never_loaded(chart_sources):
    cols = set(chart_sources.table.columns) | set(chart_sources.cohort.columns)
    assert not cols & {"label", "hospital_expire_flag", "deathtime", "dischtime",
                       "discharge_location"}


def test_admission_reasons(chart_sources):
    assert chart_sources.admission_reason[11] == "CONGESTIVE HEART FAILURE"
    assert chart_sources.admission_reason[21] == "CHEST PAIN"


def test_medications_use_only_days_before_the_index_day(chart_sources):
    meds = chart_sources.medications
    assert sorted(meds.loc[meds["hadm_id"] == 11, "drug"]) == ["Amiodarone"]
    assert sorted(meds.loc[meds["hadm_id"] == 21, "drug"]) == ["Acetaminophen",
                                                               "Potassium Chloride"]


def test_missing_prescriptions_table_gives_no_medications(tmp_path, chart_sources):
    meds = chart_data.read_medications(tmp_path, chart_sources.cohort)
    assert meds.empty and list(meds.columns) == ["hadm_id", "drug", "startdate"]


def test_last_lab_matches_the_feature_table(chart_sources, chart_env):
    table = chart_env["table"].set_index("icustay_id")
    labs = chart_sources.last_labs
    assert len(labs) > 0
    for row in labs.itertuples(index=False):
        assert row.value == table.loc[row.icustay_id, f"{row.variable}_last"]


def test_last_lab_keeps_the_hospital_flag(chart_sources):
    labs = chart_sources.last_labs
    creat = labs[(labs["icustay_id"] == 101) & (labs["variable"] == "creatinine")].iloc[0]
    assert creat["value"] == 2.0 and creat["flag"] == "abnormal"
    assert labs[(labs["icustay_id"] == 101) & (labs["variable"] == "potassium")].empty  # wrong unit


def test_split_of_covers_every_cohort_subject(chart_sources):
    assert set(chart_sources.split_of) == set(chart_sources.cohort["subject_id"].astype(int))
    assert set(chart_sources.split_of.values()) <= {"train", "val", "test"}
```

- [ ] **Step 3: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_chart_data.py -q`
Expected: FAIL (`agentic.chart_data` not found).

- [ ] **Step 4: Implement** `src/agentic/chart_data.py`:

```python
"""Everything the chart agent reads for the cohort, loaded once.

The outcome is never in memory here: the label column is dropped as soon as
the feature table is read, the cohort file is read without it, and ADMISSIONS
is read for two columns only (hadm_id, diagnosis).

The lab window and unit rules repeat those of mimic_iii.features
(window_lab_events, drop_implausible, aggregate's tie-break) because that code
returns events without the hospital's ``flag`` column. A test checks that the
last value found here equals the feature table's ``<variable>_last`` for every
stay, so the two cannot drift apart unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mimic_iii import config, dataset, features, ingestion, train
from mimic_iii.errors import SchemaError

COHORT_COLUMNS = ("subject_id", "hadm_id", "icustay_id", "intime", "index_time", "age",
                  "age_over_89", "gender", "admission_type", "first_careunit")
LAB_COLUMNS = frozenset({"subject_id", "hadm_id", "itemid", "charttime", "valuenum",
                         "valueuom", "flag"})


@dataclass
class ChartSources:
    table: pd.DataFrame  # feature table without the label
    cohort: pd.DataFrame  # COHORT_COLUMNS
    admission_reason: dict[int, str]  # hadm_id -> ADMISSIONS.diagnosis
    medications: pd.DataFrame  # hadm_id, drug, startdate (days before the index day only)
    last_labs: pd.DataFrame  # icustay_id, variable, value, flag (last value in the window)
    model: object
    meta: dict
    split_of: dict[int, str]  # subject_id -> "train" | "val" | "test"
    source: str = "demo"


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _read(path: Path, wanted) -> pd.DataFrame:
    wanted = set(wanted)
    return _lower(pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in wanted,
                              low_memory=False))


def read_admission_reasons(raw_dir, hadm_ids) -> dict[int, str]:
    df = _read(ingestion.find_table(raw_dir, "ADMISSIONS"), {"hadm_id", "diagnosis"})
    if "diagnosis" not in df:
        return {}
    df = df[df["hadm_id"].isin(set(hadm_ids)) & df["diagnosis"].notna()]
    return {int(h): str(d).strip() for h, d in zip(df["hadm_id"], df["diagnosis"])}


def read_medications(raw_dir, cohort: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"hadm_id": pd.Series(dtype="int64"), "drug": pd.Series(dtype="object"),
                          "startdate": pd.Series(dtype="datetime64[ns]")})
    try:
        path = ingestion.find_table(raw_dir, "PRESCRIPTIONS")
    except SchemaError:
        return empty
    df = _read(path, {"hadm_id", "startdate", "drug"})
    df["startdate"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.merge(cohort[["hadm_id", "index_time"]], on="hadm_id", how="inner")
    # startdate has no time of day: keep only drugs started on an earlier DAY
    # than the index time, so nothing started after the 24-hour mark slips in.
    keep = (df["startdate"].notna() & df["drug"].notna()
            & (df["startdate"] < df["index_time"].dt.normalize()))
    return df.loc[keep, ["hadm_id", "drug", "startdate"]].reset_index(drop=True)


def read_last_labs(raw_dir, cohort: pd.DataFrame, spec: config.TaskSpec | None = None,
                   chunksize: int = 1_000_000) -> pd.DataFrame:
    spec = spec or config.load_task_spec()
    lookup = config.itemid_to_variable("lab")
    subjects = set(cohort["subject_id"].astype(int))
    columns = ["icustay_id", "variable", "value", "flag"]

    parts = []
    reader = pd.read_csv(ingestion.find_table(raw_dir, "LABEVENTS"),
                         usecols=lambda c: str(c).strip().lower() in LAB_COLUMNS,
                         chunksize=chunksize, low_memory=False)
    for chunk in reader:
        chunk = _lower(chunk)
        chunk = chunk[chunk["itemid"].isin(list(lookup)) & chunk["subject_id"].isin(subjects)
                      & chunk["valuenum"].notna()]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=columns)
    df = pd.concat(parts, ignore_index=True)
    if "flag" not in df:
        df["flag"] = None
    df["charttime"] = pd.to_datetime(df["charttime"])

    stays = cohort[["icustay_id", "subject_id", "hadm_id", "intime", "index_time"]].rename(
        columns={"hadm_id": "stay_hadm_id"})
    df = df.merge(stays, on="subject_id", how="inner")
    start = df["intime"] - pd.Timedelta(hours=spec.lab_lookback_hours)
    df = df[(df["charttime"] >= start) & (df["charttime"] < df["index_time"])]
    other = (df["hadm_id"].notna() & (df["hadm_id"] != df["stay_hadm_id"])).astype(bool)
    df = df[~other].copy()

    df["variable"] = df["itemid"].map(lambda i: lookup[int(i)].name)
    df["value"] = df["valuenum"].astype(float)
    unit = df["valueuom"].astype("string").str.strip().str.lower()
    keep = pd.Series(True, index=df.index)
    for var in config.VARIABLES:
        if var.source != "lab":
            continue
        rows = df["variable"] == var.name
        bad_unit = (rows & unit.notna() & ~unit.isin(var.accepted_units)).fillna(False)
        lo, hi = var.valid_range
        implausible = rows & ((df["value"] < lo) | (df["value"] > hi))
        keep &= ~bad_unit.astype(bool) & ~implausible
    df = df[keep]

    df = df.sort_values(["icustay_id", "variable", "charttime", "value"], kind="mergesort")
    last = df.groupby(["icustay_id", "variable"], as_index=False).tail(1)
    return last[columns].reset_index(drop=True)


def load_chart_sources(out_dir, raw_dir, source: str = "demo") -> ChartSources:
    out_dir, raw_dir = Path(out_dir).expanduser(), Path(raw_dir).expanduser()
    model, meta = train.load_model(out_dir)
    table = features.read_feature_table(out_dir / "features.csv.gz").drop(
        columns=[config.LABEL_COLUMN])
    cohort = pd.read_csv(out_dir / "cohort.csv.gz", usecols=list(COHORT_COLUMNS),
                         parse_dates=["intime", "index_time"])
    split_of: dict[int, str] = {}
    split_path = out_dir / "splits.json"
    if split_path.exists():
        split = dataset.read_split(split_path)
        for name in dataset.SPLIT_NAMES:
            for subject in split.subjects(name):
                split_of[int(subject)] = name
    return ChartSources(
        table=table, cohort=cohort,
        admission_reason=read_admission_reasons(raw_dir, set(cohort["hadm_id"])),
        medications=read_medications(raw_dir, cohort),
        last_labs=read_last_labs(raw_dir, cohort),
        model=model, meta=meta, split_of=split_of, source=source)
```

- [ ] **Step 5: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_chart_data.py -q`
Expected: `7 passed`. If `test_last_lab_matches_the_feature_table` fails, the window or unit rule here differs from `mimic_iii.features`. Fix it here, never in `mimic_iii`.

- [ ] **Step 6: Checkpoint.** No commit.

---

### Task 6: Chart agent

**Files:**
- Create: `src/agentic/chart_agent.py`, `tests/agentic/test_chart_agent.py`

**Interfaces:**
- Consumes:
  - `ChartSources` (Task 5), `Knowledge` (Task 2), `FindingList` (Task 1)
  - `mimic_iii.features.model_input_columns()`
  - `mimic_iii.config.VARIABLES` and `VARIABLE_NAMES`
- Produces:
  - `analyze(stay: int, sources, knowledge) -> list[Finding]`, which raises `KeyError` for a stay outside the cohort.
  - The keys are: `lab_abnormal` → `"<var>:low|high|flagged"`; `vital_out_of_range` → `"<var>:low|high"`; `medication` → the drug group name.
  - Medication terms include `"drug:<pattern>"` for each matched pattern. The `risk_score` evidence has `threshold`, `flagged`, `reasons` and `split`.
  - `top_reasons(model, X, k=3) -> list[str]`.

- [ ] **Step 1: Write the failing tests** `tests/agentic/test_chart_agent.py`:

```python
"""The chart agent turns one ICU stay into findings, using only pre-index data."""

from __future__ import annotations

import pytest

from agentic import chart_agent


def run(stay, chart_sources, knowledge):
    findings = chart_agent.analyze(stay, chart_sources, knowledge)
    return findings, {f.kind: [g for g in findings if g.kind == f.kind] for f in findings}


def test_reference_stay(chart_sources, knowledge):
    findings, k = run(101, chart_sources, knowledge)
    assert [f.id for f in findings] == [f"C{i}" for i in range(1, len(findings) + 1)]
    assert k["admission_reason"][0].value == "CONGESTIVE HEART FAILURE"
    labs = {f.key: f for f in k["lab_abnormal"]}
    assert set(labs) == {"creatinine:high", "troponin_t:high"}
    assert labs["creatinine:high"].evidence["hospital_flag"] is True
    assert labs["creatinine:high"].value == 2.0
    assert "vital_out_of_range" not in k
    meds = {f.key: f for f in k["medication"]}
    assert set(meds) == {"antiarrhythmic"} and meds["antiarrhythmic"].evidence["drugs"] == ["Amiodarone"]
    assert "drug:amiodarone" in meds["antiarrhythmic"].terms
    risk = k["risk_score"][0]
    assert 0.0 <= risk.value <= 1.0 and "not validated" in risk.caveat
    assert set(risk.evidence) == {"threshold", "flagged", "reasons", "split"}


def test_stay_with_abnormal_vitals_and_low_potassium(chart_sources, knowledge):
    _, k = run(201, chart_sources, knowledge)
    labs = {f.key for f in k["lab_abnormal"]}
    assert {"lactate:high", "potassium:low"} <= labs
    vitals = {f.key: f.value for f in k["vital_out_of_range"]}
    assert vitals == {"heart_rate:high": 110.0, "resp_rate:high": 28.0}
    assert {f.key for f in k["medication"]} == {"potassium_supplement"}
    assert k["medication_count"][0].value == 2


def test_stay_with_no_measurements(chart_sources, knowledge):
    _, k = run(401, chart_sources, knowledge)
    assert k["demographics"][0].statement.startswith("Age >89")
    assert k["not_measured"][0].value == 18
    assert "lab_abnormal" not in k and "vital_out_of_range" not in k
    assert k["medication_count"][0].value == 0
    assert "risk_score" in k


def test_unknown_stay_raises(chart_sources, knowledge):
    with pytest.raises(KeyError):
        chart_agent.analyze(999999, chart_sources, knowledge)


def test_no_statement_mentions_the_outcome(chart_sources, knowledge):
    for stay in (101, 201, 401):
        for f in chart_agent.analyze(stay, chart_sources, knowledge):
            text = f"{f.statement} {f.caveat} {f.evidence}".lower()
            assert not any(w in text for w in ("label", "expire", "died", "survived",
                                               "discharge"))
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_chart_agent.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/agentic/chart_agent.py`:

```python
"""Chart agent: one ICU stay's first 24 hours, turned into findings.

Reads only what is known before the index time (ICU intime + 24 h): the
feature-table summaries the trained model also sees, the last lab values in
the same window, the reason for admission (recorded on arrival) and drugs
started on an earlier day. Never the label or any post-index column --
chart_data drops those before this module sees anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mimic_iii import config, features

from .chart_data import ChartSources
from .findings import Finding, FindingList
from .knowledge import Knowledge

UNITS = {v.name: v.unit for v in config.VARIABLES}
VITALS = tuple(v.name for v in config.VARIABLES if v.source == "chart")
LABELS = {
    "heart_rate": "heart rate", "sbp": "systolic blood pressure",
    "dbp": "diastolic blood pressure", "mbp": "mean arterial pressure",
    "resp_rate": "respiratory rate", "spo2": "oxygen saturation (SpO2)", "temp_c": "temperature",
    "troponin_t": "troponin T", "troponin_i": "troponin I", "ck_mb": "CK-MB",
    "creatinine": "creatinine", "potassium": "potassium", "sodium": "sodium", "bnp": "BNP",
    "lactate": "lactate", "glucose": "glucose", "hemoglobin": "hemoglobin",
    "wbc": "white blood cell count",
}
GROUP_LABELS = {"antiarrhythmic": "Antiarrhythmic", "beta_blocker": "Beta-blocker",
                "anticoagulant": "Anticoagulant", "diuretic": "Diuretic",
                "potassium_supplement": "Potassium supplement", "digoxin": "Digoxin",
                "vasopressor_inotrope": "Vasopressor/inotrope"}
MED_CAVEAT = ("dates only: a drug started on the day of the 24-hour mark is left out, "
              "because its time is unknown")


def _num(x) -> str:
    return f"{float(x):g}"


def _text(x, default: str = "not recorded") -> str:
    return default if pd.isna(x) else str(x)


def top_reasons(model, X: pd.DataFrame, k: int = 3) -> list[str]:
    """The k inputs that pushed the score most (coef x transformed value), as in
    scripts/mimic_iii/show_predictions.py; '+' raises the score, '-' lowers it."""
    names = [str(n) for n in model.named_steps["pre"].get_feature_names_out()]
    row = model.named_steps["pre"].transform(X)[0] * model.named_steps["clf"].coef_.ravel()
    top = np.argsort(-np.abs(row))[:k]
    return [f"{'+' if row[i] > 0 else '-'}{names[i]}" for i in top]


def _reason_terms(reasons: list[str]) -> set[str]:
    return {v for r in reasons for v in config.VARIABLE_NAMES if r[1:].startswith(v + "_")}


def _demographics(fl: FindingList, row) -> None:
    age = ">89" if int(row["age_over_89"]) == 1 else f"{float(row['age']):.0f}"
    sex = {"M": "male", "F": "female"}.get(_text(row["gender"]), "sex not recorded")
    fl.add("demographics", "",
           f"Age {age}, {sex}; {_text(row['admission_type']).lower()} admission; first ICU "
           f"unit {_text(row['first_careunit'])}",
           evidence={"age": age, "gender": _text(row["gender"]),
                     "admission_type": _text(row["admission_type"]),
                     "first_careunit": _text(row["first_careunit"])})


def _labs(fl: FindingList, labs: pd.DataFrame, knowledge: Knowledge) -> None:
    order = {v: i for i, v in enumerate(config.VARIABLE_NAMES)}
    labs = labs.assign(_o=labs["variable"].map(order)).sort_values("_o")
    for lab in labs.itertuples(index=False):
        var, value = lab.variable, float(lab.value)
        lo, hi = knowledge.lab_ranges.get(var, (None, None))
        direction = ("low" if lo is not None and value < lo
                     else "high" if hi is not None and value > hi else None)
        hospital = str(lab.flag).strip().lower() == "abnormal"
        if direction is None and not hospital:
            continue
        why = []
        if direction:
            why.append(f"outside the review range {_num(lo)} to {_num(hi)}")
        if hospital:
            why.append("flagged abnormal by the hospital lab")
        fl.add("lab_abnormal", f"{var}:{direction or 'flagged'}",
               f"Last {LABELS[var]} before the 24-hour mark: {_num(value)} {UNITS[var]} "
               f"({direction or 'abnormal'}; {' and '.join(why)})",
               value=float(_num(value)), unit=UNITS[var],
               evidence={"range": [lo, hi], "hospital_flag": hospital,
                         "direction": direction or "flagged"},
               terms=(var,))


def _vitals(fl: FindingList, row, knowledge: Knowledge) -> None:
    for var in VITALS:
        if var not in knowledge.vital_ranges or pd.isna(row[f"{var}_min"]):
            continue
        lo, hi = knowledge.vital_ranges[var]
        vmin, vmax = round(float(row[f"{var}_min"]), 1), round(float(row[f"{var}_max"]), 1)
        if vmin < lo:
            fl.add("vital_out_of_range", f"{var}:low",
                   f"Lowest {LABELS[var]} before the 24-hour mark: {_num(vmin)} {UNITS[var]} "
                   f"(review range {_num(lo)} to {_num(hi)})",
                   value=vmin, unit=UNITS[var], evidence={"min": vmin, "range": [lo, hi]},
                   terms=(var,))
        if vmax > hi:
            fl.add("vital_out_of_range", f"{var}:high",
                   f"Highest {LABELS[var]} before the 24-hour mark: {_num(vmax)} {UNITS[var]} "
                   f"(review range {_num(lo)} to {_num(hi)})",
                   value=vmax, unit=UNITS[var], evidence={"max": vmax, "range": [lo, hi]},
                   terms=(var,))


def _not_measured(fl: FindingList, row) -> None:
    missing = [v for v in config.VARIABLE_NAMES if int(row[f"{v}_missing"]) == 1]
    if missing:
        fl.add("not_measured", "",
               "Not measured before the 24-hour mark: " + ", ".join(LABELS[v] for v in missing),
               value=len(missing), unit="variables", evidence={"variables": missing},
               terms=tuple(missing))


def _medications(fl: FindingList, meds: pd.DataFrame, knowledge: Knowledge) -> None:
    names = sorted({str(d).strip() for d in meds["drug"]}, key=str.lower)
    grouped: dict[str, list[str]] = {}
    for group, patterns in knowledge.drug_groups.items():
        hits = [n for n in names if any(p in n.lower() for p in patterns)]
        if hits:
            grouped[group] = hits
    for group, hits in grouped.items():
        patterns = [p for p in knowledge.drug_groups[group] if any(p in h.lower() for h in hits)]
        fl.add("medication", group,
               f"{GROUP_LABELS.get(group, group)} started before the 24-hour mark: "
               f"{', '.join(hits)}",
               value=len(hits), unit="drugs", evidence={"drugs": hits},
               terms=("medication", group, *(f"drug:{p}" for p in patterns)),
               caveat=MED_CAVEAT)
    in_groups = {n for hits in grouped.values() for n in hits}
    statement = (f"{len(names)} different drugs started before the 24-hour mark "
                 f"({len(names) - len(in_groups)} outside the cardiac-relevant groups)"
                 if names else "No prescriptions found that started before the 24-hour mark")
    fl.add("medication_count", "", statement, value=len(names), unit="drugs",
           evidence={"drugs": names}, terms=("medication",), caveat=MED_CAVEAT)


def _risk(fl: FindingList, rows: pd.DataFrame, sources: ChartSources) -> None:
    X = rows[features.model_input_columns()]
    risk = float(sources.model.predict_proba(X)[0, 1])
    threshold = float(sources.meta["threshold"])
    flagged = bool(risk >= threshold)
    reasons = top_reasons(sources.model, X)
    split = sources.split_of.get(int(rows.iloc[0]["subject_id"]), "unknown")
    n_train = sources.meta.get("selection", {}).get("n_train")
    if sources.meta.get("is_demo"):
        caveat = ("MIMIC-III Demo model" + (f" trained on {n_train} patients" if n_train else "")
                  + "; not validated")
    else:
        caveat = "not validated for clinical use"
    if split == "train":
        caveat += "; this patient was in the model's training split, so the score is optimistic"
    fl.add("risk_score", "",
           f"Demo risk model score {risk:.2f} for in-hospital death (threshold {threshold:.2f}): "
           f"{'above' if flagged else 'below'} the threshold. Biggest pushes: {', '.join(reasons)}",
           value=round(risk, 2),
           evidence={"threshold": round(threshold, 2), "flagged": flagged, "reasons": reasons,
                     "split": split},
           terms=("risk_score", *sorted(_reason_terms(reasons))), caveat=caveat)


def analyze(stay: int, sources: ChartSources, knowledge: Knowledge) -> list[Finding]:
    rows = sources.table[sources.table["icustay_id"] == int(stay)]
    if rows.empty:
        raise KeyError(f"ICU stay {stay} is not in the cohort feature table")
    row = rows.iloc[0]
    fl = FindingList("chart")
    _demographics(fl, row)
    reason = sources.admission_reason.get(int(row["hadm_id"]))
    if reason:
        fl.add("admission_reason", "", f"Reason for admission, recorded on arrival: {reason}",
               value=reason, terms=("admission_reason",))
    _labs(fl, sources.last_labs[sources.last_labs["icustay_id"] == int(stay)], knowledge)
    _vitals(fl, row, knowledge)
    _not_measured(fl, row)
    _medications(fl, sources.medications[sources.medications["hadm_id"] == int(row["hadm_id"])],
                 knowledge)
    _risk(fl, rows, sources)
    return fl.items
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_chart_agent.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Checkpoint.** No commit.

---

### Task 7: Template report, orchestrator, CLI (end of P1)

**Files:**
- Create: `src/agentic/template_report.py`, `src/agentic/orchestrator.py`, `scripts/agentic/report.py`
- Test: `tests/agentic/test_orchestrator.py`, `tests/agentic/test_cli.py`
- Modify: `tests/test_scripts.py` (add `"scripts/agentic/report.py"` to `SCRIPTS`)

**Interfaces:**
- Consumes: everything from Tasks 1–6, plus:
  - `mit_bih.filters.denoise(sig, fs, low, high, order, use_notch)`
  - `mit_bih.inference.EcgClassifier` (`.bundle`, `.predict_signal(signal, fs, r_peaks)`), `detect_r_peaks(signal, fs)` and `DEFAULT_BUNDLE_PATH`
  - `mimic_iii.config.output_dir(source)`, `source_dir(source)` and `SOURCES`
  - `common.paths.data_root()`
- Produces:
  - `template_report.LIMITATIONS`, `limitations(case) -> list[str]`, `render(case) -> str` and `template_report(case, note="LLM off") -> ReportResult`.
  - `orchestrator.CaseRequest(record=None, start_s=0.0, seconds=20.0, stay=None, detect_peaks=False)`.
  - `orchestrator.StageEvent(stage, name, payload)` with `.skipped`.
  - `orchestrator.Deps(knowledge, classifier=None, chart=None, record_dir=None, source="demo", write_report=template_report.template_report, unavailable={})`.
  - `orchestrator.run_case(req, deps) -> Iterator[StageEvent]`. Stage 9's payload is `{"case": Case, "report": ReportResult}`.
  - `orchestrator.default_deps(*, need_ecg, need_chart, source="demo", mimic_dir=None, mimic_raw_dir=None, mitdb_dir=None, bundle_path=None, knowledge_path=None) -> Deps`.
  - `scripts/agentic/report.py`: `build_parser()`, `describe(event) -> str` and `main(argv) -> int`.

- [ ] **Step 1: Write the failing tests** `tests/agentic/test_orchestrator.py`:

```python
"""One case through every stage, with a tiny ECG ensemble and synthetic data."""

from __future__ import annotations

import json

from agentic import orchestrator, template_report
from agentic.findings import Case, FindingList


def deps(knowledge, tiny_classifier=None, chart_sources=None, mitdb_dir=None):
    return orchestrator.Deps(knowledge=knowledge, classifier=tiny_classifier, chart=chart_sources,
                             record_dir=mitdb_dir)


def test_full_case_emits_nine_stages_in_order(knowledge, tiny_classifier, chart_sources, mitdb_dir):
    req = orchestrator.CaseRequest(record="100", start_s=2.0, seconds=20.0, stay=101)
    events = list(orchestrator.run_case(req, deps(knowledge, tiny_classifier, chart_sources,
                                                  mitdb_dir)))
    assert [e.stage for e in events] == list(range(1, 10))
    assert not any(e.skipped for e in events)
    beats = events[3].payload
    assert beats["n_classified"] + beats["n_dropped"] == 24
    shown = events[4].payload["display_only"]
    assert shown["of"] == beats["n_classified"] and len(shown["truth"]) == beats["n_classified"]
    case, report = events[8].payload["case"], events[8].payload["report"]
    assert case.banner.startswith("SIMULATED CASE: ECG from MIT-BIH record 100")
    assert report.kind == "template" and report.text.startswith(case.banner)
    assert {f.agent for f in case.findings} == {"heart", "chart"}
    assert 24.0 in case.context_numbers and 101.0 in case.context_numbers


def test_truth_never_reaches_the_llm_facts(knowledge, tiny_classifier, chart_sources, mitdb_dir):
    req = orchestrator.CaseRequest(record="100", start_s=2.0, seconds=20.0, stay=101)
    case = list(orchestrator.run_case(req, deps(knowledge, tiny_classifier, chart_sources,
                                                mitdb_dir)))[-1].payload["case"]
    facts = json.dumps(case.facts_for_llm())
    assert "truth" not in facts and "agree" not in facts and "display" not in facts


def test_ecg_only_skips_the_chart_stage(knowledge, tiny_classifier, mitdb_dir):
    events = list(orchestrator.run_case(orchestrator.CaseRequest(record="100", start_s=2.0),
                                        deps(knowledge, tiny_classifier, None, mitdb_dir)))
    assert events[6].skipped and events[6].payload["skipped"] == "no ICU stay chosen"
    assert "SIMULATED" not in events[8].payload["case"].banner


def test_chart_only_skips_the_ecg_stages(knowledge, chart_sources):
    events = list(orchestrator.run_case(orchestrator.CaseRequest(stay=201),
                                        deps(knowledge, None, chart_sources)))
    assert [e.skipped for e in events[:6]] == [True] * 6
    assert events[8].payload["case"].stay == 201


def test_missing_chart_sources_skip_with_the_reason(knowledge):
    d = deps(knowledge)
    d.unavailable["chart"] = "MIMIC outputs not available"
    events = list(orchestrator.run_case(orchestrator.CaseRequest(stay=201), d))
    assert events[6].payload["skipped"] == "MIMIC outputs not available"


def test_template_lists_findings_rules_and_limitations():
    fl = FindingList("heart")
    fl.add("class_share", "V", "20.0% V", caveat="model output")
    case = Case(banner="BANNER", record="208", stay=1, findings=fl.items)
    text = template_report.render(case)
    assert text.startswith("BANNER") and "[H1] 20.0% V  (model output)" in text
    assert "Rules matched\n  none" in text
    assert template_report.SIMULATED_LIMITATION in text
    assert template_report.SIMULATED_LIMITATION not in template_report.render(
        Case(banner="B", record="208", findings=fl.items))
```

`tests/agentic/test_cli.py`:

```python
"""scripts/agentic/report.py end to end on synthetic data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "agentic"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"agentic_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load("report")


def _dirs(tiny_bundle, mitdb_dir, chart_env):
    return ["--bundle", str(tiny_bundle), "--mitdb-dir", str(mitdb_dir),
            "--mimic-dir", str(chart_env["out"]), "--mimic-raw-dir", str(chart_env["raw"])]


def test_full_case_prints_stages_and_report(tiny_bundle, mitdb_dir, chart_env, capsys):
    code = report.main(["--ecg-record", "100", "--start", "2", "--stay", "101",
                        *_dirs(tiny_bundle, mitdb_dir, chart_env)])
    out = capsys.readouterr().out
    assert code == 0
    assert "① raw ecg: record 100 (held-out)" in out
    assert "display only" in out
    assert "SIMULATED CASE: ECG from MIT-BIH record 100" in out
    assert "Limitations" in out


def test_list_shows_records_and_stays(tiny_bundle, mitdb_dir, chart_env, capsys):
    assert report.main(["--list", *_dirs(tiny_bundle, mitdb_dir, chart_env)]) == 0
    out = capsys.readouterr().out
    assert "208  trained-on" in out and "101" in out and "CONGESTIVE HEART FAILURE" in out


def test_bad_window_exits_1(tiny_bundle, mitdb_dir, chart_env, capsys):
    code = report.main(["--ecg-record", "100", "--start", "30", "--seconds", "20",
                        *_dirs(tiny_bundle, mitdb_dir, chart_env)])
    assert code == 1 and "34.4 s long" in capsys.readouterr().err
```

Add `"scripts/agentic/report.py",` to `SCRIPTS` in `tests/test_scripts.py`, after `"scripts/multimodal/demo_fusion_synthetic.py",`.

- [ ] **Step 2: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_orchestrator.py tests/agentic/test_cli.py -q`
Expected: FAIL (modules and script missing).

- [ ] **Step 3: Implement** `src/agentic/template_report.py`:

```python
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
```

- [ ] **Step 4: Implement** `src/agentic/orchestrator.py`:

```python
"""Run one case through every stage, yielding one event per stage.

Stage numbers match the page in PlanDoc/agentic_with_fe.md §14:
1 raw ECG, 2 cleaned, 3 beats found, 4 beats cut out, 5 model calls,
6 heart agent, 7 chart agent, 8 rules, 9 report. The CLI prints the events;
the Part B web app will stream them. A stage that cannot run yields an event
whose payload is {"skipped": reason}, so a reader always sees all nine.

The cardiologist's labels travel only in stage 5's ``display_only`` and in
``Case.display_only``; no agent, rule or prompt reads them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mimic_iii.errors import MimicError
from mit_bih import filters
from mit_bih import inference as ecg_inference

from . import chart_agent, ecg_window, heart_agent, rules, template_report
from .chart_data import ChartSources, load_chart_sources
from .findings import Case, ReportResult, make_banner
from .knowledge import DEFAULT_PATH, Knowledge, load_knowledge

STAGES = {1: "raw_ecg", 2: "cleaned", 3: "beats_found", 4: "beats_cut", 5: "model_calls",
          6: "heart_agent", 7: "chart_agent", 8: "rules", 9: "report"}
TRUTH_TOLERANCE_S = 0.15
OBSERVATION_HOURS = 24.0


@dataclass(frozen=True)
class CaseRequest:
    record: str | None = None
    start_s: float = 0.0
    seconds: float = 20.0
    stay: int | None = None
    detect_peaks: bool = False


@dataclass(frozen=True)
class StageEvent:
    stage: int
    name: str
    payload: dict

    @property
    def skipped(self) -> bool:
        return "skipped" in self.payload


@dataclass
class Deps:
    knowledge: Knowledge
    classifier: object | None = None  # mit_bih.inference.EcgClassifier
    chart: ChartSources | None = None
    record_dir: Path | None = None
    source: str = "demo"
    write_report: Callable[[Case], ReportResult] = template_report.template_report
    unavailable: dict = field(default_factory=dict)  # "ecg"/"chart" -> why it is not loaded


def _event(stage: int, payload: dict) -> StageEvent:
    return StageEvent(stage, STAGES[stage], payload)


def match_truth(peaks: np.ndarray, win: ecg_window.EcgWindow) -> list[str | None]:
    """The cardiologist's label for each classified beat (display only)."""
    if win.r_peaks.size == 0:
        return [None] * len(peaks)
    tol = TRUTH_TOLERANCE_S * win.fs
    out = []
    for p in peaks:
        j = int(np.argmin(np.abs(win.r_peaks - p)))
        out.append(win.truth[j] if abs(int(win.r_peaks[j]) - int(p)) <= tol else None)
    return out


def _ecg_stages(req: CaseRequest, deps: Deps, findings: list, display: dict, context: list):
    win = ecg_window.load_window(req.record, req.start_s, req.seconds, deps.record_dir)
    context += [float(req.record), float(req.start_s), float(req.seconds)]
    yield _event(1, {"record": win.record, "lead": win.lead, "fs": win.fs,
                     "start_s": win.start_s, "signal": win.signal})
    b = deps.classifier.bundle
    yield _event(2, {"signal": filters.denoise(win.signal, fs=b.fs, low=b.bandpass_low_hz,
                                               high=b.bandpass_high_hz, order=b.bandpass_order,
                                               use_notch=b.notch)})
    if req.detect_peaks:
        peaks, source = ecg_inference.detect_r_peaks(win.signal, win.fs), "detected"
    else:
        peaks, source = win.r_peaks, "annotation"
    yield _event(3, {"r_peaks": peaks, "source": source})
    pred = deps.classifier.predict_signal(win.signal, win.fs, r_peaks=peaks)
    p = pred.prepared
    yield _event(4, {"n_input": p.n_input, "n_classified": p.n_beats,
                     "n_dropped": p.n_dropped_boundary + p.n_dropped_edge, "examples": p.X[:8]})
    truth = match_truth(pred.r_peaks, win)
    called = [pred.classes[i] for i in pred.predicted]
    pairs = [(c, t) for c, t in zip(called, truth) if t is not None]
    display.update({"truth": truth, "agree": sum(c == t for c, t in pairs), "of": len(pairs)})
    yield _event(5, {"classes": pred.classes, "called": called, "scores": pred.scores,
                     "r_peaks": pred.r_peaks, "display_only": dict(display)})
    heart = heart_agent.analyze(heart_agent.BeatCalls.from_prediction(pred, win.fs, source))
    findings += heart
    yield _event(6, {"findings": heart})


def run_case(req: CaseRequest, deps: Deps) -> Iterator[StageEvent]:
    if req.record is None and req.stay is None:
        raise ValueError("choose an ECG record, an ICU stay, or both")
    findings: list = []
    display: dict = {}
    context: list = [OBSERVATION_HOURS]

    if req.record is None or deps.classifier is None:
        reason = ("no ECG chosen" if req.record is None
                  else deps.unavailable.get("ecg", "ECG model not loaded"))
        for stage in range(1, 7):
            yield _event(stage, {"skipped": reason})
    else:
        yield from _ecg_stages(req, deps, findings, display, context)

    if req.stay is None or deps.chart is None:
        reason = ("no ICU stay chosen" if req.stay is None
                  else deps.unavailable.get("chart", "MIMIC outputs not loaded"))
        yield _event(7, {"skipped": reason})
    else:
        chart = chart_agent.analyze(req.stay, deps.chart, deps.knowledge)
        context.append(float(req.stay))
        findings += chart
        yield _event(7, {"findings": chart})

    matches = rules.evaluate(deps.knowledge.rules, findings)
    yield _event(8, {"matches": matches})
    case = Case(banner=make_banner(req.record, req.stay, deps.source), record=req.record,
                stay=req.stay, findings=findings, rules=matches,
                context_numbers=tuple(context), display_only=display)
    yield _event(9, {"case": case, "report": deps.write_report(case)})


def default_deps(*, need_ecg: bool, need_chart: bool, source: str = "demo", mimic_dir=None,
                 mimic_raw_dir=None, mitdb_dir=None, bundle_path=None,
                 knowledge_path=None) -> Deps:
    """Load whatever is available; record why anything is not."""
    from mimic_iii import config as mimic_config

    deps = Deps(knowledge=load_knowledge(knowledge_path or DEFAULT_PATH),
                record_dir=mitdb_dir, source=source)
    if need_ecg:
        try:
            deps.classifier = ecg_inference.EcgClassifier.from_bundle(
                bundle_path or ecg_inference.DEFAULT_BUNDLE_PATH)
        except (FileNotFoundError, ValueError) as exc:
            deps.unavailable["ecg"] = f"ECG model not available: {exc}"
    if need_chart:
        try:
            deps.chart = load_chart_sources(mimic_dir or mimic_config.output_dir(source),
                                            mimic_raw_dir or mimic_config.source_dir(source),
                                            source)
        except (FileNotFoundError, ValueError, KeyError, MimicError) as exc:
            deps.unavailable["chart"] = (f"MIMIC outputs not available ({exc}); run "
                                         f"scripts/mimic_iii/build_features.py, make_splits.py, "
                                         f"train.py")
    return deps
```

- [ ] **Step 5: Implement** `scripts/agentic/report.py` (P1 version; Task 11 replaces `build_parser` and `main`):

```python
#!/usr/bin/env python3
"""Run the agents on one case and print every stage and the report.

    python scripts/agentic/report.py --list
    python scripts/agentic/report.py --ecg-record 208 --seconds 20 --stay 200282
    python scripts/agentic/report.py --ecg-record 100                  # ECG only
    python scripts/agentic/report.py --stay 200282                     # chart only

The ECG comes from an MIT-BIH record (held-out records give the honest view);
the chart from one MIMIC-III ICU stay. With both, they are different people
and the report says SIMULATED CASE. A research demo, not medical advice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic import ecg_window, orchestrator  # noqa: E402
from common import paths  # noqa: E402
from mimic_iii import config as mimic_config  # noqa: E402

CIRCLED = "①②③④⑤⑥⑦⑧⑨"


def describe(event: orchestrator.StageEvent) -> str:
    n, p = event.stage, event.payload
    head = f"{CIRCLED[n - 1]} {event.name.replace('_', ' ')}"
    if event.skipped:
        return f"{head}: skipped ({p['skipped']})"
    if n == 1:
        r = p["record"]
        note = f"; {r.note}" if r.note else ""
        return (f"{head}: record {r.name} ({r.tag}{note}), lead {p['lead']}, "
                f"{p['signal'].size / p['fs']:.1f} s from {p['start_s']:g} s")
    if n == 2:
        return f"{head}: band-pass filtered ({p['signal'].size} samples)"
    if n == 3:
        return f"{head}: {len(p['r_peaks'])} beat positions ({p['source']})"
    if n == 4:
        return f"{head}: {p['n_classified']} beats cut out, {p['n_dropped']} dropped at the edges"
    if n == 5:
        d = p["display_only"]
        return (f"{head}: model and cardiologist agree on {d['agree']} of {d['of']} beats "
                f"(display only - never given to the AI)")
    if n in (6, 7):
        return f"{head}: {len(p['findings'])} findings"
    if n == 8:
        return f"{head}: {', '.join(m.id for m in p['matches']) or 'no rules'} matched"
    r = p["report"]
    kind = "AI-written, fact-checked" if r.kind == "llm" else "template"
    return f"{head}: {kind}" + (f" ({r.note})" if r.note else "")


def list_options(deps: orchestrator.Deps) -> None:
    print("ECG records (MIT-BIH; paced records 102, 104, 107, 217 are not offered):")
    for r in ecg_window.offered_records():
        print(f"  {r.name}  {r.tag:10s} {r.note}")
    if deps.chart is None:
        print(f"\nICU stays: {deps.unavailable.get('chart', 'not loaded')}")
        return
    print(f"\nICU stays ({deps.source}):")
    for row in deps.chart.cohort.sort_values("icustay_id").itertuples(index=False):
        age = ">89" if row.age_over_89 == 1 else f"{row.age:.0f}"
        reason = deps.chart.admission_reason.get(int(row.hadm_id), "")
        print(f"  {row.icustay_id}  age {age:>3s} {row.gender}  {reason}")


def save(case, report, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"record{case.record or 'none'}_stay{case.stay or 'none'}"
    (out_dir / f"{stem}.txt").write_text(report.text + "\n")
    payload = {"banner": case.banner, "findings": [f.to_dict() for f in case.all_findings()],
               "report": {"kind": report.kind, "model": report.model, "note": report.note,
                          "text": report.text}}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return out_dir / f"{stem}.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="list the ECG records and ICU stays, then exit")
    parser.add_argument("--ecg-record", default=None)
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the record")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--detect-peaks", action="store_true",
                        help="find beats with the QRS detector instead of the annotations")
    parser.add_argument("--stay", type=int, default=None, help="MIMIC-III ICU stay id")
    parser.add_argument("--source", choices=sorted(mimic_config.SOURCES), default="demo")
    parser.add_argument("--mimic-dir", type=Path, default=None,
                        help="folder with features.csv.gz, cohort.csv.gz and the model")
    parser.add_argument("--mimic-raw-dir", type=Path, default=None,
                        help="folder with the raw MIMIC-III CSV tables")
    parser.add_argument("--mitdb-dir", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None, help="ECG inference bundle")
    parser.add_argument("--save", action="store_true",
                        help="also write the report under <data root>/processed/agentic/reports/")
    return parser


def run(args, deps, write_report=None):
    """Print every stage; return (case, report), or None on a bad request."""
    if write_report is not None:
        deps.write_report = write_report
    req = orchestrator.CaseRequest(record=args.ecg_record, start_s=args.start,
                                   seconds=args.seconds, stay=args.stay,
                                   detect_peaks=args.detect_peaks)
    try:
        last = None
        for event in orchestrator.run_case(req, deps):
            print(describe(event))
            last = event
    except (ecg_window.WindowError, KeyError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return None
    return last.payload["case"], last.payload["report"]


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list and args.ecg_record is None and args.stay is None:
        parser.error("give --ecg-record, --stay, or both (or --list)")
    deps = orchestrator.default_deps(
        need_ecg=args.ecg_record is not None, need_chart=args.list or args.stay is not None,
        source=args.source, mimic_dir=args.mimic_dir, mimic_raw_dir=args.mimic_raw_dir,
        mitdb_dir=args.mitdb_dir, bundle_path=args.bundle)
    if args.list:
        list_options(deps)
        return 0
    result = run(args, deps)
    if result is None:
        return 1
    case, report = result
    print("\n" + report.text)
    if args.save:
        print(f"\nSaved {save(case, report, paths.data_root() / 'processed' / 'agentic' / 'reports')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic tests/test_scripts.py -q`
Expected: all pass (orchestrator 6, CLI 3, plus all earlier agentic tests and the `--help` tests).

- [ ] **Step 7: Run the CLI by hand on real data** (local, read-only, about 10 s):

Run: `venv/bin/python scripts/agentic/report.py --list | head -60`. Pick the first stay ID it prints, then run
`venv/bin/python scripts/agentic/report.py --ecg-record 208 --seconds 20 --stay <that id>`.
Expected:
- nine stage lines;
- a report that starts `SIMULATED CASE: ECG from MIT-BIH record 208`;
- record 208 tagged trained-on;
- a `class_share` finding for V.

Write down the real output for the IMPLEMENTATION.md entry in Task 13.

- [ ] **Step 8: Checkpoint (P1 done).** No commit.

---

### Task 8: Ollama client (localhost only)

**Files:**
- Create: `src/agentic/llm.py`, `tests/agentic/test_llm.py`

**Interfaces:**
- Produces:
  - `DEFAULT_MODEL = "llama3.1:8b"`, `COMPARISON_MODEL = "qwen2.5"`, `DEFAULT_URL`.
  - `check_local(url) -> str`, which raises `ValueError` for a non-local address.
  - `LlmError(RuntimeError)` and `LlmUnavailable(LlmError)`.
  - `LlmStatus(ok, model, message)`.
  - `OllamaClient(model=DEFAULT_MODEL, base_url=DEFAULT_URL, timeout=120.0, keep_alive="15m")`, with `.model`, `.status() -> LlmStatus` and `.chat_json(messages, schema) -> dict`.
  - Any object with `.model` and `.chat_json(...)` is an "LLM" for Tasks 10–12. `FakeLLM` (Task 10) is one.

- [ ] **Step 1: Write the failing tests** `tests/agentic/test_llm.py`:

```python
"""The Ollama client talks to localhost only and fails politely."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentic import llm


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(self.server.tags)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        self._send({"message": {"role": "assistant", "content": self.server.content}})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_ollama():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.tags = {"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:latest"}]}
    server.content = '{"summary": "ok"}'
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield server
    server.shutdown()
    server.server_close()


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434/"])
def test_local_addresses_are_accepted(url):
    assert llm.check_local(url).startswith("http://")


@pytest.mark.parametrize("url", ["https://api.openai.com", "http://192.168.1.5:11434",
                                 "http://example.com:11434", "ftp://localhost:11434"])
def test_other_addresses_are_refused_before_any_request(url):
    with pytest.raises(ValueError, match="only a local Ollama"):
        llm.OllamaClient(base_url=url)


def test_status_when_nothing_listens():
    status = llm.OllamaClient(base_url=f"http://127.0.0.1:{_closed_port()}", timeout=2).status()
    assert not status.ok and "ollama serve" in status.message


def test_status_when_model_missing(fake_ollama):
    status = llm.OllamaClient("mistral", base_url=fake_ollama.url).status()
    assert not status.ok and "ollama pull mistral" in status.message


def test_status_accepts_the_latest_tag(fake_ollama):
    assert llm.OllamaClient("qwen2.5", base_url=fake_ollama.url).status().ok


def test_chat_json_sends_deterministic_settings_and_parses(fake_ollama):
    client = llm.OllamaClient(base_url=fake_ollama.url)
    schema = {"type": "object"}
    out = client.chat_json([{"role": "user", "content": "hi"}], schema)
    assert out == {"summary": "ok"}
    sent = fake_ollama.requests[0]
    assert sent["model"] == "llama3.1:8b" and sent["stream"] is False
    assert sent["format"] == schema and sent["options"] == {"temperature": 0, "seed": 0}


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_bad_reply_raises_llm_error(fake_ollama, content):
    fake_ollama.content = content
    with pytest.raises(llm.LlmError):
        llm.OllamaClient(base_url=fake_ollama.url).chat_json([], {})


@pytest.mark.requires_ollama
def test_real_ollama_answers_json():
    client = llm.OllamaClient()
    if not client.status().ok:
        pytest.skip(client.status().message)
    out = client.chat_json([{"role": "user", "content": 'Reply with {"ok": true}'}],
                           {"type": "object", "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"]})
    assert out == {"ok": True}
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_llm.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/agentic/llm.py`:

```python
"""A small client for the local Ollama server -- localhost only.

PERMISSIONS.md allows patient-derived findings to go to an LLM only because
Ollama runs on this machine. So the client refuses any address that is not
localhost before sending a byte, and ignores proxy environment variables (a
proxy would carry the request off the machine). It never pulls a model.

``chat_json`` asks for JSON that matches a schema (Ollama's structured output),
at temperature 0 with a fixed seed, so the same facts give the same draft.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

DEFAULT_MODEL = "llama3.1:8b"
COMPARISON_MODEL = "qwen2.5"
DEFAULT_URL = "http://localhost:11434"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class LlmError(RuntimeError):
    """The LLM call failed; callers fall back to the template report."""


class LlmUnavailable(LlmError):
    """Nothing answered at the Ollama address."""


def check_local(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in LOCAL_HOSTS:
        raise ValueError(f"refusing LLM address {url!r}: only a local Ollama server "
                         f"(http://localhost) is allowed; see PERMISSIONS.md")
    return url.rstrip("/")


@dataclass(frozen=True)
class LlmStatus:
    ok: bool
    model: str
    message: str


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL,
                 timeout: float = 120.0, keep_alive: str = "15m") -> None:
        self.base_url = check_local(base_url)
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if data is None else "POST")
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:200]
            raise LlmError(f"Ollama answered HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LlmError(f"no answer from Ollama within {self.timeout:g} s") from exc
            raise LlmUnavailable(f"Ollama is not reachable at {self.base_url} ({exc.reason}); "
                                 f"start it with `ollama serve`") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LlmError(f"no answer from Ollama within {self.timeout:g} s") from exc
        except json.JSONDecodeError as exc:
            raise LlmError(f"Ollama sent something that is not JSON: {exc}") from exc
        except OSError as exc:
            raise LlmUnavailable(f"Ollama is not reachable at {self.base_url} ({exc}); "
                                 f"start it with `ollama serve`") from exc

    def status(self) -> LlmStatus:
        try:
            tags = self._request("/api/tags")
        except LlmError as exc:
            return LlmStatus(False, self.model, str(exc))
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return LlmStatus(True, self.model, "ready")
        return LlmStatus(False, self.model,
                         f"model {self.model} is not pulled. `ollama pull {self.model}` downloads "
                         f"several GB and needs the user's OK (PERMISSIONS.md)")

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False, "format": schema,
                "options": {"temperature": 0, "seed": 0}, "keep_alive": self.keep_alive}
        reply = self._request("/api/chat", body)
        content = (reply.get("message") or {}).get("content", "")
        try:
            out = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LlmError(f"the model's reply is not valid JSON: {content[:200]!r}") from exc
        if not isinstance(out, dict):
            raise LlmError(f"the model's reply is JSON but not an object: {content[:200]!r}")
        return out
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_llm.py -q`
Expected: `12 passed, 1 deselected`. The deselected one is the `requires_ollama` test.

- [ ] **Step 5: Checkpoint.** No commit.

---

### Task 9: The fact-checker

**Files:**
- Create: `src/agentic/verifier.py`, `tests/agentic/fakes.py`, `tests/agentic/test_verifier.py`

**Interfaces:**
- Consumes: `Finding`, `Case` (Task 1); `knowledge.load_knowledge()` (Task 2), used for the drug words.
- Produces:
  - `Problem(code, detail)`.
  - `check_report(draft, case) -> list[Problem]`.
  - `check_chat(draft, case, glossary: list[Finding]) -> list[Problem]`.
  - `check_text(text, cited, by_id, context, where) -> list[Problem]`.
  - Problem codes:
    - `schema`, `no_citation`, `unknown_id`
    - `unsupported_number`, `unsupported_term`
    - `wrong_term`, `diagnosis_language`, `treatment_language`, `dose`, `outcome_language`
- Test helpers produced in `tests/agentic/fakes.py`: `make_case() -> Case`, `GOOD_REPORT: dict`, and `FakeLLM` (completed in Task 10).

- [ ] **Step 1: Write `tests/agentic/fakes.py`** (FakeLLM is used from Task 10 on):

```python
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
```

- [ ] **Step 2: Write the failing tests** `tests/agentic/test_verifier.py`:

```python
"""The fact-checker accepts faithful drafts and names every unsupported claim."""

from __future__ import annotations

import pytest

from agentic import verifier
from agentic.findings import Finding

from .fakes import good_report, make_case


def codes(problems):
    return [p.code for p in problems]


def test_faithful_draft_passes():
    assert verifier.check_report(good_report(), make_case()) == []


def edit(point_text, cites=("C1",)):
    d = good_report()
    d["points"][1] = {"text": point_text, "cites": list(cites)}
    return d


@pytest.mark.parametrize("text, cites, code", [
    ("Potassium was low at 2.8 mEq/L.", ["C1"], "unsupported_number"),
    ("Potassium was low at 3.1 mEq/L.", ["H1"], "unsupported_number"),
    ("Potassium was low at 3.1 mEq/L.", ["H9"], "unknown_id"),
    ("Potassium was low.", [], "no_citation"),
    ("Several V waves were seen.", ["H1"], "wrong_term"),
    ("The patient is diagnosed with VT.", ["H2"], "diagnosis_language"),
    ("Potassium should be increased.", ["C1"], "treatment_language"),
    ("Give 40 mEq of potassium.", ["C1"], "dose"),
    ("The patient died later.", ["C1"], "outcome_language"),
    ("The risk score was 41%.", ["C3"], "unsupported_number"),
    ("Lactate was high.", ["C1"], "unsupported_term"),
    ("Metoprolol was given.", ["C2"], "unsupported_term"),
])
def test_each_problem_is_caught(text, cites, code):
    assert code in codes(verifier.check_report(edit(text, cites), make_case()))


def test_rounding_to_the_shown_precision_is_allowed():
    assert verifier.check_report(edit("About 18% of beats were V beats.", ["H1"]),
                                 make_case()) == []


def test_numbers_from_the_matched_findings_of_a_cited_rule_are_allowed():
    d = good_report()
    d["for_clinician_review"] = {"text": "Potassium of 3.1 mEq/L with V beats is worth checking.",
                                 "cites": ["R3"]}
    assert verifier.check_report(d, make_case()) == []


def test_context_numbers_are_allowed():
    d = good_report()
    d["summary"] = "In the first 24 hours of ICU stay 200001, potassium was 3.1 mEq/L."
    assert verifier.check_report(d, make_case()) == []


def test_units_with_a_slash_are_not_doses():
    assert verifier.check_report(edit("Potassium was 3.1 mEq/L.", ["C1"]), make_case()) == []


@pytest.mark.parametrize("draft", [
    {"summary": 3, "points": [], "for_clinician_review": {"text": "", "cites": []}},
    {"summary": "s", "points": [{"text": "t", "cites": "H1"}],
     "for_clinician_review": {"text": "", "cites": []}},
    {"summary": "s", "points": [{"text": "t", "cites": ["H1"]}]},
    ["not", "a", "dict"],
])
def test_schema_problems_are_reported(draft):
    assert "schema" in codes(verifier.check_report(draft, make_case()))


def test_empty_review_is_allowed():
    d = good_report()
    d["for_clinician_review"] = {"text": "", "cites": []}
    assert verifier.check_report(d, make_case()) == []


GLOSSARY = [Finding(id="G3", agent="glossary", kind="glossary", key="V beat",
                    statement="V beat: an early beat that starts in the ventricles.",
                    terms=("beat:V",))]


def test_chat_answer_checks():
    case = make_case()
    ok = {"answerable": True, "answer": "A V beat starts in the ventricles.", "cites": ["G3"]}
    assert verifier.check_chat(ok, case, GLOSSARY) == []
    assert verifier.check_chat({"answerable": False, "answer": "", "cites": []}, case, GLOSSARY) == []
    bad = {"answerable": True, "answer": "Potassium was 2.0.", "cites": ["C1"]}
    assert "unsupported_number" in codes(verifier.check_chat(bad, case, GLOSSARY))
    assert "schema" in codes(verifier.check_chat({"answer": "x"}, case, GLOSSARY))
```

- [ ] **Step 3: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_verifier.py -q`
Expected: FAIL (`agentic.verifier` not found).

- [ ] **Step 4: Implement** `src/agentic/verifier.py`:

```python
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
```

- [ ] **Step 5: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_verifier.py -q`
Expected: `23 passed`.

Two rows need care if they fail:
- `"Give 40 mEq of potassium."` must report `dose`.
- `"Lactate was high."` citing C1 must report `unsupported_term`.

Fix the pattern or the vocabulary. **Never loosen the test.**

- [ ] **Step 6: Checkpoint.** No commit.

---

### Task 10: Report agent (LLM draft, one retry, fallback)

**Files:**
- Create: `src/agentic/report_agent.py`, `tests/agentic/test_report_agent.py`

**Interfaces:**
- Consumes:
  - `verifier.check_report` (Task 9), `template_report.render` / `limitations` / `template_report` (Task 7)
  - `llm.LlmError` (Task 8)
  - `FakeLLM`, `make_case`, `good_report` (`tests/agentic/fakes.py`)
- Produces:
  - `REPORT_SCHEMA`, `SYSTEM_PROMPT`
  - `facts_message(case) -> str`
  - `render(draft, case, model) -> str`
  - `write_report(case, llm, attempts=2) -> ReportResult`
  - `make_writer(llm) -> Callable[[Case], ReportResult]`, for `Deps.write_report`

- [ ] **Step 1: Write the failing tests** `tests/agentic/test_report_agent.py`:

```python
"""The LLM writes; the fact-checker decides; the template is the fallback."""

from __future__ import annotations

from agentic import orchestrator, report_agent
from agentic.llm import LlmError, LlmUnavailable

from .fakes import FakeLLM, good_report, make_case

BAD = {"summary": "The patient was diagnosed with VT.", "points": [{"text": "x", "cites": []}],
       "for_clinician_review": {"text": "", "cites": []}}


def test_good_draft_is_shown():
    fake = FakeLLM([good_report()])
    r = report_agent.write_report(make_case(), fake)
    assert r.kind == "llm" and r.attempts == 1 and r.model == "fake-llm"
    assert r.text.startswith("SIMULATED CASE") and "fact-checked" in r.text
    assert "[H1, H2]" in r.text and "Limitations" in r.text


def test_one_retry_with_the_problems_listed():
    fake = FakeLLM([BAD, good_report()])
    r = report_agent.write_report(make_case(), fake)
    assert r.kind == "llm" and r.attempts == 2
    retry_prompt = fake.calls[1]["messages"][-1]["content"]
    assert "failed the fact-check" in retry_prompt and "diagnos" in retry_prompt


def test_two_failures_fall_back_to_the_template():
    r = report_agent.write_report(make_case(), FakeLLM([BAD, BAD]))
    assert r.kind == "template" and r.attempts == 2 and r.problems
    assert "failed the fact-check 2 times" in r.note and "[R3]" in r.text


def test_wrong_shape_twice_falls_back():
    shape = {"summary": 3, "points": "H1", "for_clinician_review": None}
    r = report_agent.write_report(make_case(), FakeLLM([shape, shape]))
    assert r.kind == "template" and {p.code for p in r.problems} == {"schema"}


def test_llm_unavailable_falls_back():
    r = report_agent.write_report(make_case(), FakeLLM([LlmUnavailable("start it")]))
    assert r.kind == "template" and r.note.startswith("LLM unavailable")


def test_invalid_json_falls_back():
    r = report_agent.write_report(make_case(), FakeLLM([LlmError("not valid JSON")]))
    assert r.kind == "template" and "not valid JSON" in r.note


def test_no_llm_means_template():
    r = report_agent.write_report(make_case(), None)
    assert r.kind == "template" and r.note == "LLM off"


def test_prompts_never_carry_truth_or_outcome(knowledge, tiny_classifier, chart_sources, mitdb_dir):
    fake = FakeLLM([BAD], cycle=True)
    deps = orchestrator.Deps(knowledge=knowledge, classifier=tiny_classifier, chart=chart_sources,
                             record_dir=mitdb_dir, write_report=report_agent.make_writer(fake))
    req = orchestrator.CaseRequest(record="100", start_s=2.0, seconds=20.0, stay=201)
    list(orchestrator.run_case(req, deps))
    text = fake.prompt_text().lower()
    assert fake.calls
    for word in ("truth", "agree", "display", "label", "hospital_expire_flag", "deathtime",
                 "expired", "died", "survived", "discharge"):
        assert word not in text, word
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_report_agent.py -q`
Expected: FAIL (`agentic.report_agent` not found).

- [ ] **Step 3: Implement** `src/agentic/report_agent.py`:

```python
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
from .findings import Case, ReportResult
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
    return json.dumps({"case": case.banner, "facts": case.facts_for_llm()}, indent=1,
                      ensure_ascii=False)


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
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_report_agent.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Checkpoint.** No commit.

---

### Task 11: Glossary, chat agent, and the CLI's `--llm`, `--ask` and `--chat`

**Files:**
- Create: `configs/glossary.toml`, `src/agentic/glossary.py`, `src/agentic/chat_agent.py`
- Test: `tests/agentic/test_chat_agent.py`
- Modify: `scripts/agentic/report.py` (replace `build_parser` and `main`); `tests/agentic/test_cli.py` (add `--llm off` and a chat test)

**Interfaces:**
- Consumes:
  - `verifier.check_chat` (Task 9), `Finding` (Task 1)
  - `llm.OllamaClient`, `LlmError`, `DEFAULT_MODEL` (Task 8)
  - `report_agent.make_writer` (Task 10)
- Produces:
  - `glossary.load_glossary(path=DEFAULT_PATH) -> list[Finding]`. Entries have `agent="glossary"` and IDs `G1…`. Also `GlossaryError`.
  - `chat_agent.ChatAnswer(text, cites, verified, refused, reason)`.
  - `chat_agent.ChatAgent(case, glossary, llm)` with `.ask(question, history=()) -> ChatAnswer`. Each history item is `(question, answer_text)`.
  - CLI flags: `--llm {off|MODEL}` (default `llama3.1:8b`), `--ask QUESTION` (repeatable) and `--chat` (interactive).

- [ ] **Step 1: Write `configs/glossary.toml`:**

```toml
# Glossary for the chat agent: short, plain definitions the chat may quote.
# DRAFT - needs professor review. Read by src/agentic/glossary.py, which
# refuses an entry without a source. `terms` lists the clinical words an entry
# may be cited for (see src/agentic/verifier.py).

status = "draft - needs professor review"

[[entry]]
id = "G1"
term = "N beat"
definition = "A normal beat, or a beat the AAMI scheme groups with normal ones, such as bundle branch block beats."
source = "ANSI/AAMI EC57 beat classes, as mapped in src/mit_bih/labels.py"
terms = ["beat:N"]

[[entry]]
id = "G2"
term = "S beat"
definition = "A supraventricular ectopic beat: an early beat that starts above the ventricles, for example in the atria."
source = "ANSI/AAMI EC57 beat classes, as mapped in src/mit_bih/labels.py"
terms = ["beat:S"]

[[entry]]
id = "G3"
term = "V beat"
definition = "A ventricular ectopic beat: an early beat that starts in the ventricles, often called a PVC."
source = "ANSI/AAMI EC57 beat classes, as mapped in src/mit_bih/labels.py"
terms = ["beat:V"]

[[entry]]
id = "G4"
term = "F beat"
definition = "A fusion beat: a normal beat and a ventricular beat happening at the same moment and merging."
source = "ANSI/AAMI EC57 beat classes, as mapped in src/mit_bih/labels.py"
terms = ["beat:F"]

[[entry]]
id = "G5"
term = "Q beat"
definition = "An unclassifiable beat: one the AAMI scheme cannot place in the other four classes."
source = "ANSI/AAMI EC57 beat classes, as mapped in src/mit_bih/labels.py"
terms = ["beat:Q"]

[[entry]]
id = "G6"
term = "R peak"
definition = "The tallest point of a heartbeat on the ECG. Beat positions are R-peak positions."
source = "Standard ECG terminology; PlanDoc/Project_Glossary.pdf"
terms = ["beat"]

[[entry]]
id = "G7"
term = "RR interval"
definition = "The time between two R peaks. Heart rate in beats per minute is 60 divided by the RR interval in seconds."
source = "Standard ECG terminology; PlanDoc/Project_Glossary.pdf"
terms = ["heart_rate"]

[[entry]]
id = "G8"
term = "Run"
definition = "Three or more beats of the same kind in a row, for example 3 V beats in a row."
source = "Project definition (src/agentic/heart_agent.py)"
terms = ["run"]

[[entry]]
id = "G9"
term = "Non-sustained VT"
definition = "Three or more V beats in a row, faster than 100 per minute, that stop by themselves within 30 seconds."
source = "2017 AHA/ACC/HRS ventricular arrhythmia guideline (NSVT definition); to confirm"
terms = ["run", "beat:V", "heart_rate"]

[[entry]]
id = "G10"
term = "Ectopic beat"
definition = "A beat that starts somewhere other than the heart's normal pacemaker. S beats and V beats are ectopic."
source = "Standard ECG terminology; PlanDoc/Project_Glossary.pdf"
terms = ["beat:S", "beat:V"]

[[entry]]
id = "G11"
term = "Tachycardia and bradycardia"
definition = "Tachycardia means a heart rate above 100 per minute; bradycardia means below 60 per minute, for adults at rest."
source = "Standard adult definitions; to confirm"
terms = ["heart_rate"]

[[entry]]
id = "G12"
term = "Potassium"
definition = "A blood salt the heart needs for a steady rhythm. Low or high potassium can make irregular beats more likely."
source = "MedlinePlus, potassium blood test; to confirm"
terms = ["potassium"]

[[entry]]
id = "G13"
term = "Lactate"
definition = "A blood marker that rises when tissues are short of oxygen or under strain. Above 2 mmol/L is usually called raised."
source = "Sepsis-3 (Singer et al., JAMA 2016); to confirm"
terms = ["lactate"]

[[entry]]
id = "G14"
term = "Troponin"
definition = "A protein released into the blood when heart muscle is injured."
source = "Fourth Universal Definition of Myocardial Infarction (2018); to confirm"
terms = ["troponin_t", "troponin_i"]

[[entry]]
id = "G15"
term = "Mean arterial pressure (MAP)"
definition = "The average pressure in the arteries over one heartbeat. A common minimum target in very ill patients is 65 mmHg."
source = "Surviving Sepsis Campaign MAP target; to confirm"
terms = ["mbp"]

[[entry]]
id = "G16"
term = "SpO2"
definition = "Oxygen saturation: the share of hemoglobin carrying oxygen, measured with a finger clip."
source = "MedlinePlus, pulse oximetry; to confirm"
terms = ["spo2"]

[[entry]]
id = "G17"
term = "The 24-hour mark"
definition = "The index time: 24 hours after ICU admission. The chart agent uses only data recorded before it."
source = "This project (PlanDoc/agentic_with_fe.md)"
terms = []

[[entry]]
id = "G18"
term = "Risk score"
definition = "The Demo model's score, from the first 24 hours, for in-hospital death. It was trained on a few dozen Demo patients and is not validated."
source = "This project (src/mimic_iii, PlanDoc/agentic_with_fe.md)"
terms = ["risk_score"]

[[entry]]
id = "G19"
term = "Uncalibrated score"
definition = "The ECG model's scores rank beats from more to less likely to be a class, but they are not probabilities."
source = "This project (Phase 1f calibration diagnosis)"
terms = ["beat"]

[[entry]]
id = "G20"
term = "Held-out and trained-on records"
definition = "A held-out ECG record was never used to train the model, so results on it are an honest test. A trained-on record was used in training, so results on it look better than they are."
source = "This project (DS1/DS2 split)"
terms = []

[[entry]]
id = "G21"
term = "Simulated case"
definition = "The ECG and the chart come from two different people, from MIT-BIH and MIMIC-III. The pairing is for demonstration only."
source = "This project (PlanDoc/agentic_with_fe.md)"
terms = []

[[entry]]
id = "G22"
term = "Fact-check"
definition = "Every AI-written sentence is checked against the findings it cites; anything not backed by them is rejected."
source = "This project (src/agentic/verifier.py)"
terms = []
```

- [ ] **Step 2: Write the failing tests** `tests/agentic/test_chat_agent.py`:

```python
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
```

- [ ] **Step 3: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_chat_agent.py -q`
Expected: FAIL (modules not found).

- [ ] **Step 4: Implement** `src/agentic/glossary.py`:

```python
"""Glossary entries the chat may cite (G1, G2, ...), loaded and checked.

Each entry becomes a Finding with agent "glossary", so the fact-checker treats
a definition exactly like a case fact: numbers and clinical words in an answer
must be backed by the entries it cites.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from common import paths

from .findings import Finding

DEFAULT_PATH = paths.PROJECT_ROOT / "configs" / "glossary.toml"
GLOSSARY_ID = re.compile(r"^G\d+$")


class GlossaryError(ValueError):
    pass


def load_glossary(path: Path | str = DEFAULT_PATH) -> list[Finding]:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    out, seen = [], set()
    for i, e in enumerate(raw.get("entry", [])):
        gid = str(e.get("id", ""))
        where = f"entry #{i + 1} ({gid or '?'})"
        if not GLOSSARY_ID.match(gid):
            raise GlossaryError(f"{where}: id must look like G1, G2, ...")
        if gid in seen:
            raise GlossaryError(f"{where}: duplicate id")
        seen.add(gid)
        for name in ("term", "definition", "source"):
            if not str(e.get(name, "")).strip():
                raise GlossaryError(f"{where}: missing {name}")
        out.append(Finding(id=gid, agent="glossary", kind="glossary", key=e["term"],
                           statement=f"{e['term']}: {e['definition']}",
                           evidence={"source": e["source"]}, terms=tuple(e.get("terms", []))))
    return out
```

- [ ] **Step 5: Implement** `src/agentic/chat_agent.py`:

```python
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
from .findings import Case, Finding
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
        context = {"case": case.banner, "facts": case.facts_for_llm(),
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
```

- [ ] **Step 6: Run the chat tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_chat_agent.py -q`
Expected: `13 passed`.

- [ ] **Step 7: Wire the CLI.** In `scripts/agentic/report.py`:

1. Add imports beside the existing ones:

```python
from agentic import chat_agent, glossary, report_agent  # noqa: E402
from agentic.llm import DEFAULT_MODEL, OllamaClient  # noqa: E402
```

2. At the end of `build_parser()`, before `return parser`, add:

```python
    parser.add_argument("--llm", default=DEFAULT_MODEL,
                        help=f"local Ollama model for the report and chat, or 'off' "
                             f"(default {DEFAULT_MODEL})")
    parser.add_argument("--ask", action="append", default=[], metavar="QUESTION",
                        help="ask a question about the case after the report (repeatable)")
    parser.add_argument("--chat", action="store_true",
                        help="after the report, keep asking questions until an empty line")
```

3. Add these two functions above `main`:

```python
def make_llm(name: str):
    """An Ollama client if the model is ready, else None (the template is used)."""
    if name == "off":
        return None
    client = OllamaClient(name)
    status = client.status()
    print(f"LLM: {client.model} ready" if status.ok
          else f"LLM: {status.message} -> template report, chat off")
    return client if status.ok else None


def questions(args):
    yield from args.ask
    while args.chat:
        try:
            q = input("\nask> ")
        except EOFError:
            return
        if not q.strip() or q.strip().lower() in ("quit", "exit"):
            return
        yield q
```

4. Replace `main` with:

```python
def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list and args.ecg_record is None and args.stay is None:
        parser.error("give --ecg-record, --stay, or both (or --list)")
    deps = orchestrator.default_deps(
        need_ecg=args.ecg_record is not None, need_chart=args.list or args.stay is not None,
        source=args.source, mimic_dir=args.mimic_dir, mimic_raw_dir=args.mimic_raw_dir,
        mitdb_dir=args.mitdb_dir, bundle_path=args.bundle)
    if args.list:
        list_options(deps)
        return 0
    client = make_llm(args.llm)
    result = run(args, deps, report_agent.make_writer(client))
    if result is None:
        return 1
    case, report = result
    print("\n" + report.text)
    if report.kind == "template" and report.problems:
        print("\nWhy the AI draft was rejected:")
        print("\n".join(f"  - {p.detail}" for p in report.problems))
    if args.save:
        print(f"\nSaved {save(case, report, paths.data_root() / 'processed' / 'agentic' / 'reports')}")
    if args.ask or args.chat:
        agent = chat_agent.ChatAgent(case, glossary.load_glossary(), client)
        history = []
        for q in questions(args):
            a = agent.ask(q, history)
            cites = f" [{', '.join(a.cites)}]" if a.cites else ""
            checked = " (fact-checked)" if a.verified and not a.refused else ""
            print(f"\nQ: {q}\nA: {a.text}{cites}{checked}")
            history.append((q, a.text))
    return 0
```

5. In `tests/agentic/test_cli.py`, change `_dirs` to end with `"--llm", "off"` (so no test contacts Ollama), and add:

```python
def test_ask_without_llm_says_how_to_start_it(tiny_bundle, mitdb_dir, chart_env, capsys):
    code = report.main(["--stay", "101", "--ask", "What is a V beat?",
                        *_dirs(tiny_bundle, mitdb_dir, chart_env)])
    out = capsys.readouterr().out
    assert code == 0 and "Q: What is a V beat?" in out and "ollama serve" in out


def test_treatment_question_is_refused_even_without_llm(tiny_bundle, mitdb_dir, chart_env, capsys):
    report.main(["--stay", "101", "--ask", "What dose should we give?",
                 *_dirs(tiny_bundle, mitdb_dir, chart_env)])
    assert "can't suggest treatments" in capsys.readouterr().out
```

- [ ] **Step 8: Run all agent tests and the `--help` tests**

Run: `venv/bin/python -m pytest tests/agentic tests/test_scripts.py -q`
Expected: all pass.

- [ ] **Step 9: One real LLM run by hand.** This is local and allowed by PERMISSIONS (2026-10-15 amendment). It takes about 30–60 s.

Run: `venv/bin/python scripts/agentic/report.py --ecg-record 208 --seconds 20 --stay <stay from Task 7> --ask "What is a V beat?" --ask "Why was the risk score what it was?"`

Expected:
- `LLM: llama3.1:8b ready`.
- A report marked either "AI-written … fact-checked" or "template" with the rejection reasons listed.
- Two answers, each fact-checked or refused.

Write down which one happened for IMPLEMENTATION.md. Do not tune prompts to force a pass; the bake-off measures that.

- [ ] **Step 10: Checkpoint.** No commit.

---

### Task 12: LLM bake-off

**Files:**
- Create: `src/agentic/bakeoff.py`, `scripts/agentic/compare_llms.py`, `tests/agentic/test_bakeoff.py`
- Modify: `tests/test_scripts.py` (add `"scripts/agentic/compare_llms.py"`)

**Interfaces:**
- Consumes:
  - `report_agent.write_report` (Task 10)
  - `orchestrator.run_case`, `CaseRequest`, `default_deps` (Task 7)
  - `llm.OllamaClient`, `DEFAULT_MODEL`, `COMPARISON_MODEL` (Task 8)
- Produces:
  - `Trial(model, case_label, kind, attempts, seconds, problem_codes, coverage)`
  - `coverage(draft, case) -> float|None`
  - `run_trials(cases: list[tuple[str, Case]], clients) -> list[Trial]`
  - `summarize(trials) -> list[dict]`
  - `pick_winner(rows) -> str`
  - `render_markdown(rows, winner, n_cases, seed, run_date) -> str`

- [ ] **Step 1: Write the failing tests** `tests/agentic/test_bakeoff.py`:

```python
"""The bake-off scores models on the fact-check, rule coverage and time."""

from __future__ import annotations

from agentic import bakeoff
from agentic.llm import LlmUnavailable

from .fakes import FakeLLM, good_report, make_case

BAD = {"summary": "x", "points": [{"text": "V waves", "cites": ["H1"]}],
       "for_clinician_review": {"text": "", "cites": []}}


def test_scores_and_winner():
    cases = [("case 1", make_case()), ("case 2", make_case())]
    good = FakeLLM([good_report()], model="good", cycle=True)
    flaky = FakeLLM([BAD, good_report()], model="flaky", cycle=True)
    bad = FakeLLM([BAD], model="bad", cycle=True)
    rows = bakeoff.summarize(bakeoff.run_trials(cases, [good, flaky, bad]))
    by = {r["model"]: r for r in rows}
    assert by["good"]["pass_first_try"] == 1.0 and by["good"]["coverage"] == 1.0
    assert by["flaky"]["pass_first_try"] == 0.0 and by["flaky"]["pass_after_retry"] == 1.0
    assert by["bad"]["pass_after_retry"] == 0.0 and by["bad"]["problems"]["wrong_term"] == 2
    assert by["bad"]["coverage"] == 0.0
    assert bakeoff.pick_winner(rows) in {"good", "flaky"}


def test_ties_break_on_coverage_then_speed():
    rows = [{"model": "a", "pass_after_retry": 1.0, "coverage": 0.5, "median_seconds": 1.0},
            {"model": "b", "pass_after_retry": 1.0, "coverage": 1.0, "median_seconds": 9.0},
            {"model": "c", "pass_after_retry": 1.0, "coverage": 1.0, "median_seconds": 2.0}]
    assert bakeoff.pick_winner(rows) == "c"


def test_unreachable_model_is_scored_as_an_llm_error():
    rows = bakeoff.summarize(bakeoff.run_trials(
        [("case 1", make_case())], [FakeLLM([LlmUnavailable("down")], model="off")]))
    assert rows[0]["pass_after_retry"] == 0.0 and rows[0]["problems"] == {"llm_error": 1}


def test_markdown_holds_aggregates_only():
    case = make_case()
    rows = bakeoff.summarize(bakeoff.run_trials(
        [("case 1", case)], [FakeLLM([good_report()], model="good", cycle=True)]))
    text = bakeoff.render_markdown(rows, "good", n_cases=1, seed=7, run_date="2026-10-15")
    assert "| good |" in text and "Winner: **good**" in text
    assert str(case.stay) not in text and "3.1" not in text and "SIMULATED" not in text
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `venv/bin/python -m pytest tests/agentic/test_bakeoff.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/agentic/bakeoff.py`:

```python
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
```

- [ ] **Step 4: Implement** `scripts/agentic/compare_llms.py`:

```python
#!/usr/bin/env python3
"""Compare local LLMs as report writers on the same simulated cases.

    python scripts/agentic/compare_llms.py                    # 20 cases, llama3.1:8b vs qwen2.5
    python scripts/agentic/compare_llms.py --cases 4          # quick check

About 20-40 minutes for 20 cases x 2 models on an Apple M4. Writes:
  PlanDoc/agentic_results/llm_bakeoff.md     aggregate scores only (no patient data)
  <data root>/processed/agentic/bakeoff/     every draft, for inspection

Cases pair held-out MIT-BIH windows (60-80 s) with MIMIC-III Demo ICU stays
chosen by a fixed seed. They are different people: simulated cases.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic import bakeoff, orchestrator  # noqa: E402
from agentic.llm import COMPARISON_MODEL, DEFAULT_MODEL, OllamaClient  # noqa: E402
from common import paths  # noqa: E402

#: Held-out (DS2) records; 202 is left out because it is the same person as 201.
RECORDS = ("100", "103", "105", "111", "113", "117", "121", "123", "200", "210")
START_S, SECONDS = 60.0, 20.0
SEED = 20261015


def build_cases(deps, n: int) -> list[tuple[str, object]]:
    stays = sorted(int(s) for s in deps.chart.cohort["icustay_id"])
    chosen = np.random.default_rng(SEED).choice(stays, size=min(n, len(stays)), replace=False)
    cases = []
    for i, stay in enumerate(chosen):
        req = orchestrator.CaseRequest(record=RECORDS[(i // 2) % len(RECORDS)], start_s=START_S,
                                       seconds=SECONDS, stay=int(stay))
        case = list(orchestrator.run_case(req, deps))[-1].payload["case"]
        cases.append((f"case {i + 1}", case))
    return cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL, COMPARISON_MODEL])
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--out", type=Path,
                        default=paths.PROJECT_ROOT / "PlanDoc" / "agentic_results" / "llm_bakeoff.md")
    args = parser.parse_args(argv)

    clients = []
    for name in args.models:
        client = OllamaClient(name)
        status = client.status()
        print(f"{name}: {status.message}")
        if status.ok:
            clients.append(client)
    if not clients:
        print("No model is ready; nothing to compare.", file=sys.stderr)
        return 1

    deps = orchestrator.default_deps(need_ecg=True, need_chart=True)
    if deps.unavailable:
        print(f"Cannot build cases: {deps.unavailable}", file=sys.stderr)
        return 1
    cases = build_cases(deps, args.cases)
    print(f"{len(cases)} cases built; writing reports...")

    trials = bakeoff.run_trials(cases, clients)
    rows = bakeoff.summarize(trials)
    winner = bakeoff.pick_winner(rows)
    today = dt.date.today().isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(bakeoff.render_markdown(rows, winner, len(cases), SEED, today))
    raw_dir = paths.data_root() / "processed" / "agentic" / "bakeoff"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"{today}.json"
    raw.write_text(json.dumps({"rows": rows, "trials": [asdict(t) for t in trials],
                               "cases": [{"label": label, "record": c.record, "stay": c.stay}
                                         for label, c in cases]}, indent=2) + "\n")
    print(f"Winner: {winner}\nWrote {args.out}\nWrote {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add `"scripts/agentic/compare_llms.py",` to `SCRIPTS` in `tests/test_scripts.py`.

- [ ] **Step 5: Run the tests to check they pass**

Run: `venv/bin/python -m pytest tests/agentic/test_bakeoff.py tests/test_scripts.py -q`
Expected: all pass.

- [ ] **Step 6: Do NOT run the full bake-off yet.** It takes 20–40 minutes, and PERMISSIONS.md says long-running commands need the user's OK. Ask the user first: "Run the 20-case bake-off now (about 20–40 min, local only)?" A 2-case smoke run (`--cases 2`, about 3 min) is also long enough to ask about.

- [ ] **Step 7: Checkpoint.** No commit.

---

### Task 13: Real-data checks, docs and the full suite

**Files:**
- Create: `tests/agentic/test_real_data.py`
- Modify: `README.md`, `COMMANDS.md`, `PlanDoc/IMPLEMENTATION.md`, `PlanDoc/agentic_with_fe.md` (status line)

**Interfaces:**
- Consumes: everything above.
- Produces: no code interfaces; verified real-data behaviour and documentation.

- [ ] **Step 1: Write the real-data tests** `tests/agentic/test_real_data.py`:

```python
"""On the real MIT-BIH records and the MIMIC-III Demo (skipped by default)."""

from __future__ import annotations

import pytest

from agentic import chart_agent, orchestrator, report_agent, verifier
from agentic.llm import OllamaClient

from .fakes import make_case


def _real_deps(need_ecg=True):
    from mimic_iii import config as mimic_config
    from mit_bih import config as ecg_config

    if not ecg_config.mitdb_dir().is_dir():
        pytest.skip("MIT-BIH not on disk")
    if not (mimic_config.output_dir("demo") / "model_meta.json").exists():
        pytest.skip("MIMIC-III Demo outputs not built")
    deps = orchestrator.default_deps(need_ecg=need_ecg, need_chart=True)
    if deps.unavailable:
        pytest.skip(str(deps.unavailable))
    return deps


@pytest.mark.requires_data
def test_real_case_runs_end_to_end_with_the_template():
    deps = _real_deps()
    stay = int(deps.chart.cohort["icustay_id"].sort_values().iloc[0])
    req = orchestrator.CaseRequest(record="208", seconds=20.0, stay=stay)
    events = list(orchestrator.run_case(req, deps))
    assert [e.stage for e in events] == list(range(1, 10))
    assert events[4].payload["display_only"]["of"] > 15
    case, report = events[8].payload["case"], events[8].payload["report"]
    assert report.kind == "template" and case.banner.startswith("SIMULATED CASE")
    assert any(f.kind == "class_share" for f in case.findings)


@pytest.mark.requires_data
def test_every_demo_stay_gives_chart_findings():
    deps = _real_deps(need_ecg=False)
    for stay in deps.chart.cohort["icustay_id"]:
        findings = chart_agent.analyze(int(stay), deps.chart, deps.knowledge)
        assert findings[0].kind == "demographics"
        assert any(f.kind == "risk_score" for f in findings)


@pytest.mark.requires_ollama
def test_real_llm_report_is_fact_checked_or_falls_back():
    client = OllamaClient()
    if not client.status().ok:
        pytest.skip(client.status().message)
    case = make_case()
    r = report_agent.write_report(case, client)
    assert r.kind in ("llm", "template")
    if r.kind == "llm":
        assert verifier.check_report(r.draft, case) == []
```

- [ ] **Step 2: Run them**

Run: `venv/bin/python -m pytest tests/agentic/test_real_data.py -m "requires_data or requires_ollama" -q`
Expected: `3 passed`. A skip is acceptable only with its printed reason (for example, Ollama not running). Report skips as skips, never as passes.

- [ ] **Step 3: Run the full default suite**

Run: `venv/bin/python -m pytest -q`
Expected: the 633 earlier tests plus every new agentic test pass. Record the exact count shown.

- [ ] **Step 4: Update the docs.**

**`README.md`:** add this section after the multimodal section.

````markdown
## Agents (Part A of the agentic pipeline)

Agents turn the trained models' outputs into findings with evidence, draft rules
connect them, and a local LLM (Ollama, on this machine only) words the report.
A fact-checker rejects any sentence the findings do not back; the plain template
report is shown instead. Design: `PlanDoc/agentic_with_fe.md`.

```bash
python scripts/agentic/report.py --list
python scripts/agentic/report.py --ecg-record 208 --seconds 20 --stay <icustay_id>
python scripts/agentic/report.py --ecg-record 208 --stay <icustay_id> --ask "What is a V beat?"
python scripts/agentic/report.py --ecg-record 208 --stay <icustay_id> --llm off   # template only
```

An MIT-BIH record and a MIMIC-III stay are different people: such a case is a
**simulated case** and says so. Rules and glossary are drafts awaiting review.
Research demo, not for patient care.
````

**`COMMANDS.md`:** add a section `## Agents (report and chat)` containing:
- the four commands above;
- `ollama serve` (if not running);
- `python scripts/agentic/compare_llms.py --cases 20` (20–40 minutes);
- `pytest tests/agentic -m requires_data` and `pytest tests/agentic -m requires_ollama`.

Each command gets a one-line explanation.

**`PlanDoc/IMPLEMENTATION.md`:** add a dated section `## 2026-10-15 — Agentic pipeline, Part A (agents)` that lists:
- what was built (one line per module);
- the test counts from Step 3;
- the real CLI output observed in Task 7 Step 7 and Task 11 Step 9, including whether the LLM draft passed or was rejected and why;
- the two spec adjustments made while planning: a lab is abnormal if flagged by the hospital lab **or** outside the review range, and runs are counted over consecutive classified beats;
- that the bake-off is not yet run, pending the user's OK.

**`PlanDoc/agentic_with_fe.md`:** change the status line to:
`**Status:** Part A (agents, P1–P2) built 2026-10-15; bake-off pending; Part B (frontend) not started.`

- [ ] **Step 5: Final checkpoint.**
- Run `git status --short` and list every new or changed file to the user.
- Ask whether to commit (PERMISSIONS.md) and whether to run the bake-off now.
- Report honestly: test counts, skips and their reasons, the real LLM result, and anything that did not work.

---

## Self-review (done while writing)

**1. Spec coverage**

| Spec section | Covered by |
|---|---|
| §3 honesty rules | Tasks 1, 3, 5, 6, 7, 10 tests |
| §4 data | Task 5 |
| §6 components | Tasks 1–12 |
| §7 format | Task 1 |
| §8 heart agent | Task 4 |
| §9 chart agent | Tasks 5–6 |
| §10 rules | Task 2 |
| §11 report and fact-checker | Tasks 9–10 |
| §12 chat | Task 11 |
| §13 client and bake-off | Tasks 8, 12 |
| §15 errors (Part A rows) | Tasks 3, 7, 8, 10, 11 |
| §16 tests (Part A rows) | every task |
| §19 permissions | Global Constraints; Task 12 Step 6 |

§14 (frontend) and the web-server rows of §15–§16 belong to Part B and are not in this plan.

**2. Placeholder scan.** No "TBD" or "TODO". `<icustay_id>` and `<that id>` in commands are values the person running them reads from `--list` output.

**3. Type consistency.** These names are used the same way everywhere:
- `Finding`, `FindingList.add`, `Case.facts_for_llm` and `ReportResult(kind, text, model, seconds, attempts, problems, note, draft)`;
- `Deps.write_report` and `report_agent.make_writer`;
- `verifier.Problem.code`, `FakeLLM(replies, model, cycle)` and `ChatAnswer` fields.

**4. Review Focus.** Each of the five lines names its test and the task that owns it.
