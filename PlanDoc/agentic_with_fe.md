# Agentic pipeline with a frontend — plan

**Status:** Part A (agents) and Part B (frontend) built 2026-10-15; bake-off pending.
Build plans: `PlanDoc/plans/2026-10-15-agents-end-to-end.md`,
`PlanDoc/plans/2026-10-15-frontend.md`. The frontend lives in the top-level folder
`frontend/` (moved from `src/agentic/web/` at the user's request).

**One line:** pick a recorded heartbeat strip and an ICU patient's chart; watch the
signal go through every step of the pipeline on screen; get a plain-language report
written by a local LLM and checked by the app; ask follow-up questions in a chat that
only answers from that case.

### Roadmap — agents first, frontend second

The frontend only displays what the agents produce, so the agents are built and tested
first, from the command line, with no web page involved.

| Part | What | Sections | Build phase |
|---|---|---|---|
| Shared | purpose, decisions, honesty rules, data | §1–§5 | — |
| **A. Agents** (first) | finding format, **heart agent**, **chart agent**, **knowledge rules**, **report agent + fact-checker**, **chat agent**, LLM client and model bake-off | **§6–§13** | **P1** (agents without LLM) → **P2** (LLM, fact-checker, chat) |
| **B. Frontend** (after A works) | screen, stages, server, endpoints | §14 | **P3** |
| Shared | errors, tests, build order, scope, permissions, questions | §15–§20 | P4 = docs |

The agents in one picture:

```
 MIT-BIH window ─▶ ECG model ─▶ Heart agent ─┐
                                              ├─▶ findings (H…, C…) ─▶ Rules (R…) ─▶ Report agent ─▶ fact-checker ─▶ report
 MIMIC ICU stay ─▶ risk model ─▶ Chart agent ─┘                                         (Ollama LLM)      │
                                                                                                         └─▶ Chat agent (same findings + glossary)
```

At the end of Part A you can already run a full case in the terminal:
`python scripts/agentic/report.py --ecg-record 208 --seconds 20 --stay <icustay_id>`.

---

## 1. Purpose

### What it is for
A live, local demo for you and your professor that makes the project visible:

1. the **heartbeat model** (MIT-BIH, HybridBeatNet ensemble) classifies every beat;
2. the **chart model** (MIMIC-III) reads the first 24 h of an ICU stay;
3. **agents** turn both outputs into findings with evidence;
4. **rules** with cited sources connect findings ("frequent V beats + low potassium");
5. a **local LLM** (Ollama) writes the report in plain words, and a **fact-checker**
   rejects anything not backed by a finding;
6. a **chat** answers questions about the case on screen.

### What it is not
- Not a diagnosis tool and not a medical device. Every screen says so.
- Not fusion. The ECG and the chart come from **different people** (MIT-BIH 1975–79 vs
  MIMIC-III 2001–12). The case is labelled **SIMULATED CASE** everywhere. The trained
  fusion network stays **BLOCKED** until same-patient data exists; the page says that too.
- Not a classifier built from the LLM. The LLM cannot read a signal. Classification is
  done by our trained models; reasoning by rules; the LLM only words the result.

### What success looks like
- You pick record 208 (held-out) + an ICU stay, press **Run**, and see 9 stages fill in
  with real intermediate outputs within ~3 s (ECG stages) and ~40 s (report).
- Every sentence of the report and every chat answer is either fact-checked ✓ or replaced
  by the plain template / a refusal — never shown unchecked.
- Runs fully offline on this laptop; nothing leaves the machine.

---

## 2. Decisions already made

| Question | Decision | When |
|---|---|---|
| How does the ECG get in? | **Pick a recorded MIT-BIH ECG** (record + start + length). No file upload, no photo upload. | 2026-10-15 |
| What may the chat answer? | **This case's findings + a short glossary** of terms. Everything else is politely refused. | 2026-10-15 |
| How is the frontend built? | **Plain Python server + one web page.** Python standard library only, vanilla HTML/JS/CSS, no installs. | 2026-10-15 |
| Which LLM? | **`llama3.1:8b` as default, `qwen2.5` as comparison** — both already pulled. Final pick by the bake-off in §11. | 2026-10-15 |
| Is the MIMIC-III Demo enough? | **Yes for building and demonstrating; no for proving accuracy.** See §4. | 2026-10-15 |
| Where does the clinical side come from? | A dropdown of the 51 cohort ICU stays. | default |
| Local LLM allowed with patient data? | Yes, localhost only — `PERMISSIONS.md` amended 2026-10-15. | 2026-10-15 |

Why the LLM probe favoured llama3.1:8b (2026-10-15, one invented case, both ~22–25 s):
qwen2.5 wrote "V waves" / "S wave" (wrong terms) and added "metabolic derangement"
(not in the facts); llama3.1:8b stayed faithful but left out the V-beat runs and the
risk score. One case is not evidence — hence the bake-off.

---

## 3. Honesty rules (non-negotiable, enforced by tests)

1. **SIMULATED CASE banner** on the page, in every report, and in the CLI output:
   "ECG: MIT-BIH record X. Chart: MIMIC-III Demo ICU stay Y. Different people."
2. **The LLM never sees the answer.** Ground-truth beat labels (from `.atr`) and the
   patient's outcome (`hospital_expire_flag`, `deathtime`, discharge fields) are
   **display-only**. They are never put into findings, rules or prompts. A test spies on
   every prompt and fails if any of them appears.
3. **No peeking past 24 h.** The chart agent uses only data before the index time
   (ICU intime + 24 h), the same rule as the trained model. Medications use a stricter
   cut because `PRESCRIPTIONS.startdate` has no time of day: a drug counts only if it
   started on a **day before** the index day.
4. **Training-set records are labelled.** Each MIT-BIH record shows **held-out** (DS2,
   22 records) or **trained-on** (DS1, 22 records). Record 202 is marked
   "held-out, but same person as 201 (trained-on)". The 4 paced records
   (102, 104, 107, 217) are not offered — the model was never meant for them.
5. **The risk score carries its caveat:** "MIMIC-III Demo model, 37 training patients,
   not validated."
6. **Local only.** The server binds to `127.0.0.1` and refuses any other address. The
   page loads nothing from the internet (no CDN, no web fonts). The LLM client refuses
   any URL other than `http://localhost:11434` / `http://127.0.0.1:11434`.
7. **No row-level clinical data in the repository.** Saved reports and screenshots go
   under the data root (`~/physionet-data/processed/agentic/`); only aggregate numbers
   (e.g. bake-off scores) go in `PlanDoc/`.

---

## 4. What data is available (checked 2026-10-15)

| Item | Available? | Notes |
|---|---|---|
| MIT-BIH records | 48 on disk, 44 offered | 22 held-out (DS2), 22 trained-on (DS1), 4 paced excluded |
| Beat annotations (`.atr`) | yes | used as beat positions (as in training) and as display-only truth |
| ECG model bundle | yes | `D_all_four`, 20 networks, 360 Hz only, scores uncalibrated |
| MIMIC cohort | 51 ICU stays | adult, first stay, cardiac ICD-9, alive in ICU at 24 h |
| Vitals + labs (first 24 h) | yes | 7 vitals, 11 labs, mean/min/max/last/count |
| Hospital "abnormal" lab flag | yes | `LABEVENTS.flag` = `abnormal` on 10,808 rows of cohort admissions |
| Reason for admission | 51 of 51 | `ADMISSIONS.diagnosis`, free text, known at admission (e.g. "SEPSIS") |
| Medications before the index day | 47 of 51 | `PRESCRIPTIONS`, day-before rule (§3.3) |
| Earlier admissions ("past history") | **0 of 51** | the cohort keeps each patient's first stay; the Demo has none |
| Risk model | yes | logistic regression, threshold 0.70 chosen on validation |
| Ollama | 0.18.1 | `llama3.1:8b` (4.9 GB), `qwen2.5:latest` (4.7 GB); JSON-schema output supported |
| Hardware | Apple M4, 16 GB RAM | one 8B model at a time is comfortable |

**So "patient history" in this demo = reason for admission + medications already
started + the first 24 h of measurements.** True past history needs full MIMIC-III
(credentialed). Swapping the data source later does not change the agent code.

Diagnosis codes (`DIAGNOSES_ICD`) are **not** used as history: they are assigned at
discharge, so using them would leak the future. `NOTEEVENTS` is not used.

---

## 5. How it fits together

```
 Browser (127.0.0.1)                     Python server (stdlib)
 ───────────────────                     ───────────────────────────────────────────
 pick record/window/stay ── GET /api/run ──▶ orchestrator.run_case()  (generator)
                                              │
 ① raw ECG strip      ◀── stage event ──────  ├─ load record window (wfdb)
 ② cleaned signal     ◀── stage event ──────  ├─ filters.denoise (bundle settings)
 ③ beats found        ◀── stage event ──────  ├─ R peaks (annotations | XQRS)
 ④ beats cut out      ◀── stage event ──────  ├─ prepare_beats (300-sample windows)
 ⑤ model says         ◀── stage event ──────  ├─ EcgClassifier.score_beats (+ truth, display only)
 ⑥ heart agent        ◀── stage event ──────  ├─ heart_agent → findings H1..Hn
 ⑦ chart agent        ◀── stage event ──────  ├─ chart_agent → findings C1..Cn (+ risk)
 ⑧ rules              ◀── stage event ──────  ├─ rules → matches R1..Rn (with sources)
 ⑨ report             ◀── stage event ──────  └─ report_agent: LLM draft → fact-check
                                                   → verified report | template fallback
 chat box  ── POST /api/chat ──────────────▶ chat_agent(case findings + glossary)
                                               → LLM answer → fact-check → answer | refusal
                                      Ollama ◀─ http://localhost:11434 only
```

Stage events are sent with **server-sent events** (a plain HTTP response that stays open
and sends one JSON message per stage). The browser shows each stage as it arrives.

The orchestrator is the same code for the CLI and the web app; the web server only
streams what the orchestrator yields.

---

## 6. Components

New package `src/agentic/`. Each file has one job and can be tested alone.

| File | Job | Depends on |
|---|---|---|
| `findings.py` | `Finding` and `Case` data classes; IDs; JSON form | — |
| `ecg_window.py` | load a record window (signal, annotation peaks, display-only truth); offered-record list with held-out/trained-on tags | `mit_bih.config`, `mit_bih.splits`, wfdb |
| `heart_agent.py` | beat predictions → rhythm findings (§8) | `mit_bih.inference` outputs |
| `chart_agent.py` | one ICU stay → clinical findings (§9) | `mimic_iii` feature table, model, raw Demo CSVs |
| `rules.py` + `configs/agent_rules.toml` | load rules (refuse any without a source); evaluate against findings (§10) | `findings` |
| `glossary.py` + `configs/glossary.toml` | short, reviewed definitions (G1..Gn) for the chat | — |
| `llm.py` | Ollama client: localhost guard, JSON schema, temperature 0, timeout, availability check | stdlib `urllib` |
| `verifier.py` | the fact-checker (§11) | `findings`, `glossary` |
| `report_agent.py` | template report; LLM report with one retry; fallback | `llm`, `verifier` |
| `chat_agent.py` | grounded question answering; refusals | `llm`, `verifier`, `glossary` |
| `orchestrator.py` | runs stages 1–9 in order, yields one event per stage | all of the above |
| `frontend/server.py`, `frontend/app.py` | `ThreadingHTTPServer` on 127.0.0.1; routes; case store | `orchestrator`, `chat_agent` |
| `frontend/static/index.html`, `app.js`, `app.css` | the page | — |

Scripts:

| Script | Use |
|---|---|
| `scripts/agentic/report.py` | CLI: `--ecg-record 208 --start 0 --seconds 20 --stay <id> [--llm off\|llama3.1:8b\|qwen2.5]` → prints the report |
| `scripts/agentic/serve.py` | start the web app: `--port 8765` (default), `--open` to open the browser |
| `scripts/agentic/compare_llms.py` | the bake-off (§12) |

The existing `src/multimodal/inference.run_inference` stays as it is. The orchestrator
uses `EcgClassifier` and the MIMIC model directly because it needs the intermediate
outputs (filtered signal, beat windows) that `run_inference` does not return.

---

## 7. The finding format

Every agent speaks the same format, so rules, the report, the chat and the fact-checker
all work from one list.

```json
{
  "id": "H2",
  "agent": "heart",
  "kind": "v_run",
  "statement": "2 runs of 3 or more consecutive V beats",
  "value": 2,
  "unit": "runs",
  "evidence": {"longest_run_beats": 4, "run_rate_bpm": [118, 124], "beat_indices": [[11,13],[30,33]]},
  "confidence": "model output, uncalibrated",
  "caveat": "beats classified by the ECG model, not a cardiologist"
}
```

- `id` prefixes: `H` heart agent, `C` chart agent, `R` rule matches, `G` glossary.
- `statement` is written by code from the values, never by the LLM.
- Numbers keep the precision they are displayed at, so the fact-checker can match them.

---

## 8. Heart agent (stage ⑥)

Input: the per-beat predictions for the chosen window. Output findings:

| Kind | How it is computed |
|---|---|
| `beats_classified` | number classified, number dropped at the edges (from `PreparedBeats`) |
| `class_share` | % of beats per class N/S/V/F/Q |
| `heart_rate` | 60 / mean RR (seconds) from consecutive classified beats; min and max too |
| `v_run` | runs of ≥ 3 consecutive predicted V beats; rate inside each run. Beats are dropped only at the window edges, so the classified beats are consecutive. |
| `s_run` | same for S beats |
| `low_confidence_beats` | beats whose top score < 0.5 |
| `q_share_high` | Q (unclassifiable) share ≥ 5% → "signal quality may be limited" |
| `beat_source` | "annotation positions" or "detected (XQRS)" |

Not included: agreement with the true labels. That is shown on screen (stage ⑤) but is
**not** a finding, because the report must describe what the model saw, not the answer key.

---

## 9. Chart agent (stage ⑦)

Input: one ICU stay. Output findings:

| Kind | Source | Rule |
|---|---|---|
| `demographics` | cohort | age (or ">89"), sex, admission type, first care unit |
| `admission_reason` | `ADMISSIONS.diagnosis` | free text, known at admission |
| `lab_abnormal` | `LABEVENTS` in the feature window (intime − 6 h to index) | the **last** value of each of the 11 labs counts as abnormal if the hospital lab flagged it `abnormal` **or** it is outside the review range in `configs/agent_rules.toml`; the range gives the direction (low/high); value, unit and which test fired are shown |
| `vital_out_of_range` | feature table (min/max/last) | ranges from `configs/agent_rules.toml` (marked for review) |
| `not_measured` | `*_missing` flags | e.g. "troponin not measured in the first 24 h" |
| `medications` | `PRESCRIPTIONS` | drugs started on a day before the index day; de-duplicated generic names; cardiac-relevant ones (antiarrhythmics, beta-blockers, anticoagulants, diuretics, potassium) listed first |
| `risk_score` | trained model | probability, the 0.70 threshold, top 3 reasons (coef × value, as `show_predictions.py`), caveat (§3.5) |

Never read: `label`, every column in `POST_INDEX_COLUMNS`, `DIAGNOSES_ICD`, `NOTEEVENTS`.
`features.assert_no_leakage` is reused on everything the agent reads.

---

## 10. Knowledge rules (stage ⑧)

A TOML file. Each rule: an ID, conditions on findings (all must hold), what it says, a
source, and a status. **The loader refuses a rule with an empty source.** All rules ship
as `status = "draft — needs professor review"` and the page shows that badge.

Draft rule list (thresholds and wording to be reviewed; sources to be confirmed before
the rule is enabled):

| ID | When | Says | Source (to confirm) |
|---|---|---|---|
| R1 | V share ≥ 10% | frequent ventricular ectopy in this window | threshold to review |
| R2 | `v_run` ≥ 1 with rate > 100 | run(s) consistent with the definition of non-sustained VT (≥ 3 beats, > 100/min, < 30 s); needs clinician review | 2017 AHA/ACC/HRS ventricular arrhythmia guideline |
| R3 | (R1 or R2) and potassium abnormal-low | ventricular ectopy together with low potassium — a combination worth checking | to confirm |
| R4 | heart rate > 100 | tachycardia in this window | standard definition |
| R5 | heart rate < 60 | bradycardia in this window | standard definition |
| R6 | S share ≥ 10% or `s_run` ≥ 1 | frequent supraventricular ectopy | threshold to review |
| R7 | lactate > 2 mmol/L | raised lactate | Sepsis-3 (Singer et al., 2016) |
| R8 | troponin abnormal (lab flag or review range) | troponin above the lab's reference — myocardial injury marker | Fourth Universal Definition of MI (2018) |
| R9 | mean BP min < 65 mmHg | low mean arterial pressure at some point | Surviving Sepsis Campaign MAP target |
| R10 | antiarrhythmic in medications and (R1 or R2) | ectopy despite an antiarrhythmic already started | to confirm |
| R11 | `q_share_high` | many unclassifiable beats; ECG quality may limit the findings | project rule |
| R12 | `risk_score` ≥ threshold | the Demo model flags higher risk (not validated) | project rule |

Rules never conclude a diagnosis; they phrase "consistent with", "worth checking",
"flags".

---

## 11. Report agent and the fact-checker (stage ⑨)

### Template report (always built first)
Code fills a fixed layout from the findings and rule matches. It is always correct and
always available. It is shown when the LLM is off, missing, slow or rejected.

### LLM report
- Input to the LLM: the banner, the findings (ID + statement + value + unit), the rule
  matches (ID + says), and instructions: use only these facts; cite IDs in brackets for
  every point; no diagnosis; no treatment or doses; plain language.
- Output forced by a **JSON schema** (Ollama `format`):

```json
{
  "summary": "2-4 sentences",
  "points": [{"text": "…", "cites": ["H2", "R2"]}],
  "for_clinician_review": {"text": "…", "cites": ["R3"]}
}
```

- Settings: `temperature 0`, `seed 0`, `keep_alive "15m"`, timeout 120 s.
- Fixed limitations and the banner are added **by code**, not left to the LLM.

### The fact-checker (`verifier.py`)
A draft passes only if all checks pass:

1. Valid JSON matching the schema.
2. Every point cites at least one ID, and every cited ID exists in this case.
3. **Every number** in a point appears in its cited findings (at the displayed
   precision; percentages within ±0.5). Numbers in the summary must appear somewhere in
   the findings.
4. **Every clinical term** (beat class, lab, vital, drug) in a point belongs to a cited
   finding. Known wrong terms ("V wave", "S wave", "P-wave" used for beats) fail.
5. **Banned content:** diagnosis words ("diagnosed", "confirms", "has VT"), certainty
   ("definitely"), treatment or dosing ("give", "start", "administer", "<number> mg"),
   and anything about the outcome (death, survival, discharge).

On failure: **one retry** with the list of problems sent back to the LLM. Second failure:
the template is shown, with a note "the AI draft failed the fact-check" and the reasons
behind a "details" toggle. The page always says which one you are reading:
**"AI-written, fact-checked ✓"** or **"Template (AI draft rejected / LLM off)"**.

---

## 12. Chat agent

- Opens after the report. Context: the case's findings, rule matches, the glossary, and
  the last 6 chat turns. Nothing else.
- Output schema: `{"answerable": true|false, "answer": "…", "cites": ["C3", "G2"]}`.
- The same fact-checker runs on every answer. Fail → one retry → fixed reply:
  "I can only answer from this case's findings."
- Refused on purpose (fixed replies, no LLM call needed when detected by simple
  patterns, LLM `answerable=false` otherwise): treatment or dosing questions, "does this
  patient have …" diagnosis questions, questions about the outcome, anything outside the
  case and the glossary.
- No word-by-word streaming: an answer appears only after it passes the check. A
  "thinking… 7 s" indicator shows meanwhile.
- Glossary (`configs/glossary.toml`, ~20 entries, each with a source): N/S/V/F/Q beats,
  R peak, RR interval, ectopy, run, NSVT, AAMI classes, lactate, troponin, potassium,
  MAP, SpO2, ICU index time, risk score, uncalibrated, held-out vs trained-on, simulated
  case, fact-check.

---

## 13. LLM client and model choice

- `llm.py` talks to `POST http://localhost:11434/api/chat` with `stream: false`, via
  `urllib` (no new dependency).
- Refuses any base URL that is not localhost/127.0.0.1 (raises before sending).
- `available()` checks `GET /api/tags`; the page shows a status chip:
  **LLM ready (llama3.1:8b)** / **LLM off — template reports only**.
- Never runs `ollama pull`. A missing model is reported with the command for you to run.
- On server start, one tiny warm-up request loads the model so the first report is not
  the slow one.

### Bake-off (`compare_llms.py`)
- 20 simulated cases: 10 held-out records (fixed 20 s windows) × 2 ICU stays each,
  fixed list, seed recorded.
- Each model writes all 20 reports.
- Scored per model: fact-check pass rate (first try, after retry), violations by type,
  **coverage** (share of matched rule IDs the report cites), median and max time.
- Decision rule, fixed before the run: highest pass-after-retry; ties → higher coverage;
  then faster.
- Aggregates to `PlanDoc/agentic_results/llm_bakeoff.md`; raw drafts to the data root.

---

## 14. The frontend

### Screen

```
┌───────────────────────────────────────────────────────────────────────┐
│ ⚠ SIMULATED CASE — ECG: MIT-BIH record 208. Chart: MIMIC-III Demo     │
│   ICU stay 2xxxxx. Different people. Research demo, not for care.     │
├───────────────────────────────────────────────────────────────────────┤
│ ECG [208 ▾ held-out]  start [0:00]  length [20 s ▾]   LLM ● ready     │
│ Patient [ICU stay 2xxxxx · 74 F · CHF ▾]            [▶ Run] [Step ▸]  │
├──────────────────────────────────────────────┬────────────────────────┤
│ ① Raw ECG       strip on ECG paper            │ 💬 Ask about this case │
│ ② Cleaned       filtered signal overlay       │                        │
│ ③ Beats found   ▼ markers + "annotation" tag  │ You: why is the risk   │
│ ④ Beats cut out 8 small beat windows          │      high?             │
│ ⑤ Model says    N N V N V  (coloured)         │ AI:  … lactate 4.1     │
│                 N N V N N  true (display only) │      [C4] …  ✓ checked │
│ ⑥ Heart agent   H1…Hn findings                │                        │
│ ⑦ Chart agent   C1…Cn findings + risk*        │                        │
│ ⑧ Rules         R2, R3 matched · draft badge  │                        │
│ ⑨ Report        AI-written, fact-checked ✓    │ [type a question…]     │
│ Fusion (team brain): BLOCKED — needs same-patient ECG + chart          │
│ [Reveal real outcome]  (display only; never given to the AI)          │
└──────────────────────────────────────────────┴────────────────────────┘
```

### Behaviour
- **Run** reveals each stage as its event arrives. **Step ▸** holds arrived stages and
  reveals one per click — for walking a class through it.
- Stage ⑤: hover a beat for its five scores; beats where model ≠ truth get a ring.
- Clicking a finding ID anywhere (report, chat) highlights that finding, and for heart
  findings the beats on the strip.
- **Reveal real outcome**: separate request, shown in its own box, kept out of the case
  data used by the report and chat.
- Phone width works (stacked layout); light and dark mode follow the system.
- Visual style reuses the colour tokens of `docs/index.html` (ECG-paper grid) with
  system fonts — no Google Fonts, no CDN.
- Drawing: `<canvas>` for the strip (7,200 points for 20 s at 360 Hz), small inline SVG
  for the beat windows.

### Server endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/` and `/static/{index.html,app.js,app.css}` | the page (fixed file map; no other paths) |
| GET | `/api/options` | offered records with tags and length; ICU stays (id, age, sex, admission reason); LLM status |
| GET | `/api/run?record=&start=&seconds=&stay=&peaks=annotation\|detected` | event stream, one event per stage, then `done` with a `case_id` |
| POST | `/api/chat` | `{case_id, question}` → `{answer, cites, verified, refused}` |
| GET | `/api/outcome?stay=` | the real outcome, display only |

- Cases live **in memory** (last 20), keyed by `case_id`. Nothing written to disk unless
  you press **Save report**, which writes to `~/physionet-data/processed/agentic/reports/`.
- Model loading: the ECG ensemble and the MIMIC model load once at start; a lock keeps
  ECG inference and LLM calls one at a time.
- Security: binds `127.0.0.1` only; rejects requests whose `Host` header is not
  `127.0.0.1:<port>` or `localhost:<port>` (stops other websites reaching it through the
  browser); no CORS headers; request size limit on `/api/chat`.

---

## 15. When things go wrong

| Situation | What happens |
|---|---|
| Ollama not running | chip shows **LLM off**; report = template (labelled); chat says "start Ollama with `ollama serve`" |
| Model not pulled | same, naming the model and the `ollama pull` command for you to run; never pulls itself |
| LLM slow (> 120 s) | template report; note on screen |
| LLM fails the fact-check twice | template report + reasons under "details" |
| Window past the record end / too short | the request is rejected with the reason; the form shows it |
| Paced record requested by URL | rejected: "paced records are outside what the model was built for" |
| MIMIC outputs missing | stages ⑦–⑧ show "not available — run build_features, make_splits, train"; the case continues ECG-only |
| MIT-BIH or the ECG bundle missing | the server refuses to start and prints what to run |
| No beats classified | heart agent emits one "no beats classified" finding; rules that need beats are skipped |

---

## 16. Testing

All tests run offline with a **fake LLM** (returns scripted JSON), so `pytest` stays fast
and needs no Ollama. Real-Ollama tests carry a new marker `requires_ollama` (skipped
when the server is down), added to `pytest.ini`.

| Area | Tests |
|---|---|
| heart agent | class shares; runs of 3+ found, runs of 2 ignored, runs broken by a dropped beat; heart rate from known peaks; Q-share flag |
| chart agent | findings from a synthetic stay; medication on the index day excluded, day before included; lab flag uses last value; never reads `label`/post-index columns |
| leakage | spy on every prompt: no truth labels, no outcome, no `hospital_expire_flag`/`deathtime` text |
| rules | loader rejects missing source; each draft rule fires and does not fire on edge values |
| fact-checker | passes a faithful draft; catches an invented number, an unknown ID, a term not in the cited finding, "V wave", a dose, a diagnosis word, an outcome word |
| report agent | template on LLM off / timeout / invalid JSON / two failed checks; retry happens exactly once |
| chat agent | glossary question answered with a G citation; treatment question refused without an LLM call; outcome question refused |
| LLM client | refuses non-localhost URL before any request; connection refused → "unavailable" |
| server | starts on a free port in a thread; `/api/options`; `/api/run` emits stages 1–9 in order then `done`; bad `Host` header → 403; non-loopback bind refused; unknown static path → 404 |
| CLI | `--help` for the three scripts (joins `tests/test_scripts.py`) |

The page's JavaScript has no test framework (that would need Node). It is kept small;
its behaviour is covered by the server tests plus a written manual checklist, run once
in a real browser before the phase is called done.

Target: the existing 633 tests still pass; roughly 80–100 new ones.

---

## 17. Build order

Each phase ends with its tests passing and `PlanDoc/IMPLEMENTATION.md` updated.

| Phase | Builds | Done when |
|---|---|---|
| **P1 — agents, no LLM** | `findings`, `ecg_window`, `heart_agent`, `chart_agent`, `rules` + draft rules TOML, template report, `orchestrator`, `scripts/agentic/report.py --llm off` | the CLI prints a full template report for record 208 + a stay; leakage tests pass |
| **P2 — LLM layer** | `llm`, `verifier`, LLM report with retry/fallback, `chat_agent`, glossary, `compare_llms.py` | fact-checker tests pass; bake-off run; model choice written to `PlanDoc/agentic_results/llm_bakeoff.md` |
| **P3 — web app** | `frontend/server.py`, page, stage events, chat panel, Step mode, outcome reveal | server tests pass; manual browser checklist done on a held-out record |
| **P4 — docs** | README, COMMANDS, IMPLEMENTATION, a 5-minute demo script for the professor | commands copy-paste and run |

How you will run it at the end:

```bash
ollama serve                                  # if not already running
python scripts/agentic/serve.py --open        # opens http://127.0.0.1:8765
python scripts/agentic/report.py --ecg-record 208 --seconds 20 --stay <icustay_id>   # CLI version
```

---

## 18. Out of scope (for now)

- Uploading an ECG photo or scan (would need image digitizing — a research problem).
- Uploading signal files from other devices (needs resampling and quality checks).
- Open-ended medical chat.
- Training the fusion network (stays BLOCKED until same-patient data exists).
- Any deployment, public link, GitHub Pages copy, multi-user use, or database.
- New pip packages or new Ollama models.

---

## 19. Permissions check

| Action in this plan | Status under `PERMISSIONS.md` |
|---|---|
| New files under `src/`, `scripts/`, `tests/`, `configs/`, `PlanDoc/` | allowed |
| Calling Ollama on localhost with case findings | allowed (amended 2026-10-15) |
| Running a local server on 127.0.0.1 | allowed (local, reversible) |
| Writing reports under `~/physionet-data/processed/agentic/` | allowed |
| Editing `pytest.ini` to add the `requires_ollama` marker | allowed (project file) |
| Installing anything | **not needed** |
| `ollama pull` | **not needed**; would need your OK |
| Committing this plan or the code | **needs your OK** |

---

## 20. Open questions for you

1. **Rule sources.** Do you have cardiology guidelines or course material to base the
   rules on? If not, I draft the 12 rules in §10, each marked "needs professor review".
2. **Real outcome reveal.** Keep the "Reveal real outcome" button (display only, never
   given to the AI)? Recommended: yes — it lets you compare the risk score to reality, as
   `show_predictions.py` does. *(Built as recommended, 2026-10-15: `frontend/outcome.py`,
   display only; easy to remove if you prefer.)*
3. **Screenshots for the professor.** Screenshots show per-patient values, so by default
   they go under the data root, not the repository. Is that right, or is the open Demo
   acceptable in `PlanDoc/` for screenshots only?
