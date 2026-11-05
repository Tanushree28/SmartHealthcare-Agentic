# Agents: what to read, what to run

A step-by-step guide to the agent pipeline: which files explain it, and how to
run it yourself, either as a **web page** in your browser or as commands in the
**terminal**. Every command below was run on the real data on 2026-10-15; the
numbers quoted are what it printed.

Design: [PlanDoc/agentic_with_fe.md](PlanDoc/agentic_with_fe.md) ·
Build plans: [agents](PlanDoc/plans/2026-10-15-agents-end-to-end.md), [web page](PlanDoc/plans/2026-10-15-frontend.md) ·
All other commands: [COMMANDS.md](COMMANDS.md) (sections 9b and 9c)

---

## Quick start (copy, paste, go)

Open the **Ollama app** first. Then in the VS Code terminal:

```bash
cd "/Users/tanushreenepal/Desktop/Smart Healthcare"
source venv/bin/activate
python frontend/server.py --open
```

Your browser opens the page. Pick **233 · held-out** and **224063 · 78 F · VF
ARREST**, press **▶ Run**, and watch the nine stages fill in. Stop with
**Ctrl+C** in the terminal. Full walk-through: [Run the web page yourself](#run-the-web-page-yourself).

Prefer the terminal? `python scripts/agentic/report.py --ecg-record 233 --stay 224063`
(see [Run it in the terminal](#run-it-in-the-terminal-step-by-step)).

---

## The idea in one picture

```
 MIT-BIH heartbeat ─▶ ECG model ─▶ Heart agent ─┐
                                                 ├─▶ facts (H…, C…) ─▶ Rules (R…) ─▶ AI writer ─▶ Fact-checker ─▶ report
 MIMIC-III ICU stay ─▶ risk model ─▶ Chart agent ┘                                    (local Ollama)      │
                                                                                                         └─▶ Chat (facts + glossary)
```

- **Heart agent**: labels every beat with the trained model and writes facts
  ("26.7% V beats", "heart rate 102/min").
- **Chart agent**: reads one ICU patient's first 24 hours (labs, vitals,
  medicines, reason for admission) and writes facts. It never sees whether the
  patient lived or died.
- **Rules**: 12 draft medical rules, each with a source, marked "needs
  professor review".
- **AI writer**: your local llama3.1:8b turns the facts into a short report.
- **Fact-checker**: every number and medical word in the AI's text must match a
  fact, or the AI's text is rejected and the plain report is shown instead. It
  also blocks diagnoses, doses, and anything about the patient's outcome.
- **Chat**: answers only from the facts and a small glossary.

Everything runs on this laptop; nothing leaves it.

---

## Files to read, in this order

| # | File | What it is | Look at |
|---|---|---|---|
| 1 | [PlanDoc/agentic_with_fe.md](PlanDoc/agentic_with_fe.md) | The plan in plain words | the **Roadmap** at the top, then §7–§12 |
| 2 | [src/agentic/findings.py](src/agentic/findings.py) | The "fact card" every helper fills in (ID, sentence, value, evidence) | `Finding`, `Case` |
| 3 | [src/agentic/heart_agent.py](src/agentic/heart_agent.py) | **Heart agent**: beat labels → facts (% V beats, heart rate, runs) | `analyze()` |
| 4 | [src/agentic/chart_agent.py](src/agentic/chart_agent.py) | **Chart agent**: ICU stay → facts (labs, vitals, medicines, risk) | `analyze()` at the bottom |
| 5 | [configs/agent_rules.toml](configs/agent_rules.toml) | **The 12 rules**, readable without code | each `[[rule]]` block |
| 6 | [src/agentic/report_agent.py](src/agentic/report_agent.py) | What the AI is told | `SYSTEM_PROMPT` |
| 7 | [src/agentic/verifier.py](src/agentic/verifier.py) | **Fact-checker** | the comment at the top lists its 5 checks |
| 8 | [src/agentic/chat_agent.py](src/agentic/chat_agent.py) and [configs/glossary.toml](configs/glossary.toml) | **Chat** and the definitions it may use | `BLOCKED` = questions it refuses |
| 9 | [src/agentic/orchestrator.py](src/agentic/orchestrator.py) | Runs stages ①–⑨ in order | `run_case()` |
| 10 | [scripts/agentic/report.py](scripts/agentic/report.py) | The command you type | `main()` |

For examples of what the fact-checker rejects, see
[tests/agentic/test_verifier.py](tests/agentic/test_verifier.py): each line is a
sentence it refuses and why.

---

## Run the web page yourself

### 1. Start the server (terminal)

Open the **Ollama app**. In VS Code open a terminal (**Terminal → New Terminal**)
and run:

```bash
cd "/Users/tanushreenepal/Desktop/Smart Healthcare"
source venv/bin/activate
python frontend/server.py --open
```

The terminal prints:

```
Loading the ECG ensemble and the MIMIC-III outputs...
LLM: llama3.1:8b ready
Open http://127.0.0.1:8765/   (Ctrl+C to stop)
```

and your browser opens the page. If it does not open by itself, paste
**http://127.0.0.1:8765** into the browser. **Leave this terminal running**: the
page works only while it is open. The address works on this computer only.

### 2. Run a case (browser)

1. **ECG recording**: pick **233 · held-out**. *Held-out* = the model never saw
   it (honest test); *trained-on* = it studied it (looks better than it is).
2. **ICU stay**: pick **224063 · 78 F · VF ARREST**.
3. Leave Start **0**, Length **20 s**, Beat positions **cardiologist's marks**.
4. Press **▶ Run**.

### 3. Watch the stages

| Stage | What you see |
|---|---|
| ① Raw ECG | the heartbeat on pink ECG paper |
| ② Cleaned | the same signal after the filter (blue), raw in grey behind it |
| ③ Beats found | an orange marker above every beat |
| ④ Beats cut out | small boxes: the first 8 beats, as the model sees them |
| ⑤ Model says | a letter above each beat (the model: **V** in red, **N** in grey), the cardiologist's letter below; a red ring where they disagree. **Hover a beat** to see its scores |
| ⑥ Heart agent | facts H1-H7, e.g. "26.7% of beats … classified V" |
| ⑦ Chart agent | facts C1-C18: VF ARREST, high troponin, amiodarone, risk score |
| ⑧ Rules | R1, R4, R8, R10, R12, each with "draft - needs professor review" and its source |
| ⑨ Report | a timer *"writing the report… 12 s"* (about 30 s), then **AI-written (llama3.1:8b), fact-checked ✓** and the report |

The banner at the top gets an orange border and ⚠: **SIMULATED CASE … different people**.
Click any ID (`H4`, `R1`, `C3`…) to highlight that fact everywhere.

### 4. Ask questions (right-hand panel)

Type in **Ask about this case** and press Enter:

- *What is a V beat?* → answered from the glossary, **fact-checked ✓**
- *Why is the risk score high?* → answered from fact C18
- *What dose should we give?* → refused: *"I can't suggest treatments or doses…"*

### 5. More to try

- **Reveal real outcome**: what happened to the ICU patient. Display only; the
  AI, the report and the chat never see it.
- **Step ▸** (instead of Run): one stage per click, good for explaining it to
  someone. **▶ Show all** reveals the rest.
- Try **208 · trained-on**: stage ⑤ agrees with the cardiologist on every beat,
  because the model studied that record.
- Pick only an ECG, or only an ICU stay: the other stages say *skipped*.

### 6. Stop it

Press **Ctrl+C** in the terminal (it prints `Stopped.`). The page stops working
until you start the server again.

| Option | Does |
|---|---|
| `python frontend/server.py --open` | start and open the browser |
| `python frontend/server.py --llm off` | no AI: plain template reports, chat off |
| `python frontend/server.py --llm qwen2.5` | use the other local model |
| `python frontend/server.py --port 8766` | if port 8765 is busy |

What each part of the page means: [frontend/README.md](frontend/README.md).

---

## Run it in the terminal, step by step

### Step 0 — get ready

Keep the **Ollama app open**. Then:

```bash
cd "/Users/tanushreenepal/Desktop/Smart Healthcare"
source venv/bin/activate
```

Your prompt shows `(venv)`. You do **not** need `ollama serve` while the app is
open; if you run it anyway, `address already in use` just means Ollama is
already running.

### Step 1 — see what you can choose from

```bash
python scripts/agentic/report.py --list
```

Prints 44 heartbeat recordings, each tagged **held-out** (the model never saw
it; the honest test) or **trained-on** (the model studied it), and 51 ICU
patients with age, sex and reason for admission.

### Step 2 — the heart agent alone (instant)

```bash
python scripts/agentic/report.py --ecg-record 233 --seconds 20 --llm off
```

Facts `[H1]`–`[H7]`: **26.7% V beats**, **average heart rate 102/min**, one beat
the model was unsure about. Rules **R1** (frequent V beats) and **R4** (fast
heart rate) match. The chart stage says *skipped*: no patient was chosen.

### Step 3 — the chart agent alone (instant)

```bash
python scripts/agentic/report.py --stay 224063 --llm off
```

The ECG stages say *skipped*. 18 chart facts `[C1]`–`[C18]`: reason for
admission (VF ARREST), high troponin and CK-MB, medicines (amiodarone, heparin),
the risk score. Rules **R8** (troponin) and **R12** (risk) match.

### Step 4 — both together, no AI (instant)

```bash
python scripts/agentic/report.py --ecg-record 233 --stay 224063 --llm off
```

The report now starts with **"SIMULATED CASE … different people"**. More rules
match because some combine both sides (R10: V beats although an antiarrhythmic
was started). This is the plain report: always correct.

### Step 5 — add the AI writer (about 30–60 s)

```bash
python scripts/agentic/report.py --ecg-record 233 --stay 224063
```

The first line says `LLM: llama3.1:8b ready`. Line ⑨ says **"AI-written,
fact-checked"**, then a short readable summary. Every point ends with tags such
as `[H4]` or `[C3]`; look them up in Step 4's output. If it says **"template"**
instead, the fact-checker rejected the AI's text and lists why. That is the
safety net working, not an error.

### Step 6 — ask questions

```bash
python scripts/agentic/report.py --ecg-record 233 --stay 224063 --ask "What is a V beat?" --ask "Why is the risk score high?"
```

Each answer ends with its source (`[G3]` = glossary, `[C18]` = a chart fact)
and **(fact-checked)**.

### Step 7 — watch the safety rules

```bash
python scripts/agentic/report.py --stay 224063 --llm off --ask "What dose should we give?" --ask "Did the patient survive?"
```

Both are refused at once: *"I can't suggest treatments or doses…"* and *"I don't
have the patient's outcome; the AI is never given it."*

### Step 8 — chat freely

```bash
python scripts/agentic/report.py --ecg-record 233 --stay 224063 --chat
```

Type questions at `ask>`. Press Enter on an empty line to stop.

### Step 9 — save a report and open it

```bash
python scripts/agentic/report.py --ecg-record 233 --stay 224063 --save
open ~/physionet-data/processed/agentic/reports/record233_stay224063.txt
```

Saved outside the project folder, because it contains patient values.

### Step 10 — honest test vs flattering test

```bash
python scripts/agentic/report.py --ecg-record 208 --llm off     # trained-on
python scripts/agentic/report.py --ecg-record 222 --llm off     # held-out
```

Compare line ⑤, *"model and cardiologist agree on X of Y beats"*. On a
trained-on record it is near-perfect (record 208: 29 of 29) because the model
studied it. Held-out records are the honest test, where disagreements can show
up.

### Step 11 — the automatic tests

```bash
pytest tests/agentic -q
```

Expected: **122 passed, 4 deselected** in a few seconds. The 4 deselected use the
real data or the real AI. Run them on purpose with:

```bash
pytest tests/agentic -m requires_data        # real MIT-BIH + MIMIC-III Demo
pytest tests/agentic -m requires_ollama      # real local AI
pytest tests/frontend -q                     # the web page's server: 35 passed
```

---

## How to read the output

| You see | It means |
|---|---|
| ① … ⑨ | the nine stages: raw ECG, cleaned, beats found, beats cut out, model labels, heart agent, chart agent, rules, report |
| *skipped (…)* | that stage had nothing to work on, and says why |
| `[H…]` | a fact from the **heart** agent |
| `[C…]` | a fact from the **chart** agent |
| `[R…]` | a **rule** that matched, with its source and "draft" status |
| `[G…]` | a **glossary** definition (chat only) |
| line ⑤ | model vs cardiologist: **shown to you only, never given to the AI** |
| AI-written, fact-checked | the AI's text passed every check |
| template (…) | the plain report; the reason is in brackets |

---

## Useful options (terminal: `scripts/agentic/report.py`)

| Option | Does |
|---|---|
| *(no `--llm`)* or `--llm on` | use the AI (llama3.1:8b) — the default |
| `--llm off` | no AI: instant plain report |
| `--llm qwen2.5` | use the other local model |
| `--start 60` | begin 60 s into the recording |
| `--seconds 30` | window length (at least 2 s) |
| `--detect-peaks` | find beats with the detector instead of the cardiologist's marks |
| `--save` | write the report to `~/physionet-data/processed/agentic/reports/` |

---

## If something goes wrong

| Message | Fix |
|---|---|
| `ModuleNotFoundError` | run `source venv/bin/activate` first |
| `Ollama is not reachable … start it with ollama serve` | open the Ollama app (or run `ollama serve`) |
| `model … is not pulled` | the model is missing; pulling one downloads GBs, so decide first |
| `error: the window ends at … s but record … is … s long` | use a smaller `--start` or `--seconds` |
| `error: a window must be at least 2 s long` | use `--seconds 2` or more |
| `error: ICU stay … is not in the cohort` | pick a stay from `--list` |
| `Port 8765 is not free … Try --port 8766` | the server is already running in another terminal (use that one, or stop it with Ctrl+C), or run `python frontend/server.py --port 8766 --open` |
| browser: *"This site can't be reached"* | the server is not running: start it (step 1 of the web page) |
| page: *"Could not load the options … Is the server running?"* | the server stopped; start it again and reload the page |
| page: LLM chip is grey *"LLM ○ off"* | Ollama was closed when the server started: open the Ollama app, stop the server (Ctrl+C) and start it again |
| page: report says *Template report* | the AI's draft failed the fact-check (open *"Why the AI draft was rejected"*) or the AI is off: the safety net working |

Research demo only. Not a diagnosis and not for patient care.
