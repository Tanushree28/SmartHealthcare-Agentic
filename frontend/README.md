# Frontend: watch the agent pipeline run

A local web page for the agent pipeline (Part B of
[PlanDoc/agentic_with_fe.md](../PlanDoc/agentic_with_fe.md)). Pick an ECG
recording and an ICU stay, press **Run**, and watch the nine stages fill in:
raw ECG → cleaned → beats found → beats cut out → model labels → heart agent →
chart agent → rules → report. Then ask questions in the chat.

It runs **on this computer only** (127.0.0.1) and loads nothing from the
internet. Research demo, not for patient care.

## Run it

```bash
cd "/Users/tanushreenepal/Desktop/Smart Healthcare"
source venv/bin/activate
python frontend/server.py --open
```

`--open` opens http://127.0.0.1:8765 in your browser. Keep the Ollama app open
for the AI report and the chat. Stop the server with **Ctrl+C**.

| Option | Does |
|---|---|
| `--llm off` | no AI: plain template reports, chat off |
| `--llm qwen2.5` | use the other local model |
| `--port 8766` | if 8765 is already in use |

## What you see

| Part of the page | What it shows |
|---|---|
| Banner | **SIMULATED CASE** when the ECG and the chart come from different people (always, for MIT-BIH + MIMIC-III) |
| ECG recording menu | grouped **held-out** (the model never saw it: honest test) and **trained-on** (the model studied it: looks better than it is) |
| ICU stay menu | grouped by whether the risk model was trained on that patient |
| LLM chip | green when the local model is ready |
| ① Raw ECG | the recording on ECG paper |
| ② Cleaned | after the band-pass filter (raw in grey for comparison) |
| ③ Beats found | orange markers at each beat |
| ④ Beats cut out | the first 8 beat windows, as the model sees them |
| ⑤ Model says | the model's label above each beat, the cardiologist's below (**display only**; a red ring where they disagree); hover a beat for its scores |
| ⑥ Heart agent, ⑦ Chart agent | the facts, each with an ID (H…, C…) |
| ⑧ Rules | matched draft rules, what they were based on, and their source |
| ⑨ Report | **AI-written, fact-checked ✓** or the template, with the reasons any AI draft was rejected |
| Chat | answers from this case's facts and the glossary only, each fact-checked; treatment, diagnosis and outcome questions are refused |
| Real outcome | what happened to the ICU patient, **display only**: the agents, the report and the chat never see it |
| Team brain (fusion) | BLOCKED: needs an ECG and a chart from the same patient |

Click any ID (H3, C5, R1…) to highlight that fact everywhere; clicking a run
of V beats highlights those beats on stage ⑤. **Step ▸** shows one stage per
click (good for explaining it to someone); **▶ Show all** reveals the rest.

## Files

| File | Job |
|---|---|
| `server.py` | the web server: 127.0.0.1 only, Host-header and cross-site checks, Content-Security-Policy, streams stages as server-sent events |
| `app.py` | the logic behind each request (options, run, chat, outcome); testable without HTTP |
| `serialize.py` | turns stage events into JSON |
| `outcome.py` | the real outcome, display only (kept out of `src/agentic` on purpose) |
| `static/index.html`, `app.css`, `app.js` | the page (plain HTML/CSS/JS, no build step) |

Tests: `pytest tests/frontend -q` (35 tests, synthetic data, no LLM).
