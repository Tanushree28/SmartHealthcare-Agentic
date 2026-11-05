# COMMANDS.md — copy-paste reference

Every command you need, in the order you'd normally use them.
All of these are safe to re-run.

The project has three modules. Each has its scripts in `scripts/<module>/` and
its tests in `tests/<module>/`:

| Module | What | Commands |
|---|---|---|
| `mit_bih` | ECG beats: preprocessing, baseline, neural classifier, inference | Sections 2–6 |
| `mimic_iii` | ICU clinical data: features, split, model | Section 8 |
| `multimodal` | One inference interface for both; fusion (blocked) | Section 9 |
| everything | Setup check, whole workflow | Sections 0 and 10 |

> **Paths changed on 2026-10-14.** Scripts moved one folder down:
> `scripts/train_baseline.py` is now `scripts/mit_bih/train_baseline.py`.
> An old command fails with `No such file or directory`.

---

## Start here, every time

```bash
cd "/Users/tanushreenepal/Desktop/Smart Healthcare"
source venv/bin/activate
```

Your prompt shows `(venv)` once it works. Run `deactivate` when you're done.

> Skipping this gives `ModuleNotFoundError` — plain `python` is your system
> Python, which doesn't have the packages. If you'd rather not activate, put
> `./venv/bin/` in front of every `python` below.

---

## 0. What does this machine have?

```bash
python scripts/validate_setup.py
```

Reads only. Lists every dataset, model and dependency as `OK`, `MISSING` (names
the command that makes it), `BLOCKED` (needs something outside this repo, such
as MIMIC-III credentialing) or `FAIL` (present but wrong). Today: 16 OK,
1 MISSING (ECG calibration, Phase 1f Stage B), 3 BLOCKED (full MIMIC-III,
paired data, fusion training), 0 FAIL.

---

## 1. Check everything still works

```bash
python -m pytest
```

Expect `633 passed, 19 deselected` in about 30 seconds. Needs no data — it makes up
its own recordings and its own MIMIC-shaped tables.

```bash
python -m pytest -m requires_data
```

Expect `15 passed`. These check facts about the real MIT-BIH database and the
MIMIC-III Demo on disk.

```bash
python -m pytest -q                            # quieter
python -m pytest tests/mit_bih                 # one module's tests
python -m pytest tests/mimic_iii tests/multimodal
python -m pytest tests/mit_bih/test_io.py      # just one file
python -m pytest -k lead                       # just tests matching "lead"
```

```bash
python -m pytest tests/mit_bih/test_imbalance_regression.py -m slow
```

Expect `4 passed` after about 5 minutes. These train small networks on fake
data to check the imbalance handling still works. Run them after any change to
the loss functions, the training loop or the model.

---

## 2. Get / check the data

```bash
python scripts/mit_bih/download_mitbih.py --verify
```

No internet needed. Expect `48/48 records verify.`

```bash
python scripts/mit_bih/download_mitbih.py
```

Downloads anything missing (~94 MB). Already-complete records are skipped, so
re-running costs nothing.

---

## 3. Preprocess

```bash
python scripts/mit_bih/preprocess_mitbih.py
```

Default seeded split. Writes to `~/physionet-data/processed/mitdb/`. Takes ~2 s.

```bash
python scripts/mit_bih/preprocess_mitbih.py \
  --strategy ds1_ds2 \
  --out-dir ~/physionet-data/processed/mitdb-ds12
```

The split used in published papers. **Required before training the baseline.**

```bash
python scripts/mit_bih/preprocess_mitbih.py --records 100 101 200   # a few records only
python scripts/mit_bih/preprocess_mitbih.py --include-paced         # keep records 102/104/107/217
python scripts/mit_bih/preprocess_mitbih.py --help                  # all options
```

---

## 4. Train the baseline classifier

```bash
python scripts/mit_bih/train_baseline.py \
  --processed-dir ~/physionet-data/processed/mitdb-ds12
```

Takes ~1 minute. Prints per-class recall, confusion matrices, and a summary
table; writes `baseline_metrics.json` next to the data.

```bash
# Just the main model, skip the ablations
python scripts/mit_bih/train_baseline.py \
  --processed-dir ~/physionet-data/processed/mitdb-ds12 \
  --classifiers random_forest --feature-sets rr+morphology
```

> If it refuses to run with a `NormalizationMismatchError`, that is the leakage
> tripwire doing its job: the beats on disk were scaled for a different split.
> The error message tells you which command to run. Follow it.

---

## 4b. Train the neural classifier (Phases 1d–1g, on Colab)

Training runs on a Colab GPU. Locally on CPU a single phase takes hours.

```bash
# 1. Bundle the code (not the data) for Colab
python scripts/mit_bih/package_for_colab.py --out ~/Desktop/phase-colab.zip
```

2. Upload the zip and open the phase's notebook from `notebooks/mit_bih/`:
   `phase1d_colab.ipynb`, `phase1e_colab_v2.ipynb` or
   `phase1g_colab_ensemble_mix.ipynb`. The notebook runs the tests,
   downloads and preprocesses MIT-BIH, checks the fold assignment, then trains. The training commands it ran:

```bash
python scripts/mit_bih/train_nn.py           --beats-dir <beats> --out-dir <out> --device cuda --epochs 40               # 1d
python scripts/mit_bih/train_nn_v2.py        --beats-dir <beats> --out-dir <out> --device cuda --epochs 30 --seeds 3 --folds 5   # 1e
python scripts/mit_bih/train_ensemble_mix.py --beats-dir <beats> --out-dir <out> --device cuda --epochs 50 --folds 5     # 1g
```

```bash
# 3. Download the results zip from Colab, then bring it into the repo
python scripts/mit_bih/ingest_colab_results.py ~/Downloads/phase1g-results.zip
```

This copies the metrics into `results/` and the checkpoints into `models/`.
Do not commit the results zip itself, because its checkpoints duplicate
`models/`.

```bash
# Calibration diagnosis (1f), runs locally on the checkpoints in models/
python scripts/mit_bih/diagnose_calibration.py
```

---

## 4c. ECG inference bundle (classify a raw signal)

```bash
python scripts/mit_bih/export_inference_bundle.py     # writes models/mit_bih_inference_bundle.json
```

Already done; re-run only if the checkpoints or the DS1/DS2 normalization
change. `--ensemble A_focal_only` or `focal_single` exports another model.

```bash
python scripts/mit_bih/evaluate_bundle.py               # all of DS2, saved beats, ~7 min
python scripts/mit_bih/evaluate_bundle.py --from-raw    # all of DS2 from raw records, ~9 min
python scripts/mit_bih/evaluate_bundle.py --records 100 208   # a quick subset
```

Expect `Headline macro-F1 (N/S/V/F): 0.5554` and `difference +0.0000` against
the published number for the full runs. A subset run says it is not comparable.
The scores are uncalibrated: rankings, not probabilities.

---

## 4d. External test on INCART (new patients, new equipment)

Design: `PlanDoc/specs/2026-10-29-incart-external-test-design.md`. Run in this order.

```bash
python scripts/incart/download_incart.py            # 75 records, ~800 MB, open access (network)
python scripts/incart/download_incart.py --verify   # check what is on disk, no network
python scripts/incart/resampling_check.py           # DS2 as recorded vs 360 -> 257 -> 360 Hz, ~12 min
python scripts/incart/evaluate_on_incart.py --smoke 2   # pipeline check: counts only, no scores
caffeinate -i python scripts/incart/evaluate_on_incart.py   # the external test, ~25 min; caffeinate stops the Mac idling to sleep
pytest tests/incart -m requires_data                # one real INCART record loads and maps
```

The resampling check must print `Reproduces published 0.5554: yes`; the
INCART run refuses to start otherwise. INCART is scored once: a second run
refuses unless given `--overwrite`, and any rerun must be explained in the
write-up. Outputs go to `$SMART_HEALTH_DATA/processed/incartdb/`; the frozen
copies are in `results/` (`*-incart-*`).

---

## 5. Rebuild the results page

The page published at GitHub Pages is generated from your processed data, so
its numbers can never drift from the pipeline.

```bash
python scripts/mit_bih/build_report.py
```

Writes `docs/index.html`. Preview it before pushing:

```bash
python -m http.server -d docs 8000
# then open http://localhost:8000
```

Re-run it after any change to the pipeline, the model, or the page templates
in `site/`. It refuses to run if the data it needs is missing, and names the
command that produces it.

---

## 6. Look at the results

```bash
open ~/physionet-data/processed/mitdb/manifest.csv          # per-record spreadsheet
cat  ~/physionet-data/processed/mitdb/splits.json           # who's in train/val/test
cat  ~/physionet-data/processed/mitdb-ds12/baseline_metrics.json
du -sh ~/physionet-data/*                                   # how much space it uses
```

---

## 7. Git

To see what changed:

```bash
git status
git diff
```

---

## 8. MIMIC-III (Phase 2)

**Get the Demo** (one time, open access, about 11 MB to download):

```bash
cd ~/physionet-data
curl -L -o mimiciii-demo.zip https://physionet.org/content/mimiciii-demo/get-zip/1.4/
unzip -q mimiciii-demo.zip -d mimiciii-demo-tmp
mv mimiciii-demo-tmp/mimic-iii-clinical-database-demo-1.4 mimiciii-demo
rm -r mimiciii-demo-tmp mimiciii-demo.zip
cd "/Users/tanushreenepal/Desktop/Smart Healthcare"
```

Already done on this machine; it lives in `~/physionet-data/mimiciii-demo/`.
Full MIMIC-III needs PhysioNet credentialing first.

**Run the exploration:**

```bash
python scripts/mimic_iii/explore_demo.py                                   # print the results
python scripts/mimic_iii/explore_demo.py | tee ~/Desktop/mimic_demo.txt    # and save them
```

Prints the cohort steps, mortality split, data coverage and median values for
the first 24 h in the ICU. Aggregates only. This was the spike; the pipeline
below replaces it.

**Run the pipeline** (each step reads the previous step's output from
`~/physionet-data/processed/mimiciii-demo/`):

```bash
python scripts/mimic_iii/build_features.py     # cohort + feature table (~1 s on the Demo)
python scripts/mimic_iii/make_splits.py        # patient-level train/val/test
python scripts/mimic_iii/train.py              # fit, choose on val, score test once, write report.md
```

Expect `51 stays, 15 positive`, a split of `37 / 7 / 7` patients, and a report
headed **PIPELINE VERIFICATION ONLY - NOT CLINICAL PERFORMANCE**. The Demo is too
small and too biased (all 100 patients had died) for any model result to mean
anything.

The task (cohort, 24 h index time, windows, target, seed, model grid) is in
`configs/mimic_iii.toml`. Change it only together with `PlanDoc/`.

**Full MIMIC-III**, once credentialed: put the v1.4 CSVs (`.csv.gz` is fine) in
`~/physionet-data/mimiciii/`, then add `--source full` to all three commands.
Until then `build_features.py --source full` prints `BLOCKED` and exits with 3.
On the full database expect tens of minutes for `build_features.py`: it reads
`CHARTEVENTS` (~330 M rows) in chunks of `--chunksize` rows.

**See each patient's prediction next to what really happened:**

```bash
python scripts/mimic_iii/show_predictions.py              # the 7 test patients
python scripts/mimic_iii/show_predictions.py --split all  # all 51
```

One line per ICU stay: risk score, the model's call (died / survived at the
0.70 threshold), what really happened, right or wrong, and the three features
that pushed the score most. Expect `Right: 4 of 7. Deaths caught: 0 of 2.` on
the test patients: with 7 patients this shows the pipeline, not the model.

> If `train.py` stops with `LeakageError: splits.json was made for a different
> feature table`, the features changed after the split was made. Re-run
> `make_splits.py`.

---

## 9. Multimodal (shared inference; fusion is blocked)

```bash
python scripts/multimodal/infer.py --ecg-record 100                       # ECG only (first 60 s)
python scripts/multimodal/infer.py --clinical-stay 206504                 # one Demo ICU stay
python scripts/multimodal/infer.py --ecg-record 100 --clinical-stay 206504 --out ~/Desktop/result.json
```

Prints JSON with separate `ecg` and `clinical` sections, each saying what it
predicts and what its model was trained on. An MIT-BIH record and a MIMIC-III
stay are different people, so the output says linkage is not asserted, and
`fusion` is `BLOCKED`. Options: `--seconds 0` (whole record), `--detect-peaks`
(use the QRS detector instead of the annotations), `--include-beats`,
`--clinical-json file.json`.

```bash
python scripts/multimodal/demo_fusion_synthetic.py
```

Watch the fusion network learn on **invented** patients (a few seconds): it
scores new made-up patients with both inputs, ECG only and chart only, shows
five examples, then shows the pairing check refusing a real MIT-BIH +
MIMIC-III pair. Nothing real is combined.

```bash
python scripts/multimodal/train_fusion.py                    # prints BLOCKED, exit 3
python scripts/multimodal/train_fusion.py --pairs pairs.csv  # validates a pairing manifest
```

---

## 9b. Agents (report and chat)

```bash
ollama serve                                   # only if Ollama is not already running
python scripts/agentic/report.py --list        # the ECG records (held-out / trained-on) and ICU stays
python scripts/agentic/report.py --ecg-record 208 --seconds 20 --stay 224063
python scripts/agentic/report.py --ecg-record 208 --stay 224063 --ask "What is a V beat?"
python scripts/agentic/report.py --ecg-record 208 --stay 224063 --chat   # keep asking; empty line ends
python scripts/agentic/report.py --ecg-record 208 --stay 224063 --llm off  # no LLM: template report
```

- `--list` prints every offered record with its tag and every Demo ICU stay with its age, sex and reason for admission.
- A run prints the nine stages (raw ECG to report), then the report: **AI-written, fact-checked** when the local LLM's draft passes, otherwise the plain template with the reasons it was rejected.
- `--ask` (repeatable) and `--chat` answer from this case's findings and the glossary only; treatment, diagnosis and outcome questions are refused.
- `--llm qwen2.5` uses the other local model; `--start 60` starts 60 s into the record; `--detect-peaks` finds beats with the QRS detector; `--save` writes the report under `$SMART_HEALTH_DATA/processed/agentic/reports/`.

```bash
python scripts/agentic/compare_llms.py --cases 20   # 20-40 minutes: llama3.1:8b vs qwen2.5
pytest tests/agentic -m requires_data               # real MIT-BIH + MIMIC-III Demo
pytest tests/agentic -m requires_ollama             # real local LLM
```

The bake-off writes aggregate scores to `PlanDoc/agentic_results/llm_bakeoff.md`
and every draft to `$SMART_HEALTH_DATA/processed/agentic/bakeoff/`.

---

## 9c. The web page (frontend)

```bash
python frontend/server.py --open          # opens http://127.0.0.1:8765 in your browser
python frontend/server.py --llm off       # no AI: template reports, chat off
python frontend/server.py --port 8766     # if 8765 is busy
```

Pick an ECG recording and an ICU stay and press **▶ Run**: the nine stages fill
in as they finish, then the report (AI-written and fact-checked, or the
template). **Step ▸** shows one stage per click. The chat and the **Reveal real
outcome** button (display only; never given to the AI) unlock when the run is
done. The server listens on 127.0.0.1 only; stop it with Ctrl+C. Details:
`frontend/README.md`. Tests: `pytest tests/frontend -q`.

---

## 10. Run the whole workflow

```bash
python scripts/run_end_to_end.py                  # ~1.5 min
python scripts/run_end_to_end.py --full-ecg-eval  # + all of DS2 for the ECG model, ~9 min
```

Runs sections 0, 8, 3 (into its own folder), 4c, 9 in order and writes
`~/physionet-data/processed/end_to_end/report.md`. Expect seven `OK` and fusion
training `BLOCKED`. It never trains a neural network and never overwrites the
processed data behind the published MIT-BIH numbers.

---

## Publishing to GitHub Pages

One-time setup, after the first push:

1. Push the repo (see section 7).
2. On GitHub: **Settings** &rarr; **Pages**.
3. Under _Build and deployment_, set **Source** to `Deploy from a branch`.
4. Set **Branch** to `master` and the folder to **`/docs`**. Save.
5. Wait about a minute. The page appears at:

   `https://tanushree28.github.io/SmartHealthcare-Agentic/`

After that, every push that includes a changed `docs/index.html` republishes
automatically. To update the page:

```bash
python scripts/mit_bih/build_report.py
git add docs/index.html
git commit -m "Rebuild results page"
git push
```

> The `docs/.nojekyll` file tells GitHub to serve the HTML as-is instead of
> running Jekyll over it. Leave it in place.

---

## If something breaks

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError` | You skipped `source venv/bin/activate` |
| `No module named 'ecg'` | Old code or notebook; the package is now `mit_bih` (`from mit_bih import ...`) |
| `can't open file 'scripts/....py'` | Old path; scripts are now in `scripts/mit_bih/` or `scripts/mimic_iii/` |
| `DataRootError` | Data missing — run `python scripts/mit_bih/download_mitbih.py` |
| `MIMIC-III Demo not found` | Download it — see section 8 |
| `BLOCKED: full MIMIC-III not found` | Expected until PhysioNet credentialing is granted |
| `LeakageError: ... different feature table` | Re-run `python scripts/mimic_iii/make_splits.py` |
| `no clinical model at ...` | Run the three MIMIC-III commands in section 8 |
| `SamplingRateError ... 360 Hz` | The ECG model only takes 360 Hz signals; resample first |
| `bundle member ... changed` | A checkpoint in `models/` was modified; restore it with git |
| `NormalizationMismatchError` | Read the message; it names the exact command to run |
| Tests fail after you edit something | `git diff` to see what changed; `git checkout <file>` to undo |
| `build_report.py` says data is missing | It names the exact command to run first |
| Pages shows a 404 | Check Settings &rarr; Pages points at `master` + `/docs`; give it a minute |

Rebuild the environment from scratch if needed:

```bash
rm -rf venv
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Use a different data location:

```bash
export SMART_HEALTH_DATA=/Volumes/MyDrive/physionet-data
```
