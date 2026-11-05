# SmartHealthcare-Agentic

An agentic cardiac-AI system built from two data sources: ECG beats from the
MIT-BIH Arrhythmia Database and ICU clinical records from MIMIC-III. Each source
has its own module, and a third module will combine them.

## Project structure

| Module | Purpose | Status |
|---|---|---|
| [`src/mit_bih/`](src/mit_bih/) | ECG beat preprocessing, the classical baseline, the neural beat classifier, training, evaluation, and inference from a raw signal | **Complete** through Phase 1g; inference bundle added 2026-10-14 |
| [`src/incart/`](src/incart/) | External test of the MIT-BIH beat model on the open INCART database (new patients, new equipment): resampling 257 → 360 Hz, beat-position check, patient bootstrap | **Built 2026-10-31** (step 2 of the open-data plan) |
| [`src/mimic_iii/`](src/mimic_iii/) | MIMIC-III clinical data: ingestion, cardiac cohort, time-window features, patient-level splits, preprocessing, training, evaluation | **Software complete**, run on the open Demo. Model on full MIMIC-III **blocked** (credentialing) |
| [`src/multimodal/`](src/multimodal/) | One inference interface for ECG, clinical or both; a fusion network | Interface and fusion architecture **complete and tested**. Fusion training **blocked** (no paired data) |
| [`src/agentic/`](src/agentic/) | Agents: heart and chart agents turn model outputs into findings, draft rules connect them, a local LLM (Ollama) words the report, a fact-checker gates it, a chat answers from the findings | **Part A built 2026-10-15** (terminal). Rules and glossary are drafts awaiting review |
| [`frontend/`](frontend/) | Local web page: pick a case, watch the nine stages fill in, read the fact-checked report, chat | **Part B built 2026-10-15**; 127.0.0.1 only |
| [`src/common/`](src/common/) | Shared paths, provenance, binary metrics | Complete |

Each module has matching folders: `scripts/<module>/` for command-line entry
points, `tests/<module>/` for its tests, and, where used, `notebooks/<module>/`.
Settings that define the MIMIC-III task live in [`configs/mimic_iii.toml`](configs/mimic_iii.toml).

Dependencies run one way: `mit_bih` and `mimic_iii` never import each other,
and `multimodal` imports both. `incart` imports `mit_bih`; nothing imports
`incart`. All of them may import `common`. Scripts and tests
import from the packages (`from mit_bih import config`), never from each
other's folders.

```
            MIT-BIH ECG                                MIMIC-III tables
                 |                                            |
   mit_bih: filter, segment, normalize,        mimic_iii: ingest (chunked), cohort,
   RR features, 20-network ensemble            24 h features, patient split, model
                 |                                            |
   multimodal.ecg_encoder                     multimodal.clinical_encoder
   per-beat scores + 13-value summary         preprocessed vector + mortality score
                 \                                          /
                  +---- multimodal.inference.run_inference ---+
                  |  validates each input; returns each output |
                  |  separately, labelled; never merges them   |
                  +--------------------------------------------+
                                     |
                       multimodal.fusion.ConcatFusion
                 built and tested on synthetic pairs only; training
                 BLOCKED until a genuinely paired dataset exists
```

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Environment + MIT-BIH acquisition | ✅ Done |
| 1 | MIT-BIH preprocessing | ✅ Done |
| 1b | Classical baseline (Random Forest, LDA) | ✅ Done |
| 1c | Published results page | ✅ Done |
| 1d | Hybrid neural network + imbalance study | ✅ Done (trained on Colab GPU) |
| 1e | Repeated CV, architecture variants, ensembling | ✅ Done (trained on Colab GPU) |
| 1f | Calibration diagnosis, Stage A | ✅ Done |
| 1f | Calibration correction, Stage B | ⬜ Not started. Nothing blocks it |
| 1g | Ensemble of different losses | ✅ Done (trained on Colab GPU) |
| 2 | MIMIC-III exploration on the Demo | ✅ Done (`scripts/mimic_iii/explore_demo.py`) |
| 2 | MIMIC-III pipeline (tested code in `src/mimic_iii/`) | ✅ Done 2026-10-14; 107 tests; runs on the Demo |
| 2 | MIMIC-III mortality model | ⛔ **Blocked by data:** needs full MIMIC-III (PhysioNet credentialing pending). The pipeline trains on the Demo, but the Demo's 100 patients had all died, so no model trained on it means anything |
| — | ECG inference bundle (raw signal → beat scores) | ✅ Done 2026-10-14; reproduces DS2 macro-F1 0.5554 from raw records |
| — | External test on INCART (new patients, new equipment) | ✅ Done 2026-10-31; macro-F1 0.570 vs DS2 0.555: holds up ([results](results/2026-10-31-incart-external-test.md)) |
| 3 | MIMIC waveform / signal-quality layer | ⛔ **Blocked by storage and compute:** the MIMIC waveform set is terabyte-scale |
| 4 | Shared inference interface (ECG, clinical, or both) | ✅ Done 2026-10-14 |
| 4 | Fusion architecture | ✅ Built and tested on synthetic pairs only |
| 4 | Fusion training and clinical validation | ⛔ **Blocked by data:** no genuinely paired ECG + clinical dataset; MIT-BIH and MIMIC-III contain different patients |
| 4 | Agentic layer | ⬜ Not started |

**Software integration is complete. A multimodal model trained and evaluated
on real paired data does not exist.** Those are different claims, and only the
first is true.

**Compute note:** the neural classifier was trained on Google Colab GPUs.
Retraining it on this laptop's CPU is possible but takes hours per phase; the
trained checkpoints in `models/` make retraining unnecessary for evaluation.

See [PlanDoc/PLAN.md](PlanDoc/PLAN.md) for the plan and
[PlanDoc/IMPLEMENTATION.md](PlanDoc/IMPLEMENTATION.md) for what was done.

> **Not a medical device.** Everything here produces research artifacts. Nothing
> in this repository is clinically validated, diagnostic, or fit for patient
> care.

**[View the results page &rarr;](https://tanushree28.github.io/SmartHealthcare-Agentic/)**
&nbsp;&mdash;&nbsp; a visual walkthrough of the pipeline and the baseline results.
It is generated from the processed data by `python scripts/mit_bih/build_report.py`, so
its figures always match the pipeline that produced them.

---

## Setup

Requires Python 3.13 (the checked-in `venv/` was built on 3.13.9, arm64 macOS).

```bash
python3 -m venv venv                     # if venv/ is absent
./venv/bin/pip install -r requirements.txt
```

### Where the data lives

Raw and processed data live **outside the repository**, under the path named by
`SMART_HEALTH_DATA` (default `~/physionet-data`):

```
~/physionet-data/
├── mitdb/                     raw MIT-BIH WFDB records (48 records, ~94 MB)
├── mimiciii-demo/             MIMIC-III Clinical Database Demo v1.4 (26 CSV tables, 94 MB)
└── processed/
    ├── mitdb/                 MIT-BIH, seeded record-wise split
    ├── mitdb-ds12/            MIT-BIH, DS1/DS2 inter-patient split (used by all models)
    │   ├── beats/<record>.npz X (n, 300) float32, y (n,) int8, r_peaks (n,) int64
    │   ├── manifest.csv       per-record accounting
    │   ├── splits.json        train/test record assignment
    │   └── normalization.json z-score stats fitted on training records only
    ├── mimiciii-demo/         MIMIC-III pipeline output on the Demo
    │   ├── cohort.csv.gz, features.csv.gz   one row per ICU stay (row-level: never leaves this machine)
    │   ├── build_summary.json cohort flow, row accounting, ITEMID labels
    │   ├── splits.json        patient-level split + feature-table hash
    │   ├── model.pkl, model_meta.json       fitted pipeline + task, columns, hash
    │   └── report.json, report.md           metrics with bootstrap CIs
    └── end_to_end/            output of scripts/run_end_to_end.py
```

Full MIMIC-III, once credentialed, goes in `~/physionet-data/mimiciii/`.

Keeping it outside the tree stops ~94 MB of binaries entering git history, lets
the dataset survive a re-clone, and keeps every path in `src/` machine-independent.

```bash
export SMART_HEALTH_DATA=/somewhere/else   # optional
```

---

## Running

```bash
./venv/bin/python scripts/mit_bih/download_mitbih.py     # one-time, ~94 MB, network
./venv/bin/python scripts/mit_bih/preprocess_mitbih.py   # ~2 s for all 48 records
```

The download is idempotent — records already present and passing their integrity
check are skipped, so re-running costs nothing. `--verify` checks what is on disk
and makes no network request.

Useful preprocessing flags:

| Flag | Effect |
|------|--------|
| `--strategy ds1_ds2` | Use the inter-patient DS1/DS2 split from de Chazal et al. (2004) instead of the seeded record-wise split |
| `--records 100 101` | Process a subset |
| `--include-paced` | Keep records 102, 104, 107, 217 (excluded by default per AAMI) |
| `--strict-symbols` | Fail on any annotation symbol missing from the AAMI table |
| `--notch` | Enable the 60 Hz notch (a no-op under the default 40 Hz low-pass) |

Exit status is non-zero if any record fails, even when others succeed.

### Tests

```bash
./venv/bin/python -m pytest              # 633 tests, offline, no dataset needed
./venv/bin/python -m pytest -m requires_data   # 15 more, need MIT-BIH and the MIMIC-III Demo on disk
./venv/bin/python -m pytest tests/mit_bih/test_imbalance_regression.py -m slow   # 4 more; trains real networks, minutes
```

Counts as of 2026-10-15: `tests/mit_bih` 436, `tests/mimic_iii` 109,
`tests/multimodal` 59, `tests/common` 15, `tests/test_scripts.py` 14. Run one
module's tests with `./venv/bin/python -m pytest tests/mimic_iii`. The MIMIC-III
and multimodal tests use an invented MIMIC-shaped database
(`tests/mimic_iii/synthetic.py`), never real patients.

The default suite synthesizes real WFDB records in `tmp_path`, so the reading
path is genuinely exercised without a download. It passes on a fresh clone with
`SMART_HEALTH_DATA` pointing at nothing.

---

## What the pipeline guarantees

These are asserted by tests, not just intended:

1. **No record appears in more than one split.** Beat-level splitting is not
   implemented and no function accepts a split-unit argument — the leakage PDF
   §2.8 warns about is unexpressible rather than merely discouraged.
2. **Normalization statistics never see validation or test data**, and
   `normalization.json` names the records they came from so the claim is auditable.
3. **Filtering is zero-phase** (`filtfilt`), so R-peak sample indices are not
   shifted; the annotated R-peak sits at index 100 of every 300-sample beat.
4. **Leads are selected by name, not column index.** Records 102 and 104 carry no
   MLII; index-based selection would have silently mixed V5 into the dataset.
5. **Beats whose window crosses a record boundary are dropped and counted**, never
   zero-padded. 48 beats across the database are dropped this way.
6. **Non-beat annotations are excluded, never mapped to a beat class.** 3,153
   rhythm, quality and landmark markers are counted separately.
7. **Manifest arithmetic reconciles**: `kept + dropped + excluded + unknown ==
   total annotations`, per record and overall.
8. **Re-running produces identical output** (array contents and all text artifacts).

### Class imbalance is measured, not corrected

`normalization.json` reports per-class counts and suggested inverse-frequency
weights **for the training split only**. No SMOTE, no undersampling. Resampling is
a training decision that belongs next to the loss function; doing it here would
destroy the true class distribution and make honest evaluation impossible.

---

## Results on the real database

48 records, 112,647 annotations, all at 360 Hz. Matches the published figures for
MIT-BIH Arrhythmia Database v1.0.0.

| | Count |
|---|---|
| Beats kept | 109,446 |
| Dropped at record boundary | 48 |
| Excluded non-beat annotations | 3,153 |
| Unknown symbols | 0 |
| **Total annotations** | **112,647** |

Beat annotations = 109,446 + 48 = **109,494**, the documented total.

Per AAMI class (default seeded record-wise split, 31/7/6 records):

| Class | All processed | Train split |
|-------|--------------:|------------:|
| N | 90,589 | 63,184 |
| S | 2,781 | 2,462 |
| V | 7,235 | 6,660 |
| F | 802 | 797 |
| Q | 8,039 | 13 |

Q collapses from 8,039 to 13 between the two columns because almost every Q beat
is a paced beat inside records 102/104/107/217, which AAMI excludes from the
splits. They are still processed and written to disk — excluded is not discarded.

---

## Baseline classifier

A classical baseline, to confirm the preprocessing output is learnable and to
give any future deep model a floor to beat. See [COMMANDS.md](COMMANDS.md) to run it.

Features follow PDF §2.9: six RR-interval features (derived from the saved
`r_peaks`, so the raw recordings are never re-read) plus the beat mean-pooled
from 300 samples to 30. Classifiers are Random Forest and LDA, each wrapped in an
sklearn `Pipeline` so the scaler cannot see test data even by accident.

**Results on DS2 (49,646 beats), Random Forest on RR + morphology:**

| Class | Support | Recall | Precision |
|-------|--------:|-------:|----------:|
| N | 44,196 | 0.904 | 0.959 |
| S | 1,836 | 0.271 | 0.200 |
| V | 3,219 | 0.844 | 0.594 |
| F | 388 | 0.273 | 0.120 |
| Q | 7 | 0.000 | 0.000 |

Macro recall 0.459. Accuracy 0.872, reported but uninformative.

### On the comparison with published work — read this before quoting it

de Chazal et al. (2004) report V sensitivity 0.803 and S sensitivity 0.760 on
this same DS1/DS2 split. **That is a reference range, not a like-for-like
benchmark, and our numbers are not a reproduction of their method.**

No code from that paper exists publicly — it predates code release as a norm —
so nothing of theirs is integrated here. What we borrowed is the DS1/DS2 record
lists in [`src/mit_bih/splits.py`](src/mit_bih/splits.py), i.e. which 44 patients go on
which side of the split. Everything else is our own.

The methods differ in ways that matter:

| | de Chazal 2004 | This project |
|---|---|---|
| ECG leads | two | **one** |
| Morphology features | fixed sampling windows around the beat | 300 samples mean-pooled to 30 |
| RR features | 4 | 6 (two ratio features added) |
| Classifier | LDA, their configuration | Random Forest |

So our V recall of 0.844 against their 0.803 is **not** evidence of a better
method. The defensible reading is narrower and is the one this baseline was
built for: V detection lands in the published range, which indicates the Phase 1
output carries the signal it should. Anything stronger than that is unsupported.

**Ventricular detection is at parity with published work. Supraventricular is
not** — and the reason is worth stating plainly, because it is *not* a data
problem. In the training split, S beats arrive at 0.501 s against 0.845 s for
normal beats, exactly the premature timing that defines the class, and **a single
threshold on `pre_rr_ratio` reaches S recall of 0.972 on DS2.** The signal is
there; this classifier is not exploiting it. The likely cause — 30 morphology
features drowning out 6 RR features under `max_features='sqrt'` — is recorded in
[PlanDoc/IMPLEMENTATION.md](PlanDoc/IMPLEMENTATION.md). The neural classifier
below fixes this by giving RR timing its own input branch.

### The leakage tripwire

Phase 1 z-scores beats using statistics fitted on *its* training records. Training
on a different split than the data was preprocessed for means the test beats were
scaled with statistics that already saw them. `train_baseline.py` refuses to run
in that case and names the command that fixes it, rather than trusting anyone to
remember.

---

## Neural classifier (Phases 1d–1g)

`HybridBeatNet` ([`src/mit_bih/nn/model.py`](src/mit_bih/nn/model.py)) has two input
branches: a 1-D CNN over the 300-sample beat and a small dense branch over the
six RR-interval features. Every model choice is made by 5-fold
cross-validation on DS1, grouped by record so that no patient is on both sides
of a fold. DS2 is scored only after the choice is made.

Training ran on Google Colab (CUDA). See section 4b of
[COMMANDS.md](COMMANDS.md) for the workflow.

### Results on DS2 (49,646 beats)

Headline macro-F1 is averaged over N/S/V/F. Q is excluded because it has 8
training beats and 7 test beats.

| Model | Macro-F1 | S recall | S precision | V recall | V precision |
|---|---:|---:|---:|---:|---:|
| Random Forest baseline (1b) | 0.506 | 0.271 | 0.200 | 0.844 | 0.594 |
| Network, plain cross-entropy (1d) | 0.379 | 0.090 | 0.338 | 0.96 | 0.31 |
| Network, focal loss (1d) | 0.545 | 0.693 | 0.294 | 0.96 | 0.71 |
| 5 × focal ensemble (1e) | 0.590 | 0.736 | 0.388 | — | — |
| **20-model mixed-loss ensemble (1g, selected)** | **0.555** | 0.423 | 0.355 | — | — |
| 5 × focal ensemble, retrained (1g, not selected) | 0.612 | 0.841 | 0.410 | — | — |

All rows are on the same 49,646 beats. **Corrected 2026-10-14:** the baseline
row used to read 0.405, which is its macro-F1 over all five classes (Q included
at 0), not the N/S/V/F value this table uses; on the same footing the baseline
is 0.506. See PlanDoc/IMPLEMENTATION.md, Problem 23.

**Imbalance.** Five treatments were compared in 1d: plain cross-entropy,
class-weighted cross-entropy (weights capped at 10, cap chosen by CV from
10/25/50), focal loss (γ=2), oversampling, and per-class thresholds. Focal loss
won on macro-F1. The RR branch is necessary: removing it (`waveform_only`)
drops focal's macro-F1 from 0.545 to 0.311. It is not sufficient: with the RR
branch, plain cross-entropy (0.379), weighted cross-entropy (0.492) and
oversampling (0.485) still score below the Random Forest baseline (0.506). Only
focal loss, tuned thresholds and the ensembles beat it.

**Architecture (1e).** Rhythm context (a BiLSTM over ±2 neighbouring beats) and
a rolling-median RR reference were tested over 5 folds × 3 seeds. Neither
beat the single-beat focal model consistently. Ensembling the 5 fold models
did help (0.528 → 0.590).

**Calibration (1f, Stage A).** Phase 1f measured threshold-free ranking and
calibration. Focal's win partly comes from a moved threshold rather than better
ranking: on S average precision it ranks 4th of 5 (0.285, against 0.388 for
oversampling). Its probabilities are also inflated: mean predicted P(S) is 3.2×
the true S rate and P(F) is 14.5×. Use these as rankings, not as probabilities,
until Stage B fits a correction.

**Ensemble of losses (1g).** Five ensembles were fixed in advance and the winner
was chosen by pooled out-of-fold macro average precision on DS1. That picked
`D_all_four` (all four losses, 20 models). On DS2 the focal-only ensemble
scored higher (0.612 vs 0.555 macro-F1). **`D_all_four` remains the reported
model.** Switching because of the DS2 score would mean selecting on the test
set.

### What is still unsolved

- **S precision stays between 0.26 and 0.41** across every loss and ensemble.
  The imbalance treatments trade recall for precision along one curve rather
  than shifting it.
- **F is not usable.** Average precision is about 0.02–0.03 on DS2; recall
  above 0.6 comes only with precision of 0.03–0.04.
- **Q cannot be learned** from 8 training beats.
- **Variance between patients is large:** the standard deviation of macro-F1
  across CV folds is about 0.13.

### Reproducibility across devices

The published numbers were produced with torch 2.11.0+cu128 on a Colab GPU.
Locally the project pins torch 2.14.0 on CPU. Probabilities differ slightly
across devices, so a local re-run will not match to the last digit. The 1f
reproduction control re-scored `models/focal.pt` locally and matched the
published macro-F1 to within 0.00004.

---

## ECG inference from a raw signal (`mit_bih.inference`)

The checkpoints in `models/` hold weights only. The z-score statistics the
networks expect live in a processed folder outside the repository, so a
checkpoint on its own cannot classify a new ECG.
[`models/mit_bih_inference_bundle.json`](models/mit_bih_inference_bundle.json)
records everything inference needs: band-pass settings, beat window,
normalization statistics, RR-feature definition, and the 20 member checkpoints
of `D_all_four` with their SHA-256 hashes. It contains no patient data.

`prepare_beats` calls the same functions Phase 1 used (`filters.denoise`,
`segment.segment_beats`, `normalize.apply`, `features.rr_features`), so the
two cannot drift apart. Scored on all of DS2 from the raw WFDB records, the
bundle gives macro-F1 **0.5554**, the published Colab number, with an
identical confusion matrix.

R peaks can be supplied (annotations) or detected (wfdb XQRS, moved to the
local maximum within ±50 ms; less exact than annotations). Signals not sampled
at 360 Hz are rejected, not resampled. **The scores are not probabilities**:
Phase 1f found P(S) inflated 3.2× and P(F) 14.5×, and every output says so.

---

## External test on INCART (`src/incart/`)

Does the model work on patients and equipment it has never seen? The unchanged
`D_all_four` ensemble was scored once on lead II of the open St Petersburg
INCART database: 75 records, 32 patients, recorded at 257 Hz and resampled to
360 Hz. Nothing was tuned on it.

| | Headline macro-F1 (N/S/V/F) | 95% interval (by patient) |
|---|---|---|
| INCART, 175,679 beats | **0.570** | 0.482–0.646 |
| DS2 (MIT-BIH held-out), 49,646 beats | **0.555** | 0.481–0.595 |

- **Verdict.** Under the rule fixed before the run, the model **holds up**: DS2's
  score falls inside INCART's interval.
- **What did not cause it.** Two checks rule out artifacts:
  - Converting DS2 to 257 Hz and back changes its score by +0.0001.
  - INCART's beat marks sit where MIT-BIH's do (median offset 0 ms on both).
- **What differs.** The error pattern is not the same:
  - S beats are found far more often (recall 0.80 against 0.42), but with
    more false alarms (S precision 0.27).
  - 10,061 normal beats are called F. Most of them come from 4 patients.
- **Unsolved.** F remains unusable on both datasets.

Full analysis: [`results/2026-10-31-incart-external-test.md`](results/2026-10-31-incart-external-test.md).

---

## MIMIC-III (Phase 2, `src/mimic_iii/`)

### The task

| | Definition |
|---|---|
| Cohort | Adults (≥ 18) at ICU admission; each patient's first adult ICU stay; the admission has a cardiac ICD-9 code (410–412 MI, 414 CAD, 427 arrhythmia, 428 heart failure); still in the ICU and alive at the index time |
| Index time | `ICUSTAYS.intime` + 24 h. The prediction is made here, from data recorded strictly before it |
| Observation window | Vitals `[intime, index)`; labs `[intime − 6 h, index)` |
| Target | In-hospital mortality, `ADMISSIONS.hospital_expire_flag` |
| Horizon | Index time to hospital discharge. Deaths before the index time are excluded, so every positive is a future event |
| Features | 7 vitals and 11 labs (ITEMIDs from MIT-LCP `mimic-code`, both charting systems): mean, min, max, last, count, and a missing flag each; age, sex, admission type, first care unit. 113 model inputs |
| Never features | Diagnosis codes, discharge fields, length of stay, ICU outtime, date of death: anything known only after the index time |

The values live in [`configs/mimic_iii.toml`](configs/mimic_iii.toml); a test
fails if they drift from the code defaults. **Limitation:** ICD-9 codes are
assigned at discharge. They select the cohort but never feed the model; a
deployed model would need a different trigger for "cardiac patient".

### What the pipeline guarantees (each is a test)

1. **No patient is in two splits.** The split unit is `SUBJECT_ID`, stratified
   on the patient-level label and seeded.
2. **Preprocessing sees only training patients.** Imputation, scaling and
   one-hot encoding are inside one sklearn `Pipeline` with the classifier.
3. **Nothing at or after the index time reaches a feature.** A test rewrites
   every event from each stay's index time onward and asserts the feature
   table is unchanged.
4. **Joins are validated.** A stay whose `subject_id` differs from its
   admission's, or that points at a missing admission, raises an error. A lab
   whose `hadm_id` names a different admission is dropped and counted.
5. **Units are handled, not guessed.** Fahrenheit temperatures are converted
   by ITEMID; a lab with an unexpected unit is dropped, never rescaled.
   Physiologically impossible values are dropped and counted.
6. **The split belongs to its feature table.** `splits.json` stores the table's
   SHA-256, and `train.py` refuses to run if the table changed.
7. **Malformed input fails loudly**, naming the table and column: missing
   tables or columns, unparseable dates, non-numeric values, duplicate keys.
8. **Full MIMIC-III is read in chunks.** `CHARTEVENTS` (~330 M rows) is
   filtered chunk by chunk; a test asserts the chunk size does not change the
   result. Both the Demo's format (lower-case `.csv`) and the full release's
   (upper-case `.csv.gz`) are read.

These were mutation-checked: an inclusive window end, a training split that
contains every row, and a lab join without the admission check each make tests
fail.

### Results on the Demo

> **Pipeline verification only. Not clinical performance.** The MIMIC-III
> Demo has 100 patients, all of whom had died by the time PhysioNet selected
> them. It shows the pipeline runs; it cannot estimate how a model performs.

136 ICU stays → 135 adult → 99 first stays → 66 cardiac → **51 in the ICU and
alive at 24 h, 15 deaths (29%)**. Split by patient 37/7/7. Logistic
regression, C chosen on validation: test AUROC 0.40 on 7 patients, 95% CI
0.00–0.83. Full report:
[`results/2026-10-14-mimic-demo-pipeline.md`](results/2026-10-14-mimic-demo-pipeline.md).

The Phase 2 exploration found 52 stays. The pipeline also excludes a stay
whose hospital discharge precedes its index time (discharged alive 2 minutes
before 24 h; ICU outtime recorded 13 minutes after).

See [report/Phase2_Dataset_Options.pdf](report/Phase2_Dataset_Options.pdf) for
the dataset comparison, including why a Kaggle upload titled "MIMIC-III (Open
Access)" is only the Demo.

| Option | Access | Rough size | Note |
|--------|--------|-----------|------|
| MIMIC-III Clinical Database **Demo** v1.4 | Open, no credentialing | 103 MB | 100 patients. **Downloaded.** Builds and tests pipeline code; cannot support a model. |
| MIMIC-III Clinical Database v1.4 (full) | **Credentialed**: PhysioNet account + CITI human-subjects training + signed data use agreement | several GB gzipped | Approval is a human process taking days to weeks. Once granted, put the CSVs in `~/physionet-data/mimiciii/` and run with `--source full`. |
| MIMIC-III Waveform Matched Subset | Open | terabyte scale | The only source of ECG paired with MIMIC-III patients. Not viable on this machine. |

Account creation, CITI training and the data use agreement are yours to complete —
they are legal commitments that only you can make.

---

## Multimodal (Phase 4, `src/multimodal/`)

`multimodal.inference.run_inference` takes an ECG, a clinical record, or both.
It validates each, runs the matching model, and returns each output in its own
section with the task it answers, the data its model was trained on, and its
caveats. A bad ECG does not stop the clinical output, and the reverse.

**It never combines the two outputs into one score.** The ECG model classifies
heartbeats (MIT-BIH, 1975–79 ambulatory recordings); the clinical model
predicts in-hospital mortality (MIMIC-III ICU stays). They answer different
questions about different people. When both are given, the output states that
patient linkage is not asserted.

`multimodal.fusion.ConcatFusion` is the fusion architecture: each modality is
projected, an absent modality is masked to zero (its input cannot affect the
output, which is tested), presence flags are appended, and training randomly
hides one modality. It learns a synthetic task that needs both inputs.

**Fusion training is BLOCKED.** It needs ECG and clinical data from the *same*
patients at the *same* time, with one target. MIT-BIH and MIMIC-III share no
patients; `fusion.validate_pairs` rejects any MIT-BIH-with-MIMIC-III row as a
fabricated patient. The genuinely paired source (MIMIC-III Waveform Matched
Subset) needs credentialed MIMIC-III and terabyte-scale storage.
`scripts/multimodal/train_fusion.py` reports this and exits with status 3.

## Agents (Part A of the agentic pipeline, `src/agentic/`)

Agents turn the trained models' outputs into findings with evidence, draft rules
connect them, and a local LLM (Ollama, on this machine only) words the report.
A fact-checker rejects any sentence the findings do not back; the plain template
report is shown instead. Design: `PlanDoc/agentic_with_fe.md`; build plan:
`PlanDoc/plans/2026-10-15-agents-end-to-end.md`.

```bash
python scripts/agentic/report.py --list
python scripts/agentic/report.py --ecg-record 208 --seconds 20 --stay <icustay_id>
python scripts/agentic/report.py --ecg-record 208 --stay <icustay_id> --ask "What is a V beat?"
python scripts/agentic/report.py --ecg-record 208 --stay <icustay_id> --llm off   # template only
```

An MIT-BIH record and a MIMIC-III stay are different people: such a case is a
**simulated case** and says so. The AI never sees the cardiologist's beat labels
or the patient's outcome. Rules and glossary are drafts awaiting review.
Research demo, not for patient care.

### The web page (`frontend/`)

```bash
python frontend/server.py --open     # http://127.0.0.1:8765
```

Pick a recording and an ICU stay, press Run, and watch the nine stages fill
in; then chat about the case. Listens on 127.0.0.1 only and loads nothing from
the internet. See [frontend/README.md](frontend/README.md).

---

## Running everything

```bash
./venv/bin/python scripts/validate_setup.py       # what is present, missing, blocked
./venv/bin/python scripts/run_end_to_end.py       # all eight steps, ~1.5 min
```

`run_end_to_end.py` runs, in order: setup validation; MIMIC-III features,
patient split, training and evaluation; MIT-BIH preprocessing into its own
folder (published data is never overwritten); ECG model evaluation; a shared
inference smoke test; and fusion training, which reports BLOCKED. It writes
`$SMART_HEALTH_DATA/processed/end_to_end/report.md`. Nothing in it trains a
neural network. See [COMMANDS.md](COMMANDS.md) for each step on its own.

## Limitations

- **No clinical validation of anything.** The ECG model is evaluated on 22
  recordings from the 1970s; the clinical pipeline has only run on 51 Demo stays.
- **One DS2 test patient is also in training.** Records 201 (DS1) and 202
  (DS2) come from the same person (202's header: "taken from the same analog
  tape as record 201"). This is part of the standard de Chazal DS1/DS2 split
  used for comparability with published work; see PlanDoc/IMPLEMENTATION.md,
  Problem 24.
- **ECG scores are uncalibrated** (Phase 1f Stage B not started).
- **ECG model input:** single lead, 360 Hz only, beats centred on R peaks.
  Detected R peaks are less exact than the annotations it was trained on.
- **S precision 0.36, F unusable** on DS2 (see "What is still unsolved").
- **Clinical model untested at scale:** chunked ingestion is tested on
  synthetic data; it has not been run on full MIMIC-III.
- **Cohort selection uses discharge diagnoses** (see "The task").
- **No paired data, so no fusion model.**

## Reproducibility

Seeds: MIT-BIH splits `20260826`, network training `20260916`, MIMIC-III split
`20261014` (in `configs/mimic_iii.toml`). Every model and report records the
git commit, whether the tree was dirty, library versions and the SHA-256 of
its inputs. The ECG bundle and the MIMIC model are refused if their files
change after being written. One known non-reproducibility: the saved MIT-BIH
beats differ from a re-run of the pipeline today by at most one float32 step
(4.8e-7); results are unaffected (see
[IMPLEMENTATION.md](PlanDoc/IMPLEMENTATION.md), Problem 18).

---

## Layout

```
src/
├── common/         paths, provenance (git, versions, hashes), binary metrics
├── mit_bih/        ECG: everything MIT-BIH (detail below)
├── incart/         external test on INCART: config, io (load, download, patient grouping), resample, evaluate
├── mimic_iii/      clinical: MIMIC-III pipeline (detail below)
├── multimodal/     shared inference + fusion (detail below)
└── agentic/        findings, heart/chart agents, rules, LLM client, fact-checker, report + chat agents, bake-off
configs/
├── mimic_iii.toml  the MIMIC-III task: cohort, windows, target, split, model grid
├── agent_rules.toml  review ranges, drug groups, 12 draft rules (each with a source)
└── glossary.toml   22 draft definitions the chat may cite
scripts/
├── validate_setup.py     environment, data and blocker check
├── run_end_to_end.py     every step in order, with a report
├── make_integration_figures.py   figures for PlanDoc/Integration_Documentation.md
├── mit_bih/        command-line entry points for MIT-BIH (detail below)
├── incart/         download_incart.py, resampling_check.py (DS2 360 -> 257 -> 360 Hz), evaluate_on_incart.py
├── mimic_iii/      build_features.py, make_splits.py, train.py, show_predictions.py; explore_demo.py (the original spike)
├── multimodal/     infer.py (shared inference), train_fusion.py (BLOCKED gate), demo_fusion_synthetic.py
└── agentic/        report.py (run a case, ask questions), compare_llms.py (model bake-off)
frontend/           the web page: server.py (127.0.0.1 only), app.py, serialize.py, outcome.py, static/
tests/
├── mit_bih/        436 tests; conftest.py (fixtures) + synthetic.py (fake ECG/WFDB)
├── incart/         49 tests (1 more needs INCART); synthetic.py writes fake 12-lead 257 Hz records
├── mimic_iii/      113 tests (4 need the Demo); synthetic.py builds a fake MIMIC-III database
├── multimodal/     60 tests (1 needs MIT-BIH)
├── agentic/        126 tests (2 need the data, 2 need Ollama); fakes.py holds a fake LLM
├── frontend/       35 tests (a real server on a free port, synthetic data)
├── common/         15 tests
└── test_scripts.py every new script starts; the setup check is honest
notebooks/
└── mit_bih/        Colab notebooks for Phases 1d, 1e, 1g
models/             trained MIT-BIH checkpoints (.pt) + mit_bih_inference_bundle.json
results/            frozen run snapshots and their analysis
report/             written reports (.docx, .pdf) and their figures
docs/               the generated GitHub Pages site (MIT-BIH baseline)
site/               templates the results page is built from
PlanDoc/
├── PLAN.md              the approved plan
├── IMPLEMENTATION.md    what was actually done, deviations, test output
├── healthcare_ai_implementation_prompt.md   the 2026-10-14 request
├── Integration_Documentation.md (.doc)      professor-ready end-to-end report
├── Professor_Presentation_Notes.doc         speaking script, Q&A, demo guide
├── Implementation_Status.doc                completed / partial / pending checklist
├── integration_results/                     report figures and verification outputs
├── specs/               design documents for 1d, 1f, 1g, Phase 2/4, and the INCART external test
└── plans/               implementation plans (1d, agents, frontend, INCART external test)
PERMISSIONS.md           permission policy for this project
COMMANDS.md              copy-paste command reference
```

### `src/mit_bih/` in detail

```
src/mit_bih/
├── config.py      constants: fs, band edges, window sizes, class order, paths
├── errors.py      typed exceptions, so failures are diagnosable
├── io.py          WFDB loading, lead selection by name
├── filters.py     zero-phase bandpass, optional notch
├── normalize.py   z-score with separate fit/apply
├── labels.py      MIT-BIH symbol -> AAMI 5-class mapping
├── segment.py     beat-level and window-level segmentation
├── splits.py      record-wise and DS1/DS2 splits
├── acquire.py     download + integrity checking
├── pipeline.py    orchestration, manifest generation
├── features.py    RR-interval + morphology features (baseline)
├── baseline.py    Random Forest / LDA pipelines, leakage tripwire
├── evaluate.py    per-class metrics, confusion matrix
├── dataset.py     pairs each beat with its RR features (network input)
├── folds.py       record-grouped 5-fold CV over DS1
├── sequence.py    beat neighbourhoods for the rhythm-context model
├── thresholds.py  per-class decision thresholds (tuned on DS1 only)
├── ensemble.py    averaging model predictions
├── stats.py       paired comparisons and confidence intervals
├── ranking.py     threshold-free ranking (average precision)
├── calibration.py Brier, ECE, reliability, prior drift
├── plots.py       calibration figures
├── inference.py   inference bundle; raw signal -> beats -> ensemble scores
└── nn/
    ├── model.py      HybridBeatNet: CNN on the beat + dense branch on RR
    ├── seqmodel.py   rhythm-context model (CNN per beat + BiLSTM)
    ├── losses.py     plain / weighted / focal losses
    ├── train.py      training loop and CV for the 1d arms
    ├── seqtrain.py   training loop for the rhythm-context model
    └── checkpoint.py saving and reloading weights
scripts/mit_bih/
├── download_mitbih.py       preprocess_mitbih.py      train_baseline.py
├── train_nn.py (1d)         train_nn_v2.py (1e)       train_ensemble_mix.py (1g)
├── diagnose_calibration.py (1f)
├── export_inference_bundle.py   evaluate_bundle.py   record_learning_curve.py
├── package_for_colab.py     ingest_colab_results.py
└── build_report.py          make_report_figures.py    make_report_docx.py
```

Until 2026-10-14 the `mit_bih` package was called `ecg` and everything sat in
flat `scripts/`, `tests/` and `notebooks/` folders. Older documents in
`PlanDoc/` use the old paths.

The 2026-10-14 request sketched `mit_bih` as `preprocessing / dataset / model /
train / evaluate`. Those roles already existed under other names and were not
renamed:

| Role | Module |
|---|---|
| preprocessing | `io`, `filters`, `segment`, `labels`, `normalize`, `pipeline` |
| dataset + splits | `dataset`, `splits`, `folds` |
| model | `nn/model.py` (`HybridBeatNet`), `nn/checkpoint.py` |
| train | `nn/train.py`, `nn/losses.py` (training ran on Colab) |
| evaluate | `evaluate`, `ranking`, `calibration`, `stats` |
| inference | `inference` |

### `src/mimic_iii/` and `src/multimodal/` in detail

```
src/mimic_iii/
├── config.py        the task (TaskSpec), variables + ITEMIDs + units + ranges, post-index columns
├── errors.py        DataSourceMissingError, SchemaError, IntegrityError, LeakageError
├── ingestion.py     case-insensitive .csv/.csv.gz readers, schema checks, chunked event reads
├── cohort.py        cohort flow, validated joins, age (>89 handling), index time, label
├── features.py      windowed vitals/labs, unit conversion, aggregation, missing flags, leakage guard
├── dataset.py       patient-level split (SUBJECT_ID), split tripwire, X/y
├── preprocessing.py imputation + scaling + one-hot + classifier in one Pipeline
├── train.py         C and threshold chosen on validation; hashed model save/load
└── evaluate.py      metrics + bootstrap CIs + reliability; report with Demo banner
src/multimodal/
├── ecg_encoder.py      ECG -> per-beat scores + 13-value named summary
├── clinical_encoder.py one stay's features -> vector + mortality score; rejects post-index fields
├── inference.py        run_inference: either or both modalities, labelled, never merged
├── fusion.py           ConcatFusion, train_fusion, validate_pairs, save/load
└── errors.py
```

## Data licence

MIT-BIH Arrhythmia Database v1.0.0 is openly available on PhysioNet — no account
or data use agreement required. It is de-identified clinical data, which is not
the same as public: it stays on this machine.

The MIMIC-III Demo v1.4 is open under the Open Database License. Full MIMIC-III
requires PhysioNet credentialing and a signed data use agreement; under it, no
row-level data may leave this machine or be shared.
