# Design: MIMIC-III pipeline, ECG inference bundle, multimodal layer

Date: 2026-10-14. Requested in `PlanDoc/healthcare_ai_implementation_prompt.md`
("implement, test and document everything feasible without waiting for
repeated approval"). This file records the decisions taken under that
instruction, so each can be reviewed and reversed.

## 1. Layout: adapt, do not rewrite

The prompt sketches `src/mit_bih/{preprocessing,dataset,model,train,evaluate}.py`.
The existing `mit_bih` package already covers each role under other names
(`pipeline`/`filters`/`segment`/`normalize`, `dataset`, `nn/model`, `nn/train`,
`evaluate`), behind 414 passing tests and published results. Renaming them
would change nothing but import lines, so the layout is kept and the mapping
is documented in the README. New code:

| Package | Files |
|---|---|
| `src/common/` | `paths`, `provenance`, `metrics` (binary) -- only what two modules use |
| `src/mimic_iii/` | `config`, `errors`, `ingestion`, `cohort`, `features`, `dataset`, `preprocessing`, `train`, `evaluate` |
| `src/mit_bih/` | `inference` (new): inference bundle, raw signal -> beats -> scores |
| `src/multimodal/` | `ecg_encoder`, `clinical_encoder`, `fusion`, `inference`, `errors` |
| `configs/` | `mimic_iii.toml` (task + model settings) |

`cohort.py` is added beyond the prompt's list because cohort selection is its
own concern with its own tests. `mit_bih.config.data_root` is left alone;
`common.paths` duplicates it and a test asserts they agree, so no working code
had to change.

## 2. MIMIC-III task definition

Kept from the Phase 2 exploration (`scripts/mimic_iii/explore_demo.py`), now
written down as the contract:

| | Decision | Why |
|---|---|---|
| Cohort | Adults (>= 18), first adult ICU stay, cardiac ICD-9 (410-412, 414, 427, 428) on the admission, in the ICU and alive at the index time, hospital discharge not before the index time | Cardiac focus of the project; one stay per patient avoids within-patient correlation |
| Index time | `intime + 24 h` | Prediction is made once a day of ICU data exists |
| Observation window | Vitals `[intime, index)`; labs `[intime - 6 h, index)` | First labs are often drawn in the ED before transfer; same convention as `mimic-code` first-day concepts |
| Target | `hospital_expire_flag` | In-hospital mortality, as in the exploration |
| Horizon | Index time to hospital discharge | Deaths before the index time are excluded, so every positive is a future event |
| Variables | 7 vitals, 11 labs (ITEMIDs from `mimic-code`); mean/min/max/last/count + missing flag; age, sex, admission type, first care unit | Known at the index time |
| Not used as features | ICD codes, discharge fields, LOS, outtime, `last_careunit`, DOD | Assigned or known only after the index time (`config.POST_INDEX_COLUMNS`) |

**Known limitation.** ICD-9 codes are assigned at discharge. They select the
cohort but are never features; a deployed model would need a different
trigger for "cardiac patient".

**Exploration vs pipeline: 52 -> 51 stays.** The pipeline also excludes stays
whose hospital discharge precedes the index time. On the Demo one stay
(subject 42458) was discharged alive 2 minutes before `intime + 24 h` while its
ICU `outtime` is 13 minutes after it -- a timestamp inconsistency. Nothing is
left to predict for it.

## 3. Leakage controls (each has a test)

- Split by `SUBJECT_ID`, stratified on the patient-level label, seeded.
- Imputation, scaling and one-hot encoding live in one sklearn `Pipeline` with
  the classifier, fitted on training patients only.
- `C` and the decision threshold are chosen on validation; test is scored once.
- `splits.json` stores the SHA-256 of the feature table; `train.py` refuses a
  changed table (the MIMIC equivalent of the MIT-BIH normalization tripwire).
- Every window is half-open; a test rewrites every event at/after each stay's
  index time and asserts the feature table does not change.
- Labs joined by subject + window; a lab whose `hadm_id` names another
  admission is dropped. Chart rows must match the stay's `subject_id`.
- Mutation-checked: making the window inclusive, leaking all rows into
  training, or removing the admission check each makes tests fail.

## 4. Memory-safe ingestion

`CHARTEVENTS`/`LABEVENTS` are read in chunks (default 1 M rows), cut to wanted
ITEMIDs and cohort keys per chunk. Both release formats are accepted
(Demo: lower-case `.csv`; full: upper-case `.csv.gz`). A test asserts chunk
size does not change the result. Not run on full MIMIC-III (not available).

## 5. ECG inference bundle

The checkpoints hold weights only; the z-score statistics they need live in a
processed folder outside the repository. `models/mit_bih_inference_bundle.json`
records the preprocessing settings, the normalization statistics, and the 20
member checkpoints of `D_all_four` (the Phase 1g selection) with SHA-256
hashes. `prepare_beats` calls the Phase 1 functions themselves, so it cannot
drift from them. Verified on all of DS2 from the raw WFDB records: macro-F1
0.5554, identical to the published Colab number.

R peaks may be supplied (annotations) or detected (wfdb XQRS, refined to the
local maximum within +/-50 ms). Signals at another sampling rate are rejected,
not resampled.

**Scores are uncalibrated** (Phase 1f: P(S) 3.2x, P(F) 14.5x inflated). Every
output says so. Phase 1f Stage B remains not started.

## 6. Multimodal

| Stage | State |
|---|---|
| Independent baselines | ECG: `D_all_four`, evaluated on DS2. Clinical: logistic regression, runs on the Demo only |
| Shared inference interface | `multimodal.inference.run_inference`: ECG, clinical or both; per-modality validation; outputs kept separate and labelled with task, training data and caveats |
| Fusion architecture | `ConcatFusion`: per-modality projection, absent modality masked to zero, presence flags, modality dropout. Tested on synthetic pairs |
| Real multimodal training | **BLOCKED** |

**Why blocked.** MIT-BIH (48 recordings from 47 patients, 1975-79) and MIMIC-III (ICU 2001-12)
share no patients, and the two models predict different targets (beat class
vs in-hospital mortality). Pairing their records would fabricate patients;
`fusion.validate_pairs` rejects any `mitbih` + `mimiciii` row. A genuinely
paired source exists -- the MIMIC-III Waveform Database Matched Subset, linked
by `subject_id` -- but needs (a) credentialed MIMIC-III and (b)
terabyte-scale storage. With it, the work left is: extract ECG lead II segments
inside each stay's observation window, resample 125 Hz -> 360 Hz (the ECG model
rejects other rates) or retrain at 125 Hz, encode, build a pairs manifest, and
train with `fusion.train_fusion`.

**Never combined.** `run_inference` has no code path that merges the two
independent outputs into one number. A fused score is produced only by a
fusion model; a model trained on synthetic data is refused unless explicitly
allowed, and then labelled "not a risk estimate".

## 7. Not done, deliberately

- No new dependencies (`tomllib` is stdlib; models are saved with `pickle`
  after a SHA-256 check, not `joblib`, which is not pinned).
- No neural network retrained locally; no long jobs launched.
- No commits (PERMISSIONS.md: commits need approval).
