# MIMIC-III clinical model: in-hospital mortality

> **PIPELINE VERIFICATION ONLY - NOT CLINICAL PERFORMANCE. These numbers come from the MIMIC-III Demo (100 patients, all of whom had died by the time PhysioNet selected them). The test split has a handful of patients; the confidence intervals show how little the point estimates mean. Do not quote them as model performance.**

**Data source:** `demo` - MIMIC-III Clinical Database Demo v1.4: 100 patients, all of whom had died by the time PhysioNet selected them. Verifies the pipeline; cannot estimate clinical performance.

## Task

- Target: ADMISSIONS.hospital_expire_flag (death before hospital discharge)
- Index time: ICUSTAYS.intime + 24 h
- Vitals window: [intime, intime + 24 h); labs window: [intime - 6 h, intime + 24 h)
- Horizon: index time to hospital discharge
- Exclusions: age < 18, not the patient's first adult ICU stay, no cardiac ICD-9 code on the admission, left the ICU before the index time, died before the index time

## Cohort

| Step | Stays | Patients |
|---|---:|---:|
| All ICU stays | 136 | 100 |
| Adults (age >= 18) | 135 | 99 |
| First adult ICU stay per patient | 99 | 99 |
| Cardiac ICD-9 code on the admission | 66 | 66 |
| In the ICU and alive at intime + 24 h | 51 | 51 |

Final cohort: 51 stays, 15 deaths (29%).

## Split (by patient)

| Split | Patients | Stays | Deaths |
|---|---:|---:|---:|
| train | 37 | 37 | 11 |
| val | 7 | 7 | 2 |
| test | 7 | 7 | 2 |

## Model selection (validation split only)

- Selected C = 1.0 (C chosen by validation AUPRC; ties to the smaller C)
- Threshold = 0.701 (youden on validation)

## Metrics

`train` is in-sample. `val` chose C and the threshold, so it is not held out either. Only `test` was untouched until this evaluation.

| Split | n | Deaths | AUROC (95% CI) | AUPRC (95% CI) | Brier | Sens | Spec | PPV |
|---|---:|---:|---|---|---:|---:|---:|---:|
| train | 37 | 11 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.006 | 1.000 | 1.000 | 1.000 |
| val | 7 | 2 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.123 | 1.000 | 1.000 | 1.000 |
| test | 7 | 2 | 0.400 (0.00-0.83) | 0.333 (0.14-0.92) | 0.367 | 0.000 | 0.800 | 0.000 |

Undefined metrics (not computable on these rows):

- test f1: precision or recall undefined or both zero

## Largest standardized coefficients

| Feature | Coefficient |
|---|---:|
| `glucose_min` | +0.537 |
| `dbp_min` | -0.527 |
| `spo2_last` | -0.484 |
| `sodium_count` | +0.474 |
| `creatinine_count` | +0.463 |
| `resp_rate_min` | -0.461 |
| `mbp_count` | -0.428 |
| `temp_c_last` | +0.424 |
| `resp_rate_mean` | -0.406 |
| `heart_rate_count` | -0.400 |
| `temp_c_mean` | +0.397 |
| `sbp_count` | -0.394 |
| `dbp_count` | -0.393 |
| `glucose_mean` | +0.373 |
| `hemoglobin_max` | -0.334 |

## Provenance

- Generated: 2026-10-15T04:23:15+00:00
- Git commit: 156d81ac29cdde9a876e414b0ba7c1888b68bd31 (dirty: True)
- Environment: {'python': '3.13.9', 'numpy': '2.3.3', 'pandas': '2.3.3', 'sklearn': '1.7.2', 'scipy': '1.16.2', 'torch': '2.14.0'}

_Research artifact. Not clinically validated, not a medical device, not for patient care._
