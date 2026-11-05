# results/

Frozen snapshots of model runs, with the analysis that was written at the time.

This directory is **append-only and hand-curated**. It is the opposite of
`docs/`, which is regenerated from the data by `scripts/mit_bih/build_report.py` and can
be thrown away at any time. Nothing here is ever regenerated: a file lands once
and stays as it was.

## Why it exists

`scripts/mit_bih/train_baseline.py` writes `baseline_metrics.json` next to the processed
data, outside the repo (`$SMART_HEALTH_DATA`), where the next run overwrites it.
That is fine for iterating and useless for comparison. A deep model that claims
to have beaten the classical baseline has to be measured against a specific set
of numbers produced on a specific split — and those numbers have to still exist
when the claim is made.

## Naming

```
YYYY-MM-DD-<model-family>-<split>.json    the run, verbatim, plus a "run" block
YYYY-MM-DD-<model-family>-<split>.md      the analysis
```

The date is the date of the **run**, not of the commit.

## Contents

| Run | Model family | Split | Headline |
| --- | --- | --- | --- |
| [2026-08-27](2026-08-27-baseline-ds1_ds2.md) | Classical baseline (Random Forest, LDA) | DS1/DS2 inter-patient | RF on RR+morphology, macro recall 0.459, macro-F1 0.405 |
| [2026-09-16](2026-09-16-nn-ds1_ds2.json) | Phase 1d: hybrid network, 5 imbalance treatments + ablation | DS1/DS2, 5-fold CV on DS1 | Focal loss wins, headline macro-F1 0.545; no RR branch drops it to 0.311 |
| [2026-09-17](2026-09-17-nn-v2-ds1_ds2.json) | Phase 1e: rhythm context, robust RR, 3 seeds, fold ensembles | DS1/DS2, 5 folds × 3 seeds | No architecture change wins consistently; 5-model focal ensemble 0.590 |
| [2026-09-17](2026-09-17-calibration-diagnosis-ds1_ds2.json) | Phase 1f Stage A: ranking + calibration, nothing fitted | DS1 out-of-fold + DS2 | Focal ranks S 4th of 5 by average precision; probabilities inflated (S 3.2×, F 14.5×) |
| [2026-09-17](2026-09-17-ensemble-mix-ds1_ds2.json) | Phase 1g: ensembles mixing four losses | DS1/DS2, selected on DS1 out-of-fold | Selected `D_all_four`: DS2 macro-F1 0.555. The unselected focal-only ensemble scored 0.612 |
| [2026-10-14](2026-10-14-bundle-eval-ds1_ds2.json) | `D_all_four` re-scored on CPU through the inference bundle, from the saved beats | DS2 (49,646 beats) | Macro-F1 0.5554, identical to the Colab result; same confusion matrix |
| [2026-10-14](2026-10-14-bundle-eval-raw-ds1_ds2.json) | The same, with beats prepared from the raw WFDB records by `mit_bih.inference` | DS2 (49,646 beats) | Macro-F1 0.5554, identical: the saved inference preprocessing is the training preprocessing |
| [2026-10-14](2026-10-14-mimic-demo-pipeline.md) | MIMIC-III clinical pipeline, logistic regression | MIMIC-III **Demo**, by patient 37/7/7 | **Pipeline verification only.** Test AUROC 0.40 on 7 patients, 95% CI 0.00-0.83. Not clinical performance |
| [2026-10-31](2026-10-31-incart-resampling-check.json) | `D_all_four` on DS2 as recorded vs 360 → 257 → 360 Hz | DS2 (49,646 beats) | Original 0.5554 (reproduces); round trip 0.5555: resampling does not matter |
| [2026-10-31](2026-10-31-incart-external-test.md) | `D_all_four`, unchanged, external test | INCART lead II (175,679 beats, 32 patients) | Macro-F1 **0.570**, 95% CI 0.482-0.646 (patient bootstrap); DS2's 0.555 inside: holds up. S recall 0.80; 10,061 N beats called F, mostly from 4 patients |

The 2026-10-14 MIMIC-III row is from the 100-patient Demo, every one of whom
had died by the time PhysioNet selected them. Its `.md` carries that banner;
the numbers show the pipeline runs, nothing more.

Only the baseline and the MIMIC-III Demo run have a written analysis `.md`. The analysis for 1d–1g is in
[`PlanDoc/IMPLEMENTATION.md`](../PlanDoc/IMPLEMENTATION.md) (section "Phases
1d–1g") and the design specs in [`PlanDoc/specs/`](../PlanDoc/specs/). The
`*-run.log` files are the verbatim console output of each Colab run.

**Device:** 1d, 1e and 1g ran on Colab (CUDA, torch 2.11.0+cu128). Local runs use
torch 2.14.0 on CPU, so re-scored probabilities differ slightly. 1f's
reproduction control matched the published 1d focal macro-F1 to within 0.00004.

## What is and is not in these files

Aggregate metrics only: confusion matrices, per-class recall/precision/F1, and
the record IDs of each split. No per-beat data, no signals, no annotations. The
`.gitignore` ban on clinical data is not in tension with committing these.

## The `run` block

Each `.json` is the file `train_baseline.py` produced, unchanged, with one
top-level `"run"` key added: date, nearest commit, the exact commands, package
versions, and the model hyperparameters. Where provenance is uncertain it says
so rather than implying a precision we do not have.
