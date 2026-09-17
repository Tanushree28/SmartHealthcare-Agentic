# results/

Frozen snapshots of model runs, with the analysis that was written at the time.

This directory is **append-only and hand-curated**. It is the opposite of
`docs/`, which is regenerated from the data by `scripts/build_report.py` and can
be thrown away at any time. Nothing here is ever regenerated: a file lands once
and stays as it was.

## Why it exists

`scripts/train_baseline.py` writes `baseline_metrics.json` next to the processed
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
| [2026-08-27](2026-08-27-baseline-ds1_ds2.md) | Classical baseline (Random Forest, LDA) | DS1/DS2 inter-patient | RF on RR+morphology, macro recall 0.459 |

## What is and is not in these files

Aggregate metrics only: confusion matrices, per-class recall/precision/F1, and
the record IDs of each split. No per-beat data, no signals, no annotations. The
`.gitignore` ban on clinical data is not in tension with committing these.

## The `run` block

Each `.json` is the file `train_baseline.py` produced, unchanged, with one
top-level `"run"` key added: date, nearest commit, the exact commands, package
versions, and the model hyperparameters. Where provenance is uncertain it says
so rather than implying a precision we do not have.
