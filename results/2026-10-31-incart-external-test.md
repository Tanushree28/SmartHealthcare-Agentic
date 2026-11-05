# External test on INCART, 2026-10-31

> **Research result on a public dataset; not clinical validation and not a
> medical device.** The model's scores are uncalibrated: Phase 1f measured
> mean predicted P(S) at 3.2× and P(F) at 14.5× the true rates. Treat them as a
> ranking of beats, not as probabilities.

**Question.** Does the MIT-BIH beat model work on patients and equipment it
has never seen?

**Design:** [`PlanDoc/specs/2026-10-29-incart-external-test-design.md`](../PlanDoc/specs/2026-10-29-incart-external-test-design.md).

**Run files:** [`2026-10-31-incart-external-test.json`](2026-10-31-incart-external-test.json)
and [`2026-10-31-incart-resampling-check.json`](2026-10-31-incart-resampling-check.json).

## What was run

| | |
|---|---|
| **Model** | `D_all_four`: the 20-checkpoint MIT-BIH ensemble, unchanged |
| **Prediction** | Predicted class = argmax |
| **Data** | St Petersburg INCART 12-lead Arrhythmia Database v1.0.0: 75 records of 30 minutes, 32 patients |
| **Lead** | Lead II only |
| **Sampling rate** | Recorded at 257 Hz, resampled to 360 Hz before the unchanged MIT-BIH preprocessing |
| **Beats** | 175,907 annotated, 175,679 classified. Of the rest, 7 fell past the end after resampling, 71 had a window crossing the record edge, and 150 had no neighbouring beat for RR features. The counts reconcile. |
| **Excluded records** | None: no record has paced beats |
| **Unknown symbols** | None |
| **Intervals** | 95% percentile bootstrap, 1,000 draws, seed 20260916, resampling the **32 patients** |
| **Patient grouping** | Taken from INCART's `files-patients-diagnoses.txt`: all 75 records appear exactly once, in 32 groups |

INCART was scored **once**, and nothing was tuned on it. A two-record trial run
beforehand printed beat counts and timing only, no metrics.

## The two rules fixed before the run

| Rule (from the design) | Result | Verdict |
|---|---|---|
| "The model holds up on INCART" means the DS2 headline macro-F1 falls inside INCART's 95% interval | DS2 0.5554 against INCART [0.4821, 0.6463] | **Holds up** |
| "Resampling matters" means 360 → 257 → 360 Hz moves the DS2 headline by more than 0.01 | 0.5554 → 0.5555 (+0.0001) | **Does not matter** |

The resampling check also reproduces the published DS2 number (0.5554) from
the raw records, so the reference INCART is compared with is the published
one.

## INCART and DS2 side by side

Headline macro-F1, averaged over N/S/V/F (Q excluded as everywhere else):

| | Headline macro-F1 | 95% interval | Beats | Patients |
|---|---|---|---|---|
| **INCART** (lead II) | **0.570** | 0.482–0.646 | 175,679 | 32 |
| **DS2** (MIT-BIH held-out) | **0.555** | 0.481–0.595 | 49,646 | 22 |

Per class:

| Class | INCART beats | INCART recall | INCART precision | INCART F1 | DS2 beats | DS2 recall | DS2 precision | DS2 F1 |
|---|---|---|---|---|---|---|---|---|
| N | 153,473 | 0.911 | 0.995 | **0.951** | 44,196 | 0.919 | 0.967 | **0.942** |
| S | 1,990 | 0.802 | 0.271 | **0.405** | 1,836 | 0.423 | 0.355 | **0.386** |
| V | 19,991 | 0.874 | 0.945 | **0.908** | 3,219 | 0.943 | 0.845 | **0.891** |
| F | 219 | 0.384 | 0.008 | **0.015** | 388 | 0.005 | 0.001 | **0.002** |
| Q | 6 | 0 | 0 | 0 | 7 | 0 | 0 | 0 |

INCART confusion matrix (rows = true class, columns = predicted class):

| | N | S | V | F | Q |
|---|---|---|---|---|---|
| **N** | 139,815 | 2,744 | 849 | 10,061 | 4 |
| **S** | 203 | 1,595 | 161 | 31 | 0 |
| **V** | 349 | 1,544 | 17,471 | 626 | 1 |
| **F** | 109 | 12 | 14 | 84 | 0 |
| **Q** | 3 | 0 | 3 | 0 | 0 |

## What it means

- **On the headline, the model transfers.** The macro-F1 on 32 new patients,
  recorded with different equipment, is as good as on MIT-BIH's own held-out
  patients. Normal (N) and ventricular (V) beats score F1 of about 0.9 or
  higher on both datasets.

- **"Holds up" means "no detectable drop", not "the same".** INCART's
  interval is 0.16 wide. A drop of 0.05 could still hide inside it. The rule
  tests for a large drop; it cannot certify equal performance.

- **The pattern of errors is different, even though the headline is similar.**
  - On INCART the model finds **80% of S beats** (DS2: 42%). It also calls
    many non-S beats S, notably 1,544 V beats and 2,744 N beats, so S
    precision falls to 0.27.
  - The largest error is **10,061 normal beats called F** (6.6% of INCART's N
    beats). This is why F precision is 0.008.
  - These F errors are **concentrated**: 8 of the 75 records produce 87% of
    them. Two patients (3 records each) account for 69%.
  - That points to the beat shape of a few patients, not to a general shift.
    This run does not identify the cause.

- **The F class stays unusable on both datasets** (F1 0.015 and 0.002).
  F is the class the project already lists as unsolved.

- **Per-record macro-F1 is lower than the pooled number** (median 0.47, range
  0.19–0.67, in the JSON's `per_record` table). Most records contain few or no
  S and F beats, and an absent class counts as F1 0 in a record's average. The
  per-record numbers show where the errors are; they are not an estimate of
  the model's performance.

## The two checks behind any drop

**Beat positions.**
- **Method:** for each annotated beat, the offset from the annotation mark to
  the largest band-passed value within ±50 ms.
- **INCART** (after resampling): median 0.0 ms, interquartile range −5.6 to
  +5.6 ms.
- **DS2:** median 0.0 ms, interquartile range 0.0 to +2.8 ms.
- **Verdict:** the medians are identical, so the rule (a difference of more
  than 3 samples, 8.3 ms) did not fire and no moved-mark variant was scored.
  INCART's marks are a little more spread out, by about ±2 samples at 360 Hz.

**Amplitude (diagnostic only).** The median peak of the band-passed beat is
1.25 mV on INCART lead II and 1.46 mV on DS2, about 15% smaller. Nothing was
rescaled: correcting it would be a preprocessing change chosen after looking
at INCART. A smaller scale is one possible contributor to the N → F errors.
This run does not test that.

**Resampling.** Sending DS2 through 257 Hz and back changes the headline by
+0.0001, and per-class F1 by at most 0.006 (V: 0.891 → 0.885). The sampling
rate is not what separates INCART from DS2.

## Limitations

- **One lead.** Lead II, the closest of the 12 to MIT-BIH's MLII; the other 11 were not used.
- **Annotated beat positions were used, not detected ones.** A deployed system
  would have to find the beats itself, and that usually lowers scores.
- **The patients are different from MIT-BIH's, but not representative.**
  INCART's patients were Holter patients tested for coronary artery disease,
  and records were chosen for arrhythmias and ischemia.
- **The scores are uncalibrated** (see the banner). INCART does not change that.
- **The run's provenance** records a dirty working tree: the `incart` module was
  not yet committed when it ran.
- **Run time.** The resampling check took 6,088 s of wall time because the
  Mac slept twice mid-run; the computation itself took about 12 minutes.
  Sleep pauses a process and does not change its numbers. The INCART run took
  1,488 s, with idle sleep prevented.
