# Classical baseline on DS1/DS2 — 2026-08-27

Run data: [`2026-08-27-baseline-ds1_ds2.json`](2026-08-27-baseline-ds1_ds2.json)

Six models: two classifiers (Random Forest, LDA) across three feature sets
(RR + morphology, RR only, morphology only). Inter-patient DS1/DS2 split —
18 training records, 22 test records, no patient in both. Four paced records
excluded. ~49.6k test beats.

This is a **floor**, not a result to be proud of. Its job is to be the number a
later deep model has to beat before it can claim to have learned anything, and
to confirm the Phase 1 output is learnable at all. It does both.

## The full table

Macro F1 is computed here for the analysis; it is not in the JSON, because
`evaluate.py` does not currently report it. Both averages are over all five
classes — `evaluate.py` averages over classes with non-zero support, and Q
scrapes in with 7 beats, which is itself part of the problem discussed below.

| Model / features | macro recall | macro F1 | accuracy | n test |
| --- | ---: | ---: | ---: | ---: |
| **random_forest / rr+morphology** | **0.459** | **0.405** | 0.872 | 49,646 |
| lda / rr_only | 0.541 | 0.340 | 0.751 | 49,646 |
| random_forest / morphology_only | 0.358 | 0.289 | 0.742 | 49,690 |
| lda / rr+morphology | 0.349 | 0.342 | 0.828 | 49,646 |
| random_forest / rr_only | 0.303 | 0.294 | 0.816 | 49,646 |
| lda / morphology_only | 0.300 | 0.296 | 0.800 | 49,690 |

Sorted by macro recall. Note that the row with the highest macro recall is not
the best model — see "The trap in this table" below.

## 1. The headline: Random Forest on RR + morphology

Macro recall **0.459**, macro F1 **0.405** — the best on both counts.

| class | support | recall | precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| N (normal) | 44,196 | 0.904 | 0.959 | 0.931 |
| S (supraventricular) | 1,836 | 0.271 | 0.200 | 0.230 |
| V (ventricular) | 3,219 | 0.844 | 0.594 | 0.697 |
| F (fusion) | 388 | 0.273 | 0.120 | 0.166 |
| Q (unknown) | 7 | 0.000 | 0.000 | 0.000 |

Accuracy is 0.872 and means nothing: N is 89% of DS2, so predicting "normal"
for every beat scores ~0.89 while finding no arrhythmia at all. The model
clears that bar by 0.9 points of accuracy and is doing something genuinely
different — which is exactly why accuracy is the wrong instrument.

**V at 0.844 recall is the real success.** Of 3,219 ventricular beats in
held-out patients, 2,717 were found. Precision 0.594 means roughly two false
alarms for every three true detections; for a screening tool that flags beats
for review rather than acting autonomously, that is a workable trade.

## 2. The ablation: the two feature families are complementary

Within Random Forest:

| features | macro recall | N rec | S rec | V rec |
| --- | ---: | ---: | ---: | ---: |
| RR + morphology | 0.459 | 0.904 | 0.271 | 0.844 |
| morphology only | 0.358 | 0.765 | 0.147 | 0.858 |
| RR only | 0.303 | 0.871 | 0.062 | 0.580 |

The combination beats either half, so neither is redundant — but the interesting
part is *which* half does what.

**V is carried almost entirely by morphology.** Morphology alone gets V recall
0.858; RR alone gets 0.580. Adding RR back to morphology does not improve V
recall at all (0.844 vs 0.858 — slightly worse) but it triples V *precision*
(0.594 vs 0.365). RR is not finding ventricular beats; it is suppressing the
false alarms morphology generates. That is a real division of labour and it
matches the physiology: a PVC has an aberrant wide waveform, which is a shape
question, and the compensatory pause that follows is a timing question.

**S needs both and gets neither well.** 0.062 from RR alone, 0.147 from
morphology alone, 0.271 together — better than additive, and still the worst
number in the headline model.

## 3. The trap in this table

**`lda / rr_only` has the highest macro recall of all six models (0.541) and is
the worst model in the set.**

It gets there by spraying predictions into the rare classes:

| class | support | recall | precision |
| --- | ---: | ---: | ---: |
| F | 388 | 0.588 | **0.041** |
| Q | 7 | 0.429 | **0.005** |

F recall 0.588 sounds like it found most of the fusion beats. It did — by
calling 5,103 normal beats "F" to catch 228 real ones. Q recall 0.429 means 3
of 7 beats, achieved by labelling 632 normal beats Q. Meanwhile its N recall
collapses to 0.773 and its S recall is 0.056, the second-worst in the table.

Macro recall averages per-class recall with no reference to precision, so a
class with 7 members contributes exactly as much as one with 44,196, and
over-predicting a rare class is rewarded with no offsetting penalty. Macro F1
ranks the same six models correctly — `lda / rr_only` drops from 1st to 3rd
(0.340), and `random_forest / rr+morphology` rises from 2nd to 1st (0.405).

The headline model happens to be the right choice on macro recall too, so
nothing was actually decided wrongly here. But the metric got the ranking of
the field wrong, and the only reason it did not mislead is that nobody looked
past row one.

**Recommendation, not implemented:** report macro F1 alongside macro recall in
`evaluate.py`, and treat any per-class recall paired with precision below ~0.05
as an artefact rather than a finding. Not implemented and not yet planned —
deferred by agreement on 2026-09-06, recorded here so the reasoning is not lost.

## 4. What is still broken

**S (supraventricular) at 0.271 recall is the ceiling that matters.** From the
Random Forest confusion matrix:

- 1,290 of 1,836 real S beats were called N — 70% of the class missed entirely.
- 1,759 real N beats were called S — false alarms outnumber the 498 true
  detections 3.5 to 1, hence precision 0.200.

S is defined largely by *prematurity* — the beat arrives early — with a
waveform that can look nearly normal. That makes it the class where an
inter-patient split hurts most, because "early" is relative to a rhythm that
differs per patient. This is already the fixed version: Problem #9 in
`IMPLEMENTATION.md` records S recall of 0.063 before `min_samples_leaf=50`, so
0.271 is a 4.3x improvement over a model that was memorising patient-specific
morphology. It is progress on a number that is still bad.

**F (fusion) at 0.273 recall, 0.120 precision.** A fusion beat is a hybrid of a
normal and a ventricular beat, so it sits between two classes the model already
confuses. 275 of 388 F beats were called N. Not investigated; the published
comparisons this project benchmarks against do not report F either.

**Q is not a result.** 7 beats in DS2. Every model scores 0.000 except the two
that got there by mass over-prediction. Any Q number from this split, in either
direction, should be ignored — the class is too small to measure.

## 5. Caveat on the totals

The `morphology_only` rows were scored on **49,690** beats; every other row on
**49,646**. RR features need both a preceding and a following beat, so the
first and last beat of each of the 22 test records are dropped when RR is in the
feature set — 44 beats, 0.09% of the test set.

Immaterial to every comparison above. Recorded so that nobody later tries to
reconcile the totals across rows and concludes something is wrong.

## Provenance

Nearest commit `91db046`. The run wrote its metrics at 14:04 and the commit was
made at 14:44, so the working tree at run time was not captured — this is the
nearest commit, not a verified provenance record. Full commands, package
versions, and hyperparameters are in the `run` block of the JSON.

DS2 has now been evaluated more than once across Phase 1b; see Deviation 9 in
`PlanDoc/IMPLEMENTATION.md`. No hyperparameter was selected on it, but it should
be treated as partially spent for this feature set. Further tuning goes through
the validation records.
