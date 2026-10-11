# Phase 1d — Hybrid neural classifier and imbalance study

**Status:** design approved 2026-09-16. Not implemented. No implementation plan
yet. Not committed.

**Naming:** `PLAN.md` reserves "Phase 2" for MIMIC-III. This work is numbered
**1d** so the plan and the repository do not silently disagree. Like Phase 1c, it
was requested directly rather than arising from `PLAN.md`.

---

## 1. Purpose

Phase 1b established a classical floor: Random Forest on RR + morphology, macro-F1
**0.405**, S recall **0.271**. Phase 1d asks two questions and answers them with
measurements rather than assertions:

1. Does a neural network beat that floor?
2. **Which change causes the difference** — the imbalance handling, or the
   architecture?

Question 2 is the reason for the experiment's shape. A single neural-net run that
beat the baseline would tell us nothing about *why*, and the project would carry an
unexplained number forward.

### The premise this design corrects

The request that prompted this phase was "there is huge imbalance, how can we fix
that". The imbalance is real and severe, but it is not the established cause of the
S-class failure, and the design must not be built on the assumption that it is:

- The Phase 1b Random Forest **already** runs `class_weight="balanced"`. Imbalance
  was already counteracted at the loss, and S recall was still 0.271.
- A single hand-written threshold on `pre_rr_ratio < 0.85` achieves **S recall
  0.972** on DS2. A model cannot be starved of a signal one comparison extracts.
- The diagnosed cause (`IMPLEMENTATION.md`, Phase 1b) was **feature dilution**: 30
  morphology features against 6 RR features, with `max_features='sqrt'` offering a
  tree ~6 of 36 columns per split, so the RR features were rarely available.

Imbalance handling is therefore **necessary but not sufficient**. A design that
addressed only imbalance would likely reproduce ~0.27 with more code. This design
addresses both and separates their effects.

---

## 2. Class distribution — two problems wearing one label

Measured from `~/physionet-data/processed/mitdb-ds12`, the DS1/DS2 output the
Phase 1b baseline used. Four paced records excluded per AAMI.

| class | train (18 recs) | val (4 recs) | test / DS2 (22 recs) | train ratio to N |
| ----- | --------------: | -----------: | -------------------: | ---------------: |
| N     |          35,501 |       10,344 |               44,238 |              1:1 |
| V     |           3,080 |          708 |                3,220 |           11.5:1 |
| S     |             863 |           81 |                1,837 |             41:1 |
| F     |             388 |           26 |                  388 |             92:1 |
| Q     |           **8** |        **0** |                **7** |          4,438:1 |
| total |          39,840 |       11,159 |               49,690 |                  |

> **Correction, recorded deliberately.** An earlier draft of this spec quoted
> 63,184 N / 2,462 S / 13 Q as the training counts. Those came from the
> *record-wise* output directory (`processed/mitdb`), not the DS1/DS2 directory the
> baseline actually used. The table above is the correct one. Noted here rather
> than silently fixed, because the two directories are easy to confuse and the next
> person will confuse them too.

**S, V and F are an imbalance problem.** 11.5:1 to 92:1 is steep but learnable:
loss weighting, focal loss and threshold tuning all act here.

**Q is not an imbalance problem. It is an absence of data.** Eight training beats,
seven test beats, zero validation beats. No weighting scheme, resampling method or
synthetic-sample technique creates information that is not present; duplicating 8
beats teaches 8 waveforms. The 8,039 Q beats in the full dataset live almost
entirely in the four paced records, which the AAMI protocol excludes — so Q
collapses to 8 the moment the protocol is followed.

`normalization.json` for this split suggests a class weight of **996.0** for Q.
Applied literally, 8 beats would dominate the gradient. Weights are capped; see §5.

---

## 3. Data flow

The preprocessing pipeline already writes what the network needs. No new
preprocessing is required.

Per record, `<processed>/beats/<record>.npz` holds:

- `X` — `(n_beats, 300)` float32, bandpass-filtered and z-score normalized with
  train-split statistics
- `y` — AAMI class indices in the fixed order `("N", "S", "V", "F", "Q")`
- `r_peaks` — sample indices, the input to RR feature extraction

A new `src/ecg/dataset.py` assembles per-split arrays and pairs every beat with its
6 RR features (`pre_rr`, `post_rr`, `local_rr`, `global_rr`, `pre_rr_ratio`,
`post_rr_ratio`) from the existing `features.rr_features`.

### Boundary beats

RR features need both a preceding and a following beat, so the first and last beat
of every record has none. The hybrid model requires both inputs, so those beats are
dropped — the same 44 beats (0.09%) the Phase 1b `rr+morphology` rows dropped. This
is deliberate: it keeps the denominator identical to the rows this study compares
against, rather than introducing a third total. The waveform-only ablation (arm 5)
drops them too, for the same reason, even though it does not use RR features.

---

## 4. Model

A two-branch network. The waveform branch learns shape; the RR branch carries
timing; they join before the classifier.

```
  beat (300 samples, 1 channel)          RR features (6)
             |                                  |
  Conv1d(1  -> 32, k=7) + BN + ReLU + MaxPool   Linear(6 -> 32) + ReLU
  Conv1d(32 -> 64, k=5) + BN + ReLU + MaxPool   Linear(32 -> 32) + ReLU
  Conv1d(64 -> 64, k=3) + BN + ReLU + MaxPool           |
  Conv1d(64 ->128, k=3) + BN + ReLU                     |
  GlobalAvgPool -> 128                                  |
             +------------------+-------------------+
                          concat -> 160
                   Linear(160 -> 64) + ReLU + Dropout(0.3)
                          Linear(64 -> 5)
```

Roughly 100k parameters.

**Why two branches rather than one flat input vector.** This is the direct fix for
the diagnosed cause. In the Phase 1b feature matrix the 6 RR columns competed
against 30 morphology columns inside one model. Here they cannot be crowded out:
they enter through a dedicated path and are concatenated at full width with the
128-dimensional shape representation.

**Why not a waveform-only CNN**, which is the standard MIT-BIH architecture in the
literature: `config.SAMPLES_BEFORE/AFTER` crops each beat around its R peak, so
every beat is centred and the inter-beat timing is destroyed by the cropping. A
premature beat and an on-time beat yield the identical window. S is defined by
prematurity, so a waveform-only CNN is structurally blind to the feature that
identifies the class this phase exists to improve. Arm 5 measures this claim rather
than leaving it as an assertion.

**Framework:** PyTorch, pinned, CPU wheels. The training set is ~39.8k beats of 300
samples, so CPU training is adequate and no GPU configuration can go wrong. If no
cp313/arm64 wheel exists for the chosen version, that is a blocker to raise, not to
work around silently.

---

## 5. The seven rows

Every arm uses the identical architecture, data, fold assignment, seed, optimizer
and epoch budget. Exactly one thing differs per row.

| #   | Row                                 | What changes                                 |
| --- | ----------------------------------- | -------------------------------------------- |
| ref | Random Forest, rr+morphology        | nothing — carried over from Phase 1b         |
| 0   | Hybrid, plain cross-entropy         | no imbalance handling at all — the "before"  |
| 1   | Hybrid, class-weighted CE           | inverse-frequency weights, **capped**        |
| 2   | Hybrid, focal loss (gamma=2)        | easy beats down-weighted in the loss         |
| 3   | Best of 1-2, + per-class thresholds | decision cut-offs tuned out-of-fold          |
| 4   | Hybrid, random oversampling         | rare beats duplicated to balance the classes |
| 5   | **Waveform-only**, best setting     | RR branch removed — isolates architecture    |

**Arm 0** is expected to collapse toward predicting N. That is the point: it is the
before-picture, and it makes every later row's improvement attributable.

**Arm 1** caps weights. The cap value is selected by cross-validation, not asserted
here; the binding constraint is that Q's uncapped weight of 996 must not be used.

**Arm 3** costs no retraining — it reuses arm 1 or 2's out-of-fold predictions and
only moves the decision cut-offs. Given that `pre_rr_ratio` alone separates S at
0.972 recall, this is the row most likely to move S, and the reason threshold
tuning is in scope.

**Arm 4** is included because resampling is the fix most readers expect to see. It
carries a known risk in this project: duplicating beats from the same patients is
close to the patient-memorisation failure recorded as Problem #9. If it wins, that
result is to be treated with suspicion and checked, not celebrated.

**SMOTE is deliberately excluded.** It interpolates between beats drawn from
different patients, producing waveforms no heart produced, and it is a known source
of inflated published MIT-BIH figures. Adding a number that would need a paragraph
of caveats is worse than not adding it. Recorded here so the omission reads as a
decision rather than an oversight.

---

## 6. Selection protocol — grouped cross-validation over DS1

### Why the single validation split was abandoned

`splits.ds1_ds2_split` carves 4 validation records out of DS1 (112, 205, 215, 223).
Measured, those 4 records hold **81 S beats, 26 F beats and zero Q beats**. The
whole protocol rests on validation — early stopping, weight cap, focal gamma,
per-class thresholds, and which arm wins. Tuning per-class thresholds against 81 S
beats overfits them: one beat flipping moves S recall by 1.2 points. Q cannot
appear in a validation metric at all.

That is too thin a foundation, so selection moves to cross-validation.

### The protocol

1. **Fold assignment.** The 22 DS1 records are partitioned into **5 record-grouped
   folds** (sizes 4/4/4/5/5), deterministically from a fixed seed. Grouping is by
   record, so no patient appears in both the training and validation side of any
   fold. The assignment is written to JSON as a run artifact.
2. **Per arm, per fold:** train on 4 folds, validate on the held-out fold. Early
   stopping on validation macro-F1 computed over classes with non-zero support.
   Record the stopping epoch and the out-of-fold predictions.
3. **Selection** — weight cap, focal gamma, learning rate, and the winning arm —
   uses the **mean across the 5 folds**, never a single fold. All 944 DS1 S beats
   serve as validation data across the folds, against 81 before.
4. **Thresholds (arm 3)** are tuned on the **pooled out-of-fold predictions**, so
   they are fitted on 944 S beats from 22 patients rather than 81 from 4.
5. **Final fit.** Each arm is retrained once on all 22 DS1 records with its selected
   settings, using the **median stopping epoch across its folds** as a fixed budget
   — there is no held-out set left to early-stop against, and that substitution must
   be stated in the results rather than glossed.
6. **One test pass.** A single scripted run scores all seven rows on DS2 and writes
   the comparison table.

### Test-set discipline

DS2 is touched **once**, for reporting, never for selection. `IMPLEMENTATION.md`
Deviation 9 already records DS2 as partially spent for the Phase 1b feature set;
this phase must not spend it further. No fold, no threshold and no hyperparameter
may be chosen with reference to a DS2 number.

### Cost

Arm 3 trains nothing of its own — it re-scores arm 1 or 2's saved out-of-fold
predictions — so 5 arms train: 0, 1, 2, 4 and 5. That is 5 x 5 = **25 fold runs plus
5 final fits, 30 training runs total**. At a few minutes per run on CPU this is
roughly 1-2 hours, to be run in the background rather than blocking.

---

## 7. Metrics

`evaluate.py` gains **`macro_f1`** alongside the existing `macro_recall`.

This is not cosmetic. Section 3 of `results/2026-08-27-baseline-ds1_ds2.md`
demonstrated that macro recall **mis-ranks models**: `lda / rr_only` ranked first of
six on macro recall while being the worst model in the set, because it sprayed
predictions into rare classes (F precision 0.041, Q precision 0.005) and macro
recall applies no offsetting penalty. Phase 1d ranks seven rows. Shipping that
comparison on a metric already documented to rank wrongly would repeat a known
mistake.

Headline ranking is by macro-F1. Macro recall is still reported.

### Q

Q remains in training (8 beats at a capped weight is harmless) but is **excluded
from the headline macro average** and reported in its own row carrying the 8-beat
explanation. Otherwise a 7-beat class swings the number that ranks every model —
the precise trap documented in the Phase 1b results file. Q is absent from
validation folds by construction, so it cannot enter selection at all.

---

## 8. Testing

Test-driven, following the existing convention: offline, synthetic fixtures, no
processed data required, `requires_data` marker for anything needing the real set.

**The test that matters most.** The regression test for imbalance handling must be
**verified to fail** under arm 0's configuration before it is accepted. Problem #10
records a regression test that passed with both the buggy and the fixed parameter
and therefore asserted nothing; the missing ingredient was class imbalance in the
fixture. The same fixture design applies here, with S prevalence set to ~2.2% to
match the real training split. A test that cannot fail produces confidence without
evidence.

Coverage:

- `model.py` — output shapes, parameter count, determinism under fixed seed, and
  that disabling the RR branch changes the output (proving the branch is wired in)
- `losses.py` — focal loss reduces to cross-entropy at gamma=0; weighted CE matches
  a hand-computed value; the weight cap is applied
- `folds.py` — folds partition DS1 exactly, are disjoint, are record-grouped, are
  reproducible from the seed, and **never include a DS2 record**
- `thresholds.py` — tuning is deterministic, improves the target metric on the data
  it is given, and never receives the test split
- `train.py` — one-epoch smoke test on synthetic fixtures, early stopping triggers,
  checkpoint saves and reloads to identical predictions
- `dataset.py` — X and RR rows stay aligned; boundary beats are dropped from both
  consistently; no record appears in two splits
- `evaluate.py` — `macro_f1` against hand-computed values, including the case where
  a class has zero support

---

## 9. Files

| Path                    | What                                                      |
| ----------------------- | --------------------------------------------------------- |
| `src/ecg/dataset.py`    | Per-split array assembly, X/RR pairing, boundary handling |
| `src/ecg/folds.py`      | Record-grouped CV fold assignment over DS1                |
| `src/ecg/nn/model.py`   | The two-branch architecture                               |
| `src/ecg/nn/losses.py`  | Weighted cross-entropy, focal loss, weight capping        |
| `src/ecg/nn/train.py`   | Training loop, early stopping, checkpointing              |
| `src/ecg/thresholds.py` | Per-class decision threshold tuning                       |
| `scripts/train_nn.py`   | CLI entry point                                           |
| `src/ecg/evaluate.py`   | **Modified** — adds `macro_f1`                            |
| `tests/test_*.py`       | Matching tests per module                                 |
| `requirements.txt`      | **Modified** — adds pinned `torch`                        |

`src/ecg/nn/` is a subpackage, a deliberate departure from the flat `src/ecg/`
layout. Four coupled modules read better grouped than carrying a shared prefix.

Deliverables follow the Phase 1b/1c pattern: `results/<date>-nn-ds1_ds2.{json,md}`
with a full provenance block, new charts in the generated `docs/index.html`, and a
Phase 1d section in `IMPLEMENTATION.md`.

---

## 10. What might not work

Named in advance so that, if they happen, they are not reframed as successes:

- **S may not reach the published 0.76.** 41:1 imbalance on an inter-patient split,
  with only 863 training S beats, is genuinely hard. If the hybrid lands at 0.5,
  the report says 0.5.
- **Q will stay at 0.000.** Eight training beats. This phase cannot fix it and does
  not claim to.
- **F may not improve.** A fusion beat is a hybrid of the two classes the model
  already confuses; nothing in this design specifically targets it.
- **Arm 5 may not lose.** If the waveform-only model matches the hybrid, the timing
  argument in §4 is wrong and must be retracted in writing, not buried.
- **Fold variance may exceed the differences between arms.** If the spread across
  folds is wider than the gap between two arms, the comparison cannot rank them,
  and the report must say so instead of ranking them anyway.
- **Torch may lack a cp313/arm64 wheel** at the pinned version. That is a blocker to
  raise, not to route around.

---

## 11. Out of scope

- MIMIC-III (`PLAN.md` Phase 2)
- Sequence models over consecutive beats (CNN + BiLSTM) — considered and deferred;
  revisit only once the hybrid is measured
- SMOTE and other synthetic-sample methods (§5)
- Any change to preprocessing, filtering, segmentation, or the DS1/DS2 split itself
