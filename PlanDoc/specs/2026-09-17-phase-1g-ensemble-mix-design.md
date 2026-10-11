# Phase 1g — an ensemble of different losses

**Status:** designed and implemented 2026-09-17. Not yet run. Awaiting a Colab
GPU session.

**Naming:** follows 1d (hybrid network), 1e (repeated CV + ensembling) and 1f
(calibration diagnosis). `PLAN.md` still reserves Phase 2 for MIMIC-III.

---

## 1. Purpose

Phase 1f measured ranking quality free of any decision threshold, and the result
reversed a Phase 1d conclusion.

| loss | macro-F1 (Phase 1d ranking) | average precision on S |
|---|---|---|
| focal | **0.545** — 1st, and the one kept | 0.285 — 4th |
| weighted | 0.492 — 2nd | 0.256 — 5th |
| oversample | 0.484 — 4th | **0.388** — best |
| plain | 0.379 — 5th | 0.340 — 2nd |

Focal won on a score that moves when the decision line moves. On the score that
does not, it ranks S beats *worse* than the two arms that lost to it.

**The current ensemble is five copies of focal — the weakest S ranker.** An
ensemble gains from members that make different mistakes. This phase trains fold
models for all four losses and asks whether mixing them beats five copies of one.

---

## 2. Why the existing checkpoints cannot be reused

`plain_ce.pt`, `weighted_ce.pt` and `oversample.pt` were each fitted on **all of
DS1** — `final_fit_note` in the Phase 1d results says so explicitly. No
out-of-fold predictions exist for them.

That matters because choosing a composition is a *selection*. Comparing
candidates on DS2 and keeping the best would be selecting on the test set, which
`thresholds.py` forbids in as many words and which would invalidate every DS2
number the project has published. Fold models are therefore retrained so the
choice can be made on DS1 alone.

---

## 3. The rule, fixed before the run

`COMPOSITIONS` in `scripts/train_ensemble_mix.py`:

| # | composition | members |
|---|---|---|
| A | focal | 5 — today's ensemble, the reference |
| B | focal + plain | 10 |
| C | focal + plain + oversample | 15 |
| D | all four losses | 20 |
| E | plain + oversample | 10 — drops focal |

**E drops the incumbent on purpose.** Phase 1f found focal the weakest S ranker,
so "leave it out" is a hypothesis this design has to be able to express. A
candidate list where every option keeps focal could not test it, and
`tests/test_ensemble_mix.py` asserts that at least one candidate drops it.

Selection metric: **pooled out-of-fold macro average precision over N/S/V/F.**
The winner is chosen on that, and only then is DS2 scored.

### Ordering is enforced by the script, not by the operator

`main()` computes the out-of-fold scores, prints the winner and the line
`DS2 has not been touched yet.`, and only afterwards loads DS2. Every
candidate's DS2 score is then reported — looking after the choice is made costs
nothing; choosing by those numbers would cost everything. When the out-of-fold
choice and the DS2 best differ, `ingest_colab_results.py` prints both and says
which number to quote.

---

## 4. What is held constant

Everything except the loss. Same `HybridBeatNet`, same optimizer, same epoch
budget and patience, same dropout, same seed, and above all the **same fold
assignment**: `folds.make_folds(n_folds=5, seed=config.PHASE_1D_SEED)`, verified
to reproduce the `nn_folds.json` the existing `beat_focal` members were trained
on. The notebook gates on this before spending GPU time — if the new members
were trained on a different patient split, "mixing helps" and "the split
changed" become impossible to tell apart.

`ArmV2` in `train_nn_v2.py` was not reused: it has no `oversample` flag, and
that file's purpose is the four *architecture* arms. Phase 1g is a separate
question and gets a separate script.

---

## 5. Out-of-fold construction, and its known optimism

Each fold model early-stops on the fold held out from it, and that fold's
predictions become its share of the pooled out-of-fold output.

This is the same construction `train_nn.run_cv` uses to produce the predictions
`thresholds.py` already tunes against, so the mild optimism of stopping on the
held-out fold is present here exactly as it is there. It is **uniform across
arms**, so it cannot favour one composition over another — which is what the
selection depends on.

Before any averaging, the script asserts that every arm's out-of-fold labels and
record ids match the reference arm's, element for element. Averaging misaligned
members would silently mix one beat's prediction with another's.

---

## 6. What gets reported

Against the current ensemble's Phase 1f DS2 numbers: macro AP **0.593**,
S AP **0.393**, headline macro-F1 **0.590**.

Per composition: macro / S / V / F average precision, headline macro-F1,
S recall and precision, Brier, and class-wise ECE. Calibration is included
because Phase 1f showed a model can rank best and calibrate worst — the choice
here is made on ranking, and the calibration numbers say what Stage B inherits.

---

## 7. Explicit non-goals

- **No calibration fitting.** That is Phase 1f Stage B and comes after this.
- **No threshold tuning.** Phase 1f showed the S precision-recall curve is flat;
  sliding along it gains nothing on S.
- **No architecture change.** One variable moves in this phase: the loss.
- **No re-run of Phase 1d or 1e.** Their published numbers stand untouched.

---

## 8. Cost

20 training runs. Measured at 26 s/epoch on this laptop's CPU, which puts a
local run at 1.5–3 hours; the Phase 1d folds took 15–35 s each on a Colab T4,
so 10–20 minutes there. Colab was chosen.
