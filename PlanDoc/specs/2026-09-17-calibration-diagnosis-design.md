# Phase 1f — Calibration diagnosis and probability contract

**Status:** design approved in conversation 2026-09-17. Stage A implemented same
day. Stage B not started.

**Naming:** `PLAN.md` reserves Phase 2 for MIMIC-III. Like 1d and 1e, this work
was requested directly rather than arising from `PLAN.md`, so it takes the next
letter rather than a number.

---

## 1. Purpose

Phase 1d asked whether a neural network beats the classical floor, and which
change caused the difference. Both questions are answered: it does (headline
macro-F1 0.545 against Random Forest's 0.405), and the cause is the RR branch,
not the imbalance handling — `waveform_only` collapses to 0.311.

This phase exists because the **consumer changed**. The beat classifier is
intended to feed the fusion/agentic layer of `PLAN.md` Phase 4. That reframes
what "good" means:

- A consumer that reads `argmax` needs a good decision rule.
- A consumer that reads `P(S | beat)` and reasons with it needs a good *number*.

Every metric in the repository serves the first consumer. `evaluate.py` takes
hard labels; `thresholds.py` returns `argmax(proba / t)`. Nothing has ever asked
whether the probabilities themselves are trustworthy.

### The premise this design corrects

The conversation that prompted this phase proposed downsampling class N to
4–5k beats to fix S precision. That is a prior lever, and the evidence says the
problem is not the prior:

| arm | N→S false positives | S recall | S precision |
|---|---|---|---|
| plain_ce | 320 | 0.090 | 0.338 |
| oversample | 392 | 0.145 | 0.391 |
| weighted_ce | 1,998 | 0.416 | 0.273 |
| focal | 3,015 | 0.693 | 0.294 |

S precision never leaves 0.27–0.39 across an 8× range of false-positive volume.
Five different imbalance treatments slide along one curve. Downsampling N is a
sixth point on the same curve, and it additionally deletes the tail of the N
distribution — the unusual-but-normal beats that are precisely what the model
misclassifies.

Under the agentic consumer the question dissolves anyway: `argmax` precision is
a property of a decision rule the agent replaces.

---

## 2. The specific reason to expect miscalibration

The Phase 1d winner trains with `losses.inverse_frequency_weights(cap=10.0)`
(`weight_cap_selected = 10.0`). Weighting class `c` by `w_c` teaches a tilted
posterior, approximately `p̃(c|x) ∝ w_c · p(c|x)`.

So `P(S|beat) = 0.5` from that model does **not** mean half of such beats are S.
No metric in the repository would reveal this, and an agent multiplying or
thresholding those numbers would be systematically wrong, silently.

---

## 3. Stage A — measure, fit nothing

### 3.1 Questions

1. **Is focal's win a representation win or a moved threshold?** Answered by
   per-class average precision, which integrates over every threshold. If
   `plain_ce` ranks S as well as `focal` does, the unweighted model is the better
   base for a probability consumer.
2. **How far from calibrated are these probabilities?** Brier, NLL, ECE,
   per-class reliability, prior drift.

### 3.2 Why average precision and not ROC-AUC

S is 3.7% of DS2. ROC-AUC's false-positive rate divides by 44,196 N beats, so a
thousand new false alarms move it by 0.02 — it reads near-perfect while the
precision a reader experiences collapses. Average precision divides by the beats
*called* S. ROC-AUC is reported because the literature quotes it; it ranks
nothing here.

### 3.3 Evidence sets

Two, because they support different claims and only one can support a fit.

**(a) Loss comparison → Phase 1d single models on DS2.** `plain_ce`,
`weighted_ce`, `focal`, `oversample`, `waveform_only`. These were fitted on all
of DS1 with no held-out set (`final_fit_note`), so **no out-of-fold predictions
exist for them**. DS2 average precision is valid — DS2 was never trained on —
but they can never be used to fit a correction.

**(b) Calibration → `beat_focal_fold0-4`.** Pooled out-of-fold on DS1 for
diagnosis and, later, fitting. DS2 for reporting. Plus the 5-model ensemble via
`ensemble.average_proba`.

### 3.4 The reproduction control, run first

The checkpoints were written by torch 2.11.0+cu128 on CUDA and are read by torch
2.14.0 on CPU. `train.py` documents ~4.3e-4 probability drift across devices, so
re-scoring `focal.pt` on DS2 must reproduce the published headline macro-F1
**0.5454 within 0.005** — a tolerance, not an equality.

If it fails the script exits. Every number downstream would otherwise describe a
different model than the one published.

**Measured 2026-09-17: 0.5454, delta −0.00004, n_test 49,646. Passed.**

### 3.5 Module boundaries

| module | the one question it answers | status |
|---|---|---|
| `evaluate.py` | how good are the hard-label decisions? | **unchanged** |
| `ranking.py` | how well are beats ordered, independent of threshold? | new |
| `calibration.py` | do the probabilities mean what they say? | new |
| `plots.py` | render the above; computes nothing | new |
| `scripts/diagnose_calibration.py` | driver | new |

`evaluate.py` is untouched deliberately: every published number in `results/`
flows through it, and this work must not be able to disturb them.

`plots.py` computes no metric and reads no file. The arrays behind each figure
are written to the results JSON, so a figure can be redrawn without re-scoring
49,646 beats and can never disagree with the number it shows.

### 3.6 Honesty constraints, built in rather than remembered

- **ECE is bin-sensitive.** Reported at 10 and 15 bins; the reliability curve is
  the real artifact. `DEFAULT_BIN_COUNTS` carries this.
- **Rare classes break these metrics too.** Average precision over Q's 7 DS2
  beats is noise for the same reason its F1 is. All five classes are reported;
  `HEADLINE_CLASSES` ranks. A class with no positives returns `nan`, never 0.0 —
  0.0 reads as "the model failed" when the truth is "the test set cannot say".
- **Fold/model pairing is asserted.** `pooled_oof` checks each checkpoint's
  `meta["fold"]` against the fold whose records it is about to score. Scoring
  held-out records with the wrong fold's model is exactly the leak the folds
  exist to prevent.
- **Stage A fits nothing, so Stage A cannot leak.** Keeping measurement free of
  the constraint means these functions can be pointed at DS2 without anyone
  having to remember whether that was allowed.

### 3.7 Tests

`test_ranking.py`, `test_calibration.py`, `test_plots.py`. The failure mode
being avoided is Problem #10 — a regression test that passed with both the buggy
and the fixed parameter. So each metric is checked against a case where its
value is known by construction *and* a case that should move it:

- Perfect ordering → AP 1.0; uninformative ordering → AP ≈ prevalence, not 0.0.
- Calibrated synthetic fixture (draw `p` uniform, draw label with probability
  `p`) → ECE → 0; the same fixture sharpened toward 0/1 → ECE rises by >0.1.
- `prior_drift.ratio` → 1.0 when calibrated; → 10.0 for a model asserting 0.20
  on a class occurring 2% of the time, which is the cap-10 distortion isolated.
- Risk-coverage: informative confidence → accuracy falls with coverage;
  uninformative confidence → flat, which is the signal abstention buys nothing.

### 3.8 Figures

Written to `results/figures/YYYY-MM-DD-*.png`:

1. Precision-recall per class, arms overlaid, with the chance line at prevalence.
2. Reliability per class, with bin populations in a lower panel — a bin holding
   four beats plots at 0.0 or 1.0 and reads as catastrophic miscalibration; only
   the counts reveal it.
3. Confidence distribution per class.
4. Prior drift: mean predicted over observed rate, log scale, 1.0 marked.
5. Risk-coverage: accuracy against fraction of beats kept.
6. Metric comparison bars: macro AP, Brier, classwise ECE.
7. Reliability for S on pooled out-of-fold — the set a correction may be fitted on.

---

## 4. Stage B — fit the correction (not started)

Designed after Stage A's numbers land, because which correction to fit is a
question Stage A answers. Candidates, in increasing capacity:

1. **Analytic prior correction**, `p ∝ p̃ / w_c`. Zero fitting, exactly right for
   weighted CE. Only approximate for focal, whose γ term is not a per-class
   rescale. The control: does the simple thing suffice?
2. **Temperature scaling**, one scalar, minimising NLL.
3. **Vector scaling**, per-class scale and bias on logits — subsumes both above.

Logits are recoverable from the stored probabilities: `log(p) = z − logsumexp(z)`,
and softmax is invariant to additive constants, so `softmax(log(p)/T) = softmax(z/T)`
exactly. No change to `predict_proba` is needed — only a floor on `p`.

**Constraint Stage B carries and Stage A does not:** the fit may only see pooled
out-of-fold predictions, never DS2. Same discipline `thresholds.py` already
states. Given Problem #10, the guard needs a test that fails when removed.

---

## 5. Dependencies

`matplotlib==3.11.1` promoted from transitive to direct, and pinned.
`requirements.txt` previously stated it "arrives transitively via wfdb and is not
imported by any module here"; `plots.py` now imports it directly. Approved in
conversation 2026-09-17.

No other dependency introduced. `scikit-learn` was already direct (Phase 1b) and
supplies `average_precision_score` and `precision_recall_curve`.

---

## 6. Explicit non-goals

- No retraining. Stage A reads checkpoints that already exist.
- No change to `evaluate.py`, `train.py`, `losses.py`, or the model.
- No published number altered.
- No Stage B fitting until Stage A's numbers are read.
- Not the ensemble improvements (threshold tuning on the ensemble, more diverse
  members, adding already-trained models that err differently). Those are queued
  behind this and are a separate design.
