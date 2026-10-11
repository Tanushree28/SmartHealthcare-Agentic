"""The regression test for imbalance handling.

**Read this before changing it.** Problem #10 in PlanDoc/IMPLEMENTATION.md
records a regression test that passed with both the buggy parameter and the
fixed one, and so asserted nothing. The missing ingredient was class imbalance:
only once S prevalence matched the real split did the failure reproduce.

The fixture below therefore mirrors the real DS1 training split: S is ~2.2% of
beats (863 of 39,840), it is identified by timing rather than shape, and its
waveform is drawn from the same distribution as N so there is nothing to
memorise. Under plain cross-entropy the model should ignore S almost entirely.

**Calibration, measured not assumed (2026-09-16).** The first version of this
test asserted that plain cross-entropy would collapse to near-zero S recall.
Measured, it does not: on a synthetic fixture with a clean timing signal, plain
CE reaches ~0.53 given enough epochs. Two further findings forced the current
shape:

* At 8 epochs nothing has converged -- validation macro-F1 bounces between 0.02
  and 0.49 from epoch to epoch, so the numbers were noise. 20 epochs is the
  floor for a stable comparison.
* Every fixture harder than this one drives *all three* losses to ~0.05, and
  every fixture easier lets plain CE solve it. The discriminating window is
  narrow, so the test asserts the property that actually holds across it:
  **weighting materially recovers S where plain leaves most of it unfound.**

That is still the regression this test exists to catch -- if class weighting
stops working, weighted collapses to plain's number and these assertions fail.
Verified by pointing the weighted assertions at "plain" and observing the
failure before accepting them.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg import config, dataset, evaluate
from ecg.nn import train

# 15 training runs of 20 epochs: minutes, not seconds. Deselected by default so
# the everyday suite stays fast. **Run this module before accepting any change
# to losses.py, train.py or the model** -- that is what it is here for:
#     ./venv/bin/python -m pytest tests/test_imbalance_regression.py -m slow
pytestmark = pytest.mark.slow

#: 863 / 39,840 in the real DS1 training split.
S_PREVALENCE = 0.022


def imbalanced_beats(n=4000, seed=0, s_prevalence=S_PREVALENCE):
    """N and S with identical waveform distributions, separated only by timing."""
    rng = np.random.default_rng(seed)
    n_s = max(int(round(n * s_prevalence)), 1)

    y = np.zeros(n, dtype=np.int64)
    y[:n_s] = 1
    rng.shuffle(y)

    # Waveforms carry no class information at all -- if the model succeeds, it
    # succeeded on timing, which is the property under test.
    X = rng.standard_normal((n, config.BEAT_LENGTH)).astype(np.float32)

    rr = np.empty((n, 6), dtype=np.float32)
    normal_rr = 0.85 + 0.05 * rng.standard_normal(n)
    rr[:, 0] = np.where(y == 1, 0.50 + 0.04 * rng.standard_normal(n), normal_rr)
    rr[:, 1] = np.where(y == 1, 1.15 + 0.05 * rng.standard_normal(n), normal_rr)
    rr[:, 2] = normal_rr
    rr[:, 3] = 0.85
    rr[:, 4] = rr[:, 0] / 0.85
    rr[:, 5] = rr[:, 1] / 0.85

    return dataset.BeatData(
        X=X,
        rr=rr.astype(np.float32),
        y=y,
        record=np.full(n, "sim", dtype=object),
        n_input=n,
        n_dropped_edge=0,
    )


def s_recall(loss_kind, seed):
    # 20 epochs: at 8 the model has not converged and the metric is noise.
    cfg = train.TrainConfig(epochs=20, batch_size=128, patience=20, seed=seed)
    result = train.train_model(
        imbalanced_beats(seed=seed),
        imbalanced_beats(n=2000, seed=seed + 100),
        loss_kind,
        cfg,
    )
    scored = evaluate.evaluate(result.val_y, result.val_proba.argmax(axis=1))
    return scored.recall_of("S")


#: Seeds the bounds below were calibrated on. Five, not three: per-seed S recall
#: for plain CE spans 0.341 to 0.773, so a three-seed mean is still noisy.
SEEDS = (1, 2, 3, 4, 5)

# Measured 2026-09-16 on MPS, 20 epochs (mean / min over SEEDS):
#     plain     0.641 / 0.341
#     weighted  0.900 / 0.795
#     focal     0.968 / 0.955
# Every bound below sits between plain's spread and weighted's, so it fails if
# the weighting stops working and passes otherwise.


def test_class_weighting_recovers_the_minority_class():
    """Arm 1 must find what arm 0 misses.

    Plain CE averages 0.641 here, so this bound fails if weighting stops
    working -- which is the regression this test exists to catch. Verified by
    pointing this assertion at "plain": it failed on all five seeds.
    """
    scores = [s_recall("weighted", seed) for seed in SEEDS]

    assert np.mean(scores) > 0.80, f"weighted S recall {np.mean(scores):.3f}"
    assert min(scores) > 0.70, f"worst seed {min(scores):.3f}"


def test_focal_loss_recovers_the_minority_class():
    scores = [s_recall("focal", seed) for seed in SEEDS]

    assert np.mean(scores) > 0.85, f"focal S recall {np.mean(scores):.3f}"
    assert min(scores) > 0.80, f"worst seed {min(scores):.3f}"


def test_plain_cross_entropy_leaves_a_clear_gap_to_close():
    """Arm 0's measured behaviour: it finds most S on a good seed, half on a bad
    one, and never reliably. If plain CE ever matched weighted here, the fixture
    would have stopped discriminating and every bound above would be vacuous."""
    scores = [s_recall("plain", seed) for seed in SEEDS]

    assert np.mean(scores) < 0.75, f"plain S recall {np.mean(scores):.3f}"


def test_weighting_beats_plain_on_every_seed():
    """The comparison stated directly, so a marginal fixture cannot hide it.

    Only the sign is asserted, not the size: the per-seed gap ranges 0.023 to
    0.591, so any fixed margin would be fitted to the seeds it was measured on.
    """
    for seed in SEEDS:
        plain = s_recall("plain", seed)
        weighted = s_recall("weighted", seed)
        assert weighted > plain, f"seed {seed}: plain {plain:.3f} >= weighted {weighted:.3f}"
