"""Baseline classifier pipelines."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from ecg import baseline, config
from ecg.baseline import (
    CLASSIFIERS,
    FEATURE_SETS,
    build_pipeline,
    feature_importances,
    grouped_importance,
)


def toy_data(n: int = 400, seed: int = 0):
    """Two separable classes on two features, at very different scales.

    The scale gap mirrors the real thing: RR features are in seconds (~1.0),
    morphology features are z-scored amplitudes.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.normal(1.0, 0.05, n),  # "seconds"
            rng.normal(0.0, 1.0, n),  # "z-scored"
        ]
    )
    y = np.zeros(n, dtype=np.int64)
    y[n // 2 :] = 2
    X[n // 2 :, 0] += 0.4  # class V arrives earlier/later
    return X, y


@pytest.mark.parametrize("kind", CLASSIFIERS)
def test_pipeline_puts_the_scaler_before_the_classifier(kind):
    """Scaling inside the pipeline is what makes leakage structurally impossible."""
    pipeline = build_pipeline(kind)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "classifier"]


@pytest.mark.parametrize("kind", CLASSIFIERS)
def test_pipeline_learns_a_separable_problem(kind):
    X, y = toy_data()
    pipeline = build_pipeline(kind, classes=np.unique(y))

    pipeline.fit(X[:300], y[:300])
    predictions = pipeline.predict(X[300:])

    assert (predictions == y[300:]).mean() > 0.8


def test_scaler_statistics_come_from_training_data_only():
    """The leakage regression test, at the pipeline level.

    If the scaler ever saw the held-out rows, its mean would shift toward them.
    """
    rng = np.random.default_rng(1)
    train = np.column_stack([rng.normal(0.0, 1.0, 200), rng.normal(0.0, 1.0, 200)])
    held_out = train + 100.0
    y = np.array([0, 2] * 100)

    pipeline = build_pipeline("random_forest", classes=np.unique(y))
    pipeline.fit(train, y)

    fitted_mean = pipeline.named_steps["scaler"].mean_
    assert np.all(np.abs(fitted_mean) < 1.0)

    everything = np.vstack([train, held_out])
    assert np.all(np.abs(everything.mean(axis=0)) > 40), "fixture must differ"


def test_predictions_are_reproducible_for_a_fixed_seed():
    X, y = toy_data()

    a = build_pipeline("random_forest", seed=7, classes=np.unique(y))
    b = build_pipeline("random_forest", seed=7, classes=np.unique(y))
    a.fit(X[:300], y[:300])
    b.fit(X[:300], y[:300])

    assert np.array_equal(a.predict(X[300:]), b.predict(X[300:]))


def test_balanced_weighting_helps_the_minority_class():
    """This is where Phase 1's deferred class weights actually get applied."""
    rng = np.random.default_rng(2)
    n_major, n_minor = 950, 50
    X = np.concatenate(
        [rng.normal(0.0, 1.0, (n_major, 2)), rng.normal(1.2, 1.0, (n_minor, 2))]
    )
    y = np.array([0] * n_major + [2] * n_minor)

    balanced = build_pipeline("random_forest", balanced=True, classes=np.unique(y))
    plain = build_pipeline("random_forest", balanced=False, classes=np.unique(y))
    balanced.fit(X, y)
    plain.fit(X, y)

    minority = y == 2
    balanced_recall = (balanced.predict(X)[minority] == 2).mean()
    plain_recall = (plain.predict(X)[minority] == 2).mean()

    assert balanced_recall >= plain_recall


def test_lda_uses_uniform_priors_when_balanced():
    """LDA has no class_weight; uniform priors are the equivalent lever."""
    pipeline = build_pipeline("lda", balanced=True, classes=np.arange(5))
    priors = pipeline.named_steps["classifier"].priors

    assert priors is not None
    assert np.allclose(priors, 0.2)

    assert build_pipeline("lda", balanced=False).named_steps["classifier"].priors is None


def test_unknown_classifier_is_rejected():
    with pytest.raises(ValueError, match="unknown classifier"):
        build_pipeline("magic")


def test_feature_importances_group_into_the_two_families():
    """The ablation question: which half of the Phase 1 output carries signal?"""
    X, y = toy_data()
    names = ("pre_rr", "morph_00")

    pipeline = build_pipeline("random_forest", classes=np.unique(y))
    pipeline.fit(X, y)
    importances = feature_importances(pipeline, names)
    groups = grouped_importance(importances)

    assert set(importances) == set(names)
    assert set(groups) == {"rr", "morphology"}
    assert sum(groups.values()) == pytest.approx(1.0, abs=1e-6)
    # Only feature 0 separates the classes in this fixture.
    assert groups["rr"] > groups["morphology"]


def test_lda_exposes_no_feature_importances():
    X, y = toy_data()
    pipeline = build_pipeline("lda", classes=np.unique(y))
    pipeline.fit(X, y)

    assert feature_importances(pipeline, ("a", "b")) == {}
    assert grouped_importance({}) == {}


def test_three_feature_sets_are_defined_for_the_ablation():
    assert set(FEATURE_SETS) == {"rr+morphology", "rr_only", "morphology_only"}
    for name, kwargs in FEATURE_SETS.items():
        assert kwargs["use_rr"] or kwargs["use_morphology"], name


def premature_beat_data(n_records: int = 8, n_morph: int = 30, seed: int = 0):
    """Synthetic inter-patient data reproducing the real failure mode.

    Three ingredients, all of which turned out to be necessary before a
    memorising model actually failed:

    * **Severe class imbalance** -- S is ~2.2% of beats, matching the real
      training split. This is the ingredient that makes the failure appear; at a
      comfortable 20% prevalence even fully-grown trees find the timing rule.
    * **Overlapping timing distributions** (2 features) -- S beats arrive early
      on average, but the distributions overlap, so the rule is statistical
      rather than a clean threshold. It is consistent across patients, so it
      generalises.
    * **Patient-specific morphology** (30 features) -- S beats are offset along
      a direction that differs per patient and is reversed for half of them.
      Within a patient it predicts the label perfectly; across patients it
      teaches contradictory rules.

    The 30-to-2 feature ratio matters: with ``max_features='sqrt'`` a tree sees
    ~5 of 32 features per split, so morphology is usually what it gets offered.
    Tiny leaves let it carve out pure, patient-specific regions; large leaves
    force it onto the timing rule that actually transfers.
    """
    rng = np.random.default_rng(seed)
    X, y, record = [], [], []

    for r in range(n_records):
        base_rr = rng.uniform(0.6, 1.1)
        direction = rng.normal(0.0, 1.0, n_morph)
        direction /= np.linalg.norm(direction)
        if r % 2:
            direction = -direction  # half the patients teach the opposite rule

        for _ in range(2500):
            is_s = rng.random() < 0.022
            pre = base_rr * (
                rng.normal(0.62, 0.10) if is_s else rng.normal(1.0, 0.13)
            )
            morph = rng.normal(0.0, 1.0, n_morph)
            if is_s:
                morph += 6.0 * direction  # locally perfect, globally misleading

            X.append(np.concatenate([[pre, pre / base_rr], morph]))
            y.append(1 if is_s else 0)
            record.append(r)

    return np.asarray(X), np.asarray(y), np.asarray(record)


def test_baseline_generalises_across_patients_using_timing():
    """Regression test for the failure that made S recall collapse to 0.06.

    The first version of RANDOM_FOREST_PARAMS grew trees to full depth with
    ``min_samples_leaf=2``. On real data that scored S recall of 0.099 on
    held-out patients even though a single RR threshold reaches 0.97 -- the
    trees had memorised patient-specific morphology. Patients are held out here,
    so a model that memorises morphology cannot pass.

    Verified to discriminate: this fixture yields S recall 0.00-0.09 with
    ``min_samples_leaf=2`` and 0.68-0.84 with 50, across several seeds.
    """
    X, y, record = premature_beat_data()
    train = record < 6
    test = ~train

    pipeline = baseline.build_pipeline("random_forest", classes=np.unique(y[train]))
    pipeline.fit(X[train], y[train])
    predicted = pipeline.predict(X[test])

    recall = (predicted[y[test] == 1] == 1).mean()
    assert recall > 0.5, (
        f"S recall {recall:.3f} on held-out patients -- the model is keying on "
        f"patient-specific morphology instead of the timing feature"
    )


def test_leaf_size_is_large_enough_to_generalise():
    """Guards the specific parameter that caused the collapse."""
    assert baseline.RANDOM_FOREST_PARAMS["min_samples_leaf"] >= 20
