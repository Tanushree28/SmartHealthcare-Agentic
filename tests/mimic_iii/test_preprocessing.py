"""Preprocessing statistics come from the training split and nowhere else."""

from __future__ import annotations

import numpy as np
import pytest

from mimic_iii import dataset, features, preprocessing


@pytest.fixture
def parts(feature_table):
    return dataset.split_frame(feature_table, dataset.patient_split(feature_table))


def test_imputer_and_scaler_are_fitted_on_training_rows_only(parts, feature_table):
    model = preprocessing.build_model()
    X_tr, y_tr = dataset.xy(parts["train"])
    model.fit(X_tr, y_tr)

    numeric = features.feature_columns()["numeric"]
    num_pipe = model.named_steps["pre"].named_transformers_["numeric"]
    medians = num_pipe.named_steps["impute"].statistics_
    means = num_pipe.named_steps["scale"].mean_

    train_medians = X_tr[numeric].median().fillna(0).to_numpy()
    np.testing.assert_allclose(medians, train_medians)
    imputed = X_tr[numeric].fillna(X_tr[numeric].median()).fillna(0)
    np.testing.assert_allclose(means, imputed.mean().to_numpy())

    all_medians = dataset.xy(feature_table)[0][numeric].median().fillna(0).to_numpy()
    assert not np.allclose(medians, all_medians), "fixture too uniform to detect leakage"


def test_output_has_no_missing_values(parts):
    model = preprocessing.build_model()
    X_tr, y_tr = dataset.xy(parts["train"])
    model.fit(X_tr, y_tr)

    out = model.named_steps["pre"].transform(dataset.xy(parts["test"])[0])

    assert np.isfinite(out).all()


def test_unseen_category_does_not_break_prediction(parts):
    model = preprocessing.build_model()
    model.fit(*dataset.xy(parts["train"]))
    X_te, _ = dataset.xy(parts["test"])
    X_te = X_te.copy()
    X_te.loc[X_te.index[0], "first_careunit"] = "NEW_UNIT"

    proba = model.predict_proba(X_te)[:, 1]

    assert np.isfinite(proba).all()


def test_entirely_missing_variable_keeps_the_output_width(parts):
    X_tr, y_tr = dataset.xy(parts["train"])
    X_tr = X_tr.copy()
    X_tr["bnp_mean"] = np.nan  # never measured in training

    a = preprocessing.build_model().fit(*dataset.xy(parts["train"]))
    b = preprocessing.build_model().fit(X_tr, y_tr)

    assert preprocessing.output_feature_names(a) == preprocessing.output_feature_names(b)


def test_missingness_flags_pass_through_unchanged(parts):
    model = preprocessing.build_model().fit(*dataset.xy(parts["train"]))
    X_te, _ = dataset.xy(parts["test"])
    out = model.named_steps["pre"].transform(X_te)
    names = preprocessing.output_feature_names(model)

    for flag in features.missing_flag_names():
        np.testing.assert_array_equal(out[:, names.index(flag)], X_te[flag].to_numpy())
