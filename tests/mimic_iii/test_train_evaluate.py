"""Model selection on validation, saving/loading, and the report."""

from __future__ import annotations

import numpy as np
import pytest

from mimic_iii import dataset, evaluate, train


@pytest.fixture
def parts(feature_table):
    return dataset.split_frame(feature_table, dataset.patient_split(feature_table))


def test_selection_uses_validation_and_records_it(parts):
    model, sel = train.select_and_fit(parts["train"], parts["val"])

    assert sel["selected_C"] in train.ModelSpec().C_grid
    assert all("val_auprc" in c for c in sel["candidates"])
    assert 0.0 <= sel["threshold"] <= 1.0
    assert sel["n_train"] == len(parts["train"])


def test_ties_go_to_the_smallest_C(parts):
    """The synthetic positives are separable, so every C reaches AUPRC 1."""
    _, sel = train.select_and_fit(parts["train"], parts["val"])

    best = max(c["val_auprc"] for c in sel["candidates"])
    tied = [c["C"] for c in sel["candidates"] if c["val_auprc"] == best]
    assert sel["selected_C"] == min(tied)


def test_single_class_validation_falls_back_to_defaults(parts):
    val = parts["val"][parts["val"]["label"] == 0]

    _, sel = train.select_and_fit(parts["train"], val)

    assert sel["selected_C"] == 1.0
    assert sel["threshold"] == 0.5
    assert "one class" in sel["selection_note"]
    assert "one class" in sel["threshold_note"]


def test_single_class_training_is_an_error(parts):
    with pytest.raises(ValueError, match="one class"):
        train.select_and_fit(parts["train"][parts["train"]["label"] == 0], parts["val"])


def test_model_round_trips_and_predicts_identically(tmp_path, parts):
    model, sel = train.select_and_fit(parts["train"], parts["val"])
    train.save_model(tmp_path, model, {"selection": sel})

    loaded, meta = train.load_model(tmp_path)

    X, _ = dataset.xy(parts["test"])
    np.testing.assert_array_equal(model.predict_proba(X), loaded.predict_proba(X))
    assert meta["input_columns"] == list(X.columns)


def test_tampered_model_file_is_refused(tmp_path, parts):
    model, sel = train.select_and_fit(parts["train"], parts["val"])
    paths = train.save_model(tmp_path, model, {"selection": sel})
    with open(paths["model"], "ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="hash"):
        train.load_model(tmp_path)


def test_missing_model_names_the_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        train.load_model(tmp_path)


def test_evaluation_reports_cis_and_reliability(parts):
    model, sel = train.select_and_fit(parts["train"], parts["val"])

    result = evaluate.evaluate_split(model, parts["test"], sel["threshold"], n_boot=200)

    assert result["n"] == len(parts["test"])
    assert result["auroc_ci"]["n_boot"] == 200
    assert sum(b["n"] for b in result["reliability"]) == result["n"]
    cm = result["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == result["n"]


def _report(source, parts):
    from mimic_iii import config

    model, sel = train.select_and_fit(parts["train"], parts["val"])
    results = {n: evaluate.evaluate_split(model, df, sel["threshold"], n_boot=100)
               for n, df in parts.items()}
    counts = {n: {"patients": int(df["subject_id"].nunique()), "stays": len(df),
                  "positive": int(df["label"].sum())} for n, df in parts.items()}
    cohort = {"flow": [{"step": "x", "stays": 1, "subjects": 1}], "n_stays": 1, "n_positive": 0}
    return evaluate.build_report(source, config.TaskSpec(), cohort, counts, sel, results,
                                 evaluate.top_coefficients(model), inputs={})


def test_demo_report_carries_the_not_clinical_banner(parts, tmp_path):
    report = _report("demo", parts)
    written = evaluate.write_report(tmp_path, report)

    text = written["markdown"].read_text()
    assert "NOT CLINICAL PERFORMANCE" in text
    assert report["data_source"]["is_demo"] is True


def test_full_data_report_still_says_research_only(parts):
    report = _report("full", parts)

    assert "not a medical device" in report["banner"]
    assert report["data_source"]["is_demo"] is False
