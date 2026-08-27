"""Metrics for beat classification."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ecg import config
from ecg.evaluate import evaluate, format_report, write_json


def test_perfect_prediction_gives_recall_one():
    y = np.array([0, 1, 2, 3, 4, 0, 1])

    result = evaluate(y, y)

    assert result.accuracy == 1.0
    assert result.macro_recall == 1.0
    for m in result.per_class:
        assert m.recall == 1.0


def test_all_normal_predictor_is_exposed_by_recall_not_accuracy():
    """The core reason accuracy is not the headline metric.

    A model that calls everything 'normal' scores high accuracy while finding
    zero arrhythmias. Macro recall must make that obvious.
    """
    y_true = np.array([0] * 86 + [2] * 14)  # 86% N, 14% V
    y_pred = np.zeros(100, dtype=np.int64)  # predict N for everything

    result = evaluate(y_true, y_pred)

    assert result.accuracy == pytest.approx(0.86)
    assert result.recall_of("V") == 0.0
    assert result.macro_recall == pytest.approx(0.5)
    assert "uninformative" in result.to_dict()["accuracy_note"].lower()


def test_confusion_matrix_reconciles_with_test_set_size():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 500)
    y_pred = rng.integers(0, 5, 500)

    result = evaluate(y_true, y_pred)

    assert result.reconciles()
    assert result.matrix.shape == (5, 5)
    assert result.n_test == 500


def test_confusion_matrix_orientation_is_true_by_predicted():
    """Getting this backwards silently swaps recall and precision."""
    y_true = np.array([0, 0, 0, 2])
    y_pred = np.array([0, 2, 2, 2])

    result = evaluate(y_true, y_pred)

    # Two true-N beats were predicted V: row N, column V.
    assert result.matrix[0, 2] == 2
    assert result.matrix[2, 0] == 0


def test_recall_and_precision_computed_on_a_known_case():
    y_true = np.array([2, 2, 2, 2, 0])
    y_pred = np.array([2, 2, 0, 0, 2])

    result = evaluate(y_true, y_pred)

    v = next(m for m in result.per_class if m.name == "V")
    assert v.support == 4
    assert v.recall == pytest.approx(2 / 4)
    assert v.precision == pytest.approx(2 / 3)
    assert v.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))


def test_class_absent_from_test_set_does_not_crash():
    """F and Q are rare enough that a split can legitimately contain none."""
    y = np.array([0, 0, 2, 2])

    result = evaluate(y, y)

    f = next(m for m in result.per_class if m.name == "F")
    assert f.support == 0
    assert f.recall == 0.0
    # An absent class must not drag the macro average down.
    assert result.macro_recall == 1.0


def test_all_five_classes_always_appear_in_the_report():
    y = np.array([0, 0, 0])

    result = evaluate(y, y)

    assert len(result.per_class) == 5
    assert tuple(m.name for m in result.per_class) == config.AAMI_CLASSES


def test_empty_test_set_is_rejected():
    with pytest.raises(ValueError, match="empty test set"):
        evaluate(np.array([]), np.array([]))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate(np.array([0, 1]), np.array([0]))


def test_report_leads_with_recall_and_flags_accuracy():
    y_true = np.array([0] * 86 + [2] * 14)
    y_pred = np.zeros(100, dtype=np.int64)

    report = format_report(evaluate(y_true, y_pred, label="test run"))

    assert "recall" in report
    assert "uninformative" in report
    assert report.index("macro-average recall") < report.index("accuracy  ")


def test_json_round_trip(tmp_path):
    y = np.array([0, 1, 2, 3, 4])
    result = evaluate(y, y, label="run")

    path = write_json([result], tmp_path / "metrics.json", extra={"split": "ds1_ds2"})
    payload = json.loads(path.read_text())

    assert payload["split"] == "ds1_ds2"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["label"] == "run"
    assert payload["results"][0]["confusion_matrix_note"].startswith("rows = true")
