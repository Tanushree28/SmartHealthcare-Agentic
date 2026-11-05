"""Binary metrics: correct on known cases, explicit about the undefined ones."""

from __future__ import annotations

import json

import numpy as np
import pytest

from common import metrics, provenance


def test_known_confusion_matrix():
    y = [1, 1, 0, 0, 0]
    s = [0.9, 0.4, 0.6, 0.2, 0.1]

    r = metrics.binary_report(y, s, threshold=0.5)

    assert r["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 2, "fn": 1}
    assert r["sensitivity"] == 0.5
    assert r["specificity"] == pytest.approx(2 / 3)
    assert r["ppv"] == 0.5
    assert r["auroc"] == pytest.approx(5 / 6)
    assert r["undefined"] == {}


def test_one_class_makes_ranking_metrics_undefined_not_zero():
    r = metrics.binary_report([0, 0, 0], [0.1, 0.2, 0.3])

    assert r["auroc"] is None and r["auprc"] is None
    assert "one class" in r["undefined"]["auroc"]
    assert r["sensitivity"] is None


def test_no_positive_predictions_makes_ppv_undefined():
    r = metrics.binary_report([1, 0], [0.1, 0.2], threshold=0.9)

    assert r["ppv"] is None
    assert "ppv" in r["undefined"]


@pytest.mark.parametrize("y,s", [
    ([0, 1], [0.1]),
    ([0, 2], [0.1, 0.2]),
    ([0, 1], [0.1, 1.5]),
    ([0, 1], [0.1, float("nan")]),
    ([], []),
])
def test_malformed_input_is_rejected(y, s):
    with pytest.raises(ValueError):
        metrics.binary_report(y, s)


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)

    ci = metrics.bootstrap_ci(y, s, "auroc", n_boot=300, seed=1)
    point = metrics.binary_report(y, s)["auroc"]

    assert ci["lower"] <= point <= ci["upper"]
    assert ci["n_valid"] == 300


def test_bootstrap_on_tiny_set_reports_why_it_has_no_interval():
    ci = metrics.bootstrap_ci([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9], n_boot=50)

    assert ci["lower"] is None
    assert "resamples" in ci["reason"]


def test_reliability_bins_cover_every_case_including_score_one():
    y = [0, 1, 1, 0, 1]
    s = [0.0, 0.5, 1.0, 0.19, 0.99]

    bins = metrics.reliability_bins(y, s, n_bins=5)

    assert sum(b["n"] for b in bins) == 5
    assert bins[-1]["n"] == 2


def test_youden_threshold_separates_a_separable_set():
    t, note = metrics.choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert 0.2 < t <= 0.8
    assert "youden" in note


def test_threshold_on_one_class_defaults_with_a_reason():
    t, note = metrics.choose_threshold([0, 0], [0.3, 0.4])

    assert t == 0.5
    assert "one class" in note


def test_json_writer_turns_nan_into_null(tmp_path):
    path = provenance.write_json(tmp_path / "x.json", {"a": float("nan"), "b": np.float64(1.5),
                                                       "c": np.array([1, 2])})

    data = json.loads(path.read_text())
    assert data == {"a": None, "b": 1.5, "c": [1, 2]}


def test_provenance_records_environment_and_input_hashes(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("hello")

    block = provenance.provenance({"input": f})

    assert block["inputs"]["input"]["sha256"] == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
    assert "python" in block["environment"]
    assert block["generated_utc"].endswith("+00:00")
