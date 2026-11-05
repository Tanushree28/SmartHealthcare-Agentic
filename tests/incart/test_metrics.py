"""Metrics from confusion matrices, the patient bootstrap, the rules and the guards."""

from __future__ import annotations

import json

import numpy as np
import pytest

from incart import evaluate as ev
from mit_bih import stats
from mit_bih.config import HEADLINE_CLASSES
from mit_bih.evaluate import evaluate


def _ten_perfect_patients_and_one_all_wrong():
    rng = np.random.default_rng(0)
    matrices, ys, ps = {}, [], []
    for i in range(10):
        y = rng.integers(0, 4, 200)
        matrices[f"P{i:02d}"] = ev.confusion(y, y)
        ys.append(y), ps.append(y)
    y = rng.integers(0, 4, 200)
    p = (y + 1) % 4
    matrices["P10"] = ev.confusion(y, p)
    ys.append(y), ps.append(p)
    return matrices, np.concatenate(ys), np.concatenate(ps)


def test_confusion_counts_true_rows_and_predicted_columns():
    m = ev.confusion([0, 0, 2, 3], [0, 1, 2, 2])

    assert m.shape == (5, 5)
    assert m[0, 0] == 1 and m[0, 1] == 1 and m[2, 2] == 1 and m[3, 2] == 1
    assert m.sum() == 4


def test_matrix_f1_matches_evaluate_with_an_absent_class():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)  # no F and no Q anywhere
    p = rng.integers(0, 3, 500)

    expected = evaluate(y, p).macro_f1_of(HEADLINE_CLASSES)

    assert ev.headline_f1_from_matrix(ev.confusion(y, p)) == pytest.approx(expected)


def test_matrix_f1_works_on_a_stack_of_matrices():
    a = ev.confusion([0, 1, 2, 3], [0, 1, 2, 3])  # perfect
    b = ev.confusion([0, 1, 2, 3], [1, 2, 3, 0])  # all wrong

    assert ev.headline_f1_from_matrix(np.stack([a, b])).tolist() == [1.0, 0.0]


def test_every_patient_perfect_gives_a_point_interval():
    y = np.arange(4).repeat(10)
    matrices = {f"P{i}": ev.confusion(y, y) for i in range(5)}

    assert ev.grouped_bootstrap_ci(matrices, n_boot=200, seed=1) == (1.0, 1.0)


def test_one_all_wrong_patient_widens_the_interval_beyond_beat_resampling():
    matrices, y_all, p_all = _ten_perfect_patients_and_one_all_wrong()

    lo, hi = ev.grouped_bootstrap_ci(matrices, n_boot=1000, seed=1)
    blo, bhi = stats.metric_ci(y_all, p_all, n_boot=200, seed=1)

    assert lo < 1.0
    assert (hi - lo) > 2 * (bhi - blo)


def test_bootstrap_is_reproducible_with_the_seed():
    matrices, _, _ = _ten_perfect_patients_and_one_all_wrong()

    first = ev.grouped_bootstrap_ci(matrices, n_boot=300, seed=7)

    assert ev.grouped_bootstrap_ci(matrices, n_boot=300, seed=7) == first


def test_sum_by_group_adds_each_patients_records():
    a = ev.confusion([0], [0])
    b = ev.confusion([1], [1])

    grouped = ev.sum_by_group({"I01": a, "I02": b, "I03": a}, {"I01": "P01", "I02": "P01", "I03": "P02"})

    assert grouped["P01"].sum() == 2 and grouped["P01"][1, 1] == 1
    assert grouped["P02"].sum() == 1


def test_decision_rules():
    assert ev.holds_up(0.5554, (0.50, 0.60))
    assert ev.holds_up(0.5554, (0.5554, 0.60))  # inclusive
    assert not ev.holds_up(0.5554, (0.56, 0.60))
    assert not ev.resampling_matters(0.5554, 0.5604)
    assert ev.resampling_matters(0.5554, 0.5404)
    assert not ev.position_rule_fires(0, 2.5)
    assert ev.position_rule_fires(-1, 3)


def test_record_row_with_no_beats():
    assert ev.record_row("I01", [], []) == {
        "record": "I01", "n_beats": 0, "headline_macro_f1": None, "metrics": None}


def test_record_row_with_beats():
    row = ev.record_row("I02", [0, 0, 2, 2], [0, 0, 2, 0])

    assert row["n_beats"] == 4
    assert row["headline_macro_f1"] == pytest.approx((0.8 + 0 + 2 / 3 + 0) / 4)
    assert row["metrics"]["n_test"] == 4


def test_missing_resampling_check_is_refused(tmp_path):
    with pytest.raises(ev.PreconditionError, match="resampling_check.py"):
        ev.read_resampling_verdict(tmp_path / "resampling_check.json")


def test_failed_reproduction_is_refused(tmp_path):
    path = tmp_path / "resampling_check.json"
    path.write_text(json.dumps({"reproduces_published": False}))

    with pytest.raises(ev.PreconditionError, match="0.5554"):
        ev.read_resampling_verdict(path)


def test_passing_resampling_check_is_returned(tmp_path):
    path = tmp_path / "resampling_check.json"
    path.write_text(json.dumps({"reproduces_published": True,
                                "original": {"headline_macro_f1": 0.5554}}))

    assert ev.read_resampling_verdict(path)["original"]["headline_macro_f1"] == 0.5554


def test_existing_output_is_not_overwritten(tmp_path):
    out = tmp_path / "external_test.json"
    out.write_text("{}")

    with pytest.raises(ev.PreconditionError, match="--overwrite"):
        ev.refuse_to_overwrite(out, overwrite=False)
    ev.refuse_to_overwrite(out, overwrite=True)  # allowed when asked for
    ev.refuse_to_overwrite(tmp_path / "absent.json", overwrite=False)


def test_verdict_for_another_ensemble_is_refused(tmp_path):
    path = tmp_path / "resampling_check.json"
    path.write_text(json.dumps({"reproduces_published": True, "ensemble": "A_focal_only",
                                "original": {"headline_macro_f1": 0.6119}}))

    with pytest.raises(ev.PreconditionError, match="A_focal_only"):
        ev.read_resampling_verdict(path, ensemble="D_all_four")
    assert ev.read_resampling_verdict(path, ensemble="A_focal_only")["ensemble"] == "A_focal_only"
