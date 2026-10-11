"""Phase 1e orchestration: structure and guardrails."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_nn_v2  # noqa: E402

from ecg import splits  # noqa: E402


def test_there_are_four_arms():
    assert len(train_nn_v2.ARMS) == 4


def test_the_reference_arm_reproduces_phase_1d():
    ref = next(a for a in train_nn_v2.ARMS if a.name == train_nn_v2.REFERENCE)

    assert ref.model == "beat"
    assert ref.robust_rr is False
    assert ref.loss == "focal"


def test_each_single_change_arm_differs_from_the_reference_in_one_way():
    ref = next(a for a in train_nn_v2.ARMS if a.name == train_nn_v2.REFERENCE)

    for name in ("beat_robust_rr", "rhythm_context"):
        arm = next(a for a in train_nn_v2.ARMS if a.name == name)
        differences = [
            f for f in ("model", "robust_rr", "loss")
            if getattr(arm, f) != getattr(ref, f)
        ]
        assert len(differences) == 1, f"{name} differs in {differences}"


def test_the_combined_arm_changes_exactly_the_two_things():
    ref = next(a for a in train_nn_v2.ARMS if a.name == train_nn_v2.REFERENCE)
    both = next(a for a in train_nn_v2.ARMS if a.name == "rhythm_context_robust")

    assert both.model != ref.model
    assert both.robust_rr != ref.robust_rr
    assert both.loss == ref.loss


def test_arm_names_are_unique():
    names = [a.name for a in train_nn_v2.ARMS]

    assert len(names) == len(set(names))


def test_context_window_is_odd_so_there_is_a_centre_beat():
    assert (2 * train_nn_v2.CONTEXT + 1) % 2 == 1


def test_missing_data_names_the_command_that_produces_it(tmp_path, capsys):
    code = train_nn_v2.main(["--beats-dir", str(tmp_path / "nope")])

    assert code != 0
    assert "preprocess_mitbih.py" in capsys.readouterr().err


def test_bulky_arrays_are_dropped_but_summaries_kept():
    row = {"mean_s_f1": 0.4, "oof_proba": [[0.2] * 5], "oof_y": [1]}

    kept = {k: v for k, v in row.items() if k not in train_nn_v2._BULKY}

    assert kept == {"mean_s_f1": 0.4}


def test_ds2_records_never_enter_a_fold():
    from ecg import folds

    assignment = folds.make_folds()
    flat = {r for fold in assignment.folds for r in fold}

    assert not (flat & set(splits.DS2_RECORDS))
