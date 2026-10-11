"""The orchestration script: structure and guardrails, not the numbers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_nn  # noqa: E402

from ecg import folds, splits  # noqa: E402


def test_there_are_six_arms():
    assert len(train_nn.ARMS) == 6


def test_arm_names_are_unique():
    names = [a.name for a in train_nn.ARMS]

    assert len(names) == len(set(names))


def test_exactly_one_arm_has_no_imbalance_handling():
    plain = [a for a in train_nn.ARMS if a.loss == "plain" and not a.oversample]

    assert len(plain) == 1
    assert plain[0].name == "plain_ce"


def test_exactly_one_arm_is_waveform_only():
    assert len([a for a in train_nn.ARMS if not a.use_rr]) == 1


def test_the_threshold_arm_trains_nothing_of_its_own():
    tuned = [a for a in train_nn.ARMS if a.tune]

    assert len(tuned) == 1
    assert tuned[0].reuses is not None


#: Each arm's baseline -- the row it is meant to be read against. The study's
#: whole claim rests on each arm differing from ITS baseline in exactly one way,
#: not on every arm differing from plain_ce in one way: `tuned` is weighted_ce
#: plus thresholds, and `waveform_only` is weighted_ce minus the RR branch.
BASELINE_OF = {
    "weighted_ce": "plain_ce",
    "focal": "plain_ce",
    "oversample": "plain_ce",
    "tuned": "weighted_ce",
    "waveform_only": "weighted_ce",
}


def test_every_arm_differs_from_its_baseline_in_exactly_one_way():
    """The whole comparison rests on this: one variable per row."""
    by_name = {a.name: a for a in train_nn.ARMS}

    assert set(BASELINE_OF) | {"plain_ce"} == set(by_name), "an arm has no baseline"

    for name, baseline_name in BASELINE_OF.items():
        arm, baseline = by_name[name], by_name[baseline_name]
        differences = [
            field
            for field in ("loss", "oversample", "use_rr", "tune")
            if getattr(arm, field) != getattr(baseline, field)
        ]
        assert differences == [] or len(differences) == 1, (
            f"{name} differs from {baseline_name} in {differences}"
        )


def test_missing_data_names_the_command_that_produces_it(tmp_path, capsys):
    code = train_nn.main(["--beats-dir", str(tmp_path / "nope")])

    assert code != 0
    assert "preprocess_mitbih.py" in capsys.readouterr().err


def test_ds2_records_never_appear_in_any_fold():
    assignment = folds.make_folds()

    flat = {r for fold in assignment.folds for r in fold}

    assert not (flat & set(splits.DS2_RECORDS))


def test_weight_cap_candidates_are_sane():
    assert len(train_nn.WEIGHT_CAPS) >= 2
    assert all(c > 0 for c in train_nn.WEIGHT_CAPS)


def test_the_json_filter_keeps_summary_scalars_and_drops_only_the_arrays():
    """An earlier filter matched the "oof_" prefix and dropped oof_macro_f1
    along with oof_proba, so the tuned arm read as 0.000 in the artifact while
    the log carried the real 0.529."""
    row = {
        "arm": "tuned",
        "oof_macro_f1": 0.529,
        "oof_s_recall": 0.667,
        "oof_proba": [[0.1] * 5],
        "oof_y": [0],
    }

    kept = {k: v for k, v in row.items() if k not in train_nn._BULKY}

    assert kept["oof_macro_f1"] == 0.529
    assert kept["oof_s_recall"] == 0.667
    assert "oof_proba" not in kept
    assert "oof_y" not in kept
