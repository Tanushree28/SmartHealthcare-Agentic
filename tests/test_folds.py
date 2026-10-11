"""Record-grouped CV folds over DS1. The leakage guarantees are the point."""

from __future__ import annotations

import json

import pytest

from ecg import config, folds, splits


def test_folds_partition_ds1_exactly():
    assignment = folds.make_folds()

    flat = [r for fold in assignment.folds for r in fold]

    assert sorted(flat) == sorted(splits.DS1_RECORDS)
    assert len(flat) == len(set(flat))  # no record in two folds


def test_five_folds_over_twenty_two_records_are_balanced():
    assignment = folds.make_folds(n_folds=5)

    assert sorted(len(f) for f in assignment.folds) == [4, 4, 4, 5, 5]


def test_no_ds2_record_can_enter_a_fold():
    assignment = folds.make_folds()

    flat = {r for fold in assignment.folds for r in fold}

    assert not (flat & set(splits.DS2_RECORDS))


def test_validate_rejects_a_ds2_record():
    bad = folds.FoldAssignment(folds=(("101",), ("100",)), seed=1)

    with pytest.raises(ValueError, match="DS2"):
        bad.validate()


def test_validate_rejects_a_record_in_two_folds():
    bad = folds.FoldAssignment(folds=(("101", "106"), ("106",)), seed=1)

    with pytest.raises(ValueError, match="two folds"):
        bad.validate()


def test_validate_rejects_a_paced_record():
    bad = folds.FoldAssignment(folds=(("101",), ("102",)), seed=1)

    with pytest.raises(ValueError, match="paced"):
        bad.validate()


def test_train_val_split_is_complementary():
    assignment = folds.make_folds()

    for i in range(assignment.n_folds):
        train, val = assignment.train_val(i)
        assert set(val) == set(assignment.folds[i])
        assert not (set(train) & set(val))
        assert sorted(set(train) | set(val)) == sorted(splits.DS1_RECORDS)


def test_every_record_is_validated_exactly_once_across_the_folds():
    """The whole reason for cross-validating: all 944 DS1 S beats get used."""
    assignment = folds.make_folds()

    validated = [r for i in range(assignment.n_folds) for r in assignment.train_val(i)[1]]

    assert sorted(validated) == sorted(splits.DS1_RECORDS)


def test_train_val_rejects_an_out_of_range_fold():
    assignment = folds.make_folds()

    with pytest.raises(IndexError):
        assignment.train_val(assignment.n_folds)


def test_fold_assignment_is_reproducible_from_the_seed():
    assert folds.make_folds(seed=7).folds == folds.make_folds(seed=7).folds


def test_a_different_seed_gives_a_different_assignment():
    assert folds.make_folds(seed=7).folds != folds.make_folds(seed=8).folds


def test_input_order_does_not_change_the_assignment():
    """Sorting before shuffling: the result depends on the seed and the set."""
    forward = folds.make_folds(records=splits.DS1_RECORDS, seed=3)
    backward = folds.make_folds(records=tuple(reversed(splits.DS1_RECORDS)), seed=3)

    assert forward.folds == backward.folds


def test_write_json_round_trips(tmp_path):
    assignment = folds.make_folds()

    path = assignment.write_json(tmp_path / "folds.json")
    payload = json.loads(path.read_text())

    assert payload["seed"] == config.PHASE_1D_SEED
    assert payload["group_unit"] == "record"
    assert len(payload["folds"]) == 5


def test_too_many_folds_is_rejected():
    with pytest.raises(ValueError, match="more folds than records"):
        folds.make_folds(records=["101", "106"], n_folds=5)


def test_fewer_than_two_folds_is_rejected():
    with pytest.raises(ValueError, match="at least 2 folds"):
        folds.make_folds(n_folds=1)
