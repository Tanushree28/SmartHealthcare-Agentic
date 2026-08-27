"""Group E -- splits (PlanDoc/PLAN.md). PDF 2.8 calls this out as critical."""

from __future__ import annotations

import json

import pytest

from ecg import config
from ecg.splits import DS1_RECORDS, DS2_RECORDS, ds1_ds2_split, record_wise_split

# Verified 2026-08-26 against two independent published sources restating
# de Chazal et al. (2004) Section II-B:
#   https://arxiv.org/html/2503.07276v1
#   https://par.nsf.gov/servlets/purl/10086899
# Both give identical lists. Written out here as an independent expectation so
# the test does not check the module against itself.
REFERENCE_DS1 = {
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
}
REFERENCE_DS2 = {
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
}


def test_e1_no_record_appears_in_more_than_one_split():
    """E1: *the* leakage test. This failure mode produces meaningless accuracy."""
    split = record_wise_split(config.ALL_RECORDS)

    train, val, test = set(split.train), set(split.val), set(split.test)

    assert train & val == set()
    assert train & test == set()
    assert val & test == set()

    eligible = set(config.ALL_RECORDS) - set(config.PACED_RECORDS)
    assert train | val | test == eligible


def test_e1b_ds1_ds2_has_no_record_overlap():
    split = ds1_ds2_split()

    assert set(split.train) & set(split.test) == set()
    assert set(split.val) & set(split.test) == set()
    assert set(split.train) & set(split.val) == set()


def test_e2_beat_level_splitting_is_not_offered():
    """E2: the constraint is encoded in the API, not in a comment."""
    with pytest.raises(TypeError):
        record_wise_split(config.ALL_RECORDS, split_unit="beat")

    with pytest.raises(TypeError):
        ds1_ds2_split(split_unit="beat")

    with pytest.raises(TypeError):
        record_wise_split(config.ALL_RECORDS, by="beat")


def test_e3_paced_records_excluded_by_default_and_listed():
    split = record_wise_split(config.ALL_RECORDS)

    assigned = set(split.train) | set(split.val) | set(split.test)
    for record in ("102", "104", "107", "217"):
        assert record not in assigned
        assert record in split.excluded

    assert set(split.excluded) == set(config.PACED_RECORDS)


def test_e3b_paced_records_can_be_included_explicitly():
    split = record_wise_split(config.ALL_RECORDS, exclude_paced=False)

    assigned = set(split.train) | set(split.val) | set(split.test)
    assert set(config.PACED_RECORDS) <= assigned
    assert split.excluded == ()


def test_e4_ds1_ds2_matches_the_verified_reference_lists():
    """E4: the point of DS1/DS2 is comparability; an approximate list is worthless."""
    assert set(DS1_RECORDS) == REFERENCE_DS1
    assert set(DS2_RECORDS) == REFERENCE_DS2
    assert len(DS1_RECORDS) == len(DS2_RECORDS) == 22
    assert len(set(DS1_RECORDS) | set(DS2_RECORDS)) == 44


def test_e4b_ds1_ds2_union_is_exactly_the_non_paced_records():
    expected = set(config.ALL_RECORDS) - set(config.PACED_RECORDS)
    assert set(DS1_RECORDS) | set(DS2_RECORDS) == expected


def test_e4c_ds2_is_never_used_for_validation():
    """Carving validation out of DS2 would contaminate the published comparison."""
    split = ds1_ds2_split()

    assert set(split.test) == set(DS2_RECORDS)
    assert set(split.val) <= set(DS1_RECORDS)
    assert set(split.train) <= set(DS1_RECORDS)
    assert set(split.train) | set(split.val) == set(DS1_RECORDS)


def test_e5_seeded_splits_are_reproducible_and_seeds_differ():
    a = record_wise_split(config.ALL_RECORDS, seed=7)
    b = record_wise_split(config.ALL_RECORDS, seed=7)
    c = record_wise_split(config.ALL_RECORDS, seed=8)

    assert a == b
    assert (a.train, a.val, a.test) != (c.train, c.val, c.test)


def test_e6_every_split_is_non_empty_for_the_standard_44_records():
    """Guards a config change that would starve validation."""
    split = record_wise_split(config.ALL_RECORDS)

    assert len(split.train) > 0
    assert len(split.val) > 0
    assert len(split.test) > 0
    assert len(split.train) + len(split.val) + len(split.test) == 44


def test_small_record_sets_still_produce_three_non_empty_splits():
    split = record_wise_split(["901", "902", "903"])

    assert len(split.train) >= 1
    assert len(split.val) >= 1
    assert len(split.test) >= 1


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        record_wise_split(config.ALL_RECORDS, fractions=(0.5, 0.2, 0.2))


def test_empty_record_set_raises():
    with pytest.raises(ValueError, match="no eligible records"):
        record_wise_split(config.PACED_RECORDS)


def test_ds1_ds2_restricted_to_available_records():
    available = list(DS1_RECORDS[:5]) + list(DS2_RECORDS[:5])
    split = ds1_ds2_split(records=available)

    assert set(split.test) == set(DS2_RECORDS[:5])
    assert set(split.train) | set(split.val) == set(DS1_RECORDS[:5])


def test_split_json_round_trip_records_the_unit(tmp_path):
    split = record_wise_split(config.ALL_RECORDS)
    path = split.write_json(tmp_path / "splits.json")

    payload = json.loads(path.read_text())

    assert payload["split_unit"] == "record"
    assert payload["strategy"] == "record_wise"
    assert sorted(payload["excluded"]) == sorted(config.PACED_RECORDS)


def test_split_of_reports_membership():
    split = record_wise_split(config.ALL_RECORDS)

    assert split.split_of(split.train[0]) == "train"
    assert split.split_of(split.test[0]) == "test"
    assert split.split_of("102") is None
