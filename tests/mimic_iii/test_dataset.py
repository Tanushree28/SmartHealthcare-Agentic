"""Patient-level splits: no patient on two sides, ever."""

from __future__ import annotations

import pandas as pd
import pytest

from mimic_iii import config, dataset
from mimic_iii.errors import LeakageError


def _multi_stay_table(n_subjects=40, stays_per_subject=3):
    rows = []
    for s in range(n_subjects):
        for k in range(stays_per_subject):
            rows.append({"subject_id": 500 + s, "icustay_id": 10 * s + k,
                         "label": int(s % 4 == 0 and k == stays_per_subject - 1)})
    return pd.DataFrame(rows)


def test_no_subject_in_two_splits(feature_table):
    split = dataset.patient_split(feature_table)

    tr, va, te = set(split.train), set(split.val), set(split.test)
    assert not (tr & va or tr & te or va & te)
    assert tr | va | te == set(feature_table["subject_id"])


def test_all_stays_of_a_patient_land_together():
    table = _multi_stay_table()
    split = dataset.patient_split(table, seed=3)
    parts = dataset.split_frame(table, split)

    seen = {}
    for name, df in parts.items():
        for s in df["subject_id"]:
            assert seen.setdefault(s, name) == name
    assert sum(len(df) for df in parts.values()) == len(table)


def test_split_is_deterministic_for_a_seed(feature_table):
    a = dataset.patient_split(feature_table, seed=11)
    b = dataset.patient_split(feature_table, seed=11)
    c = dataset.patient_split(feature_table, seed=12)

    assert a == b
    assert a.to_dict() != c.to_dict()


def test_every_split_gets_positive_patients(feature_table):
    parts = dataset.split_frame(feature_table, dataset.patient_split(feature_table))

    for name, df in parts.items():
        assert df["label"].sum() > 0, name
        assert (df["label"] == 0).sum() > 0, name


def test_patient_label_is_positive_if_any_stay_is():
    table = _multi_stay_table(n_subjects=40)
    parts = dataset.split_frame(table, dataset.patient_split(table, seed=1))

    positives = {n: df.groupby("subject_id")["label"].max().sum() for n, df in parts.items()}
    assert all(v > 0 for v in positives.values())


def test_overlapping_split_cannot_be_constructed():
    with pytest.raises(LeakageError):
        dataset.PatientSplit(train=(1, 2), val=(2,), test=(3,), seed=0, fractions=(0.7, 0.15, 0.15))


def test_split_with_a_non_patient_unit_is_refused():
    payload = dataset.patient_split(_multi_stay_table()).to_dict()
    payload["unit"] = "icustay_id"

    with pytest.raises(LeakageError, match="subject_id"):
        dataset.PatientSplit.from_dict(payload)


def test_table_with_unassigned_subject_is_refused(feature_table):
    split = dataset.patient_split(feature_table)
    extra = feature_table.iloc[[0]].copy()
    extra["subject_id"] = 999_999

    with pytest.raises(LeakageError, match="no split assignment"):
        dataset.split_frame(pd.concat([feature_table, extra]), split)


def test_too_few_subjects_is_an_error():
    table = pd.DataFrame({"subject_id": [1, 2, 3], "label": [0, 0, 1]})

    with pytest.raises(ValueError, match="too few"):
        dataset.patient_split(table)


def test_split_round_trips_through_json(tmp_path, feature_table):
    split = dataset.patient_split(feature_table, feature_table_sha256="abc")
    path = dataset.write_split(tmp_path / "splits.json", split, feature_table)

    assert dataset.read_split(path) == split


def test_xy_never_contains_identifiers_or_label(feature_table):
    X, y = dataset.xy(feature_table)

    assert not set(X.columns) & set(config.ID_COLUMNS)
    assert config.LABEL_COLUMN not in X.columns
    assert y.tolist() == feature_table["label"].tolist()
