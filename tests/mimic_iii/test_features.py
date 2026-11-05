"""Windows, units, joins and temporal leakage in the feature table.

The expected numbers are worked out by hand from ``synthetic.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mimic_iii import config, features
from mimic_iii.errors import IntegrityError, LeakageError, SchemaError

from .conftest import load_cohort
from .synthetic import T0, build_tables, write_database


@pytest.fixture
def ref(feature_table):
    """The reference stay, subject 1 / icustay 101."""
    return feature_table.set_index("icustay_id").loc[101]


def test_heart_rate_uses_only_the_observation_window(ref):
    # In window: 80, 90, 100. Out: 200 before intime, 70 error-flagged,
    # 150 exactly at the index time, 160 after it.
    assert ref["heart_rate_count"] == 3
    assert ref["heart_rate_mean"] == pytest.approx(90.0)
    assert ref["heart_rate_min"] == 80
    assert ref["heart_rate_max"] == 100
    assert ref["heart_rate_last"] == 100


def test_fahrenheit_is_converted_to_celsius(ref):
    assert ref["temp_c_mean"] == pytest.approx(37.0)


def test_implausible_value_is_dropped(ref):
    assert ref["sbp_count"] == 1
    assert ref["sbp_mean"] == 120


def test_labs_use_the_lookback_window(ref):
    # 9.0 at -7 h is before the 6 h lookback; 7.0 at +26 h is after the index.
    assert ref["creatinine_count"] == 2
    assert ref["creatinine_mean"] == pytest.approx(1.5)
    assert ref["creatinine_last"] == 2.0


def test_lab_with_wrong_unit_is_dropped_not_rescaled(ref):
    assert ref["potassium_count"] == 0
    assert ref["potassium_missing"] == 1
    assert np.isnan(ref["potassium_mean"])


def test_lab_without_admission_id_is_kept(ref):
    assert ref["sodium_mean"] == 140


def test_lab_from_another_admission_is_dropped(ref):
    assert ref["glucose_missing"] == 1


def test_unit_matching_ignores_case(ref):
    assert ref["troponin_t_mean"] == pytest.approx(0.5)  # charted as "ng/ml"


def test_row_without_a_value_counts_as_missing(ref):
    assert ref["spo2_missing"] == 1


def test_stay_with_no_events_is_all_missing(feature_table):
    row = feature_table.set_index("icustay_id").loc[401]

    for v in config.VARIABLE_NAMES:
        assert row[f"{v}_count"] == 0
        assert row[f"{v}_missing"] == 1
        assert np.isnan(row[f"{v}_mean"])


def test_events_after_the_index_time_cannot_change_any_feature(tmp_path, tables, feature_table):
    """The temporal-leakage test: rewrite everything at/after the index and rebuild."""
    icu = tables["ICUSTAYS"]
    first = icu.sort_values("intime").groupby("subject_id")["intime"].first()
    index_by_subject = pd.to_datetime(first) + pd.Timedelta(hours=24)
    for name in ("CHARTEVENTS", "LABEVENTS"):
        df = tables[name]
        index_time = df["subject_id"].map(index_by_subject)
        late = pd.to_datetime(df["charttime"]) >= index_time
        assert late.any() and (~late).any()
        df.loc[late, "valuenum"] = df.loc[late, "valuenum"] * 1.7 + 3
    extra = tables["CHARTEVENTS"].iloc[[1]].copy()
    extra["charttime"] = (T0 + pd.Timedelta(hours=24, seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    extra["valuenum"] = 299.0
    tables["CHARTEVENTS"] = pd.concat([tables["CHARTEVENTS"], extra], ignore_index=True)
    root = write_database(tmp_path / "late", tables)

    result = load_cohort(root)
    events, _ = features.extract_events(root, result.cohort)
    rebuilt = features.build_feature_table(result.cohort, events)

    pd.testing.assert_frame_equal(rebuilt, feature_table)


def test_outcome_changes_only_the_label(tmp_path, tables, feature_table):
    """Flipping a death to survival must not move a single feature."""
    adm = tables["ADMISSIONS"]
    row = adm["hadm_id"] == 21
    adm.loc[row, ["deathtime", "hospital_expire_flag", "discharge_location"]] = [None, 0, "HOME"]
    root = write_database(tmp_path / "flip", tables)

    result = load_cohort(root)
    events, _ = features.extract_events(root, result.cohort)
    rebuilt = features.build_feature_table(result.cohort, events)

    diff = rebuilt.compare(feature_table)
    assert list(diff.columns.get_level_values(0).unique()) == ["label"]


def test_chart_row_with_wrong_subject_is_dropped(tmp_path, tables):
    ce = tables["CHARTEVENTS"]
    bad = ce.iloc[[1]].copy()  # an in-window heart rate of stay 101 ...
    bad["subject_id"] = 2  # ... recorded against another patient
    bad["valuenum"] = 250.0
    tables["CHARTEVENTS"] = pd.concat([ce, bad], ignore_index=True)
    root = write_database(tmp_path / "db", tables)

    result = load_cohort(root)
    events, stats = features.extract_events(root, result.cohort)
    table = features.build_feature_table(result.cohort, events)

    assert stats.dropped["chart:subject_mismatch"] == 1
    assert table.set_index("icustay_id").loc[101, "heart_rate_max"] == 100


def test_drop_accounting_names_every_reason(mimic_dir, cohort_result):
    _, stats = features.extract_events(mimic_dir, cohort_result.cohort)
    d = stats.to_dict()

    assert d["dropped"]["lab:unit_mismatch:potassium"] == 1
    assert d["dropped"]["lab:other_admission"] == 1
    assert d["dropped"]["out_of_range:sbp"] == 1
    assert d["converted"]["temp_c:fahrenheit_to_celsius"] == 1
    assert d["chart_ingest"]["rows_error_flag"] == 1


def test_feature_table_has_one_row_per_cohort_stay(cohort_result, feature_table):
    assert len(feature_table) == len(cohort_result.cohort)
    assert feature_table["icustay_id"].is_unique


def test_model_inputs_contain_no_leakage_or_identifiers():
    cols = features.model_input_columns()

    features.assert_no_leakage(cols)
    assert not set(cols) & config.POST_INDEX_COLUMNS
    assert not set(cols) & set(config.ID_COLUMNS)


@pytest.mark.parametrize("col", ["hospital_expire_flag", "icustay_id", "los", "index_time", "label"])
def test_leakage_guard_rejects(col):
    with pytest.raises(LeakageError):
        features.assert_no_leakage(features.model_input_columns() + [col])


def test_events_for_unknown_stays_are_rejected(cohort_result):
    events = pd.DataFrame({"icustay_id": [999_999], "variable": ["heart_rate"],
                           "charttime": [pd.Timestamp(T0)], "value": [80.0]})

    with pytest.raises(IntegrityError):
        features.build_feature_table(cohort_result.cohort, events)


def test_feature_table_round_trips_through_csv(tmp_path, feature_table):
    path = tmp_path / "features.csv.gz"
    feature_table.to_csv(path, index=False)

    back = features.read_feature_table(path)

    pd.testing.assert_frame_equal(back, feature_table, check_dtype=False)


def test_feature_table_from_another_code_version_is_refused(tmp_path, feature_table):
    path = tmp_path / "features.csv.gz"
    feature_table.drop(columns="heart_rate_mean").to_csv(path, index=False)

    with pytest.raises(SchemaError, match="heart_rate_mean"):
        features.read_feature_table(path)


def test_lab_windows_attribute_each_lab_to_its_own_stay():
    """Two stays of one patient: a lab belongs to the stay whose window holds it."""
    cohort = pd.DataFrame({
        "icustay_id": [1, 2], "subject_id": [9, 9], "hadm_id": [90, 91],
        "intime": pd.to_datetime(["2150-01-01 00:00", "2150-03-01 00:00"]),
        "index_time": pd.to_datetime(["2150-01-02 00:00", "2150-03-02 00:00"]),
    })
    labs = pd.DataFrame({
        "subject_id": [9, 9], "hadm_id": pd.array([90, 91], dtype="Int64"),
        "itemid": [50912, 50912],
        "charttime": pd.to_datetime(["2150-01-01 05:00", "2150-03-01 05:00"]),
        "valuenum": [1.0, 3.0], "valueuom": ["mg/dL", "mg/dL"],
    })

    out = features.window_lab_events(labs, cohort, config.TaskSpec(), features.EventStats())

    assert sorted(zip(out["icustay_id"], out["value"])) == [(1, 1.0), (2, 3.0)]


def test_synthetic_tables_are_not_real_data():
    """Guard: the fixture must stay obviously synthetic (future-dated, tiny)."""
    t = build_tables()
    assert len(t["PATIENTS"]) < 100
    assert pd.to_datetime(t["ICUSTAYS"]["intime"]).dt.year.min() >= 2150
