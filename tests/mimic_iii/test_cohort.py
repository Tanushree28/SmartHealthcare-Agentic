"""Cohort rules, labels, index times and join validation."""

from __future__ import annotations

import pandas as pd
import pytest

from mimic_iii import cohort, config
from mimic_iii.errors import IntegrityError, SchemaError

from .conftest import load_cohort
from .synthetic import EXCLUDED_STAYS, EXPECTED_FIXED_STAYS, T0, write_database


def test_reference_stays_are_included_with_their_labels(cohort_result):
    df = cohort_result.cohort.set_index("icustay_id")

    for stay, label in EXPECTED_FIXED_STAYS.items():
        assert stay in df.index
        assert df.loc[stay, "label"] == label


def test_every_exclusion_rule_removes_its_stay(cohort_result):
    """102 second stay, 301 child, 501 not cardiac, 601 died early, 701 left early."""
    assert not EXCLUDED_STAYS & set(cohort_result.cohort["icustay_id"])


def test_one_stay_per_patient(cohort_result):
    assert cohort_result.cohort["subject_id"].is_unique


def test_flow_only_shrinks_and_ends_at_the_cohort(cohort_result):
    counts = [row["stays"] for row in cohort_result.flow]

    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == len(cohort_result.cohort)


def test_index_time_is_intime_plus_observation_window(cohort_result):
    df = cohort_result.cohort

    assert ((df["index_time"] - df["intime"]) == pd.Timedelta(hours=24)).all()


def test_observation_hours_is_configurable(mimic_dir):
    """At 13 h the stay that left the ICU at 12 h is still excluded; at 6 h it is in."""
    short = load_cohort(mimic_dir, config.TaskSpec(observation_hours=6))

    assert 701 in set(short.cohort["icustay_id"])
    assert 601 not in set(load_cohort(mimic_dir, config.TaskSpec(observation_hours=13)).cohort["icustay_id"])


def test_shifted_age_over_89_is_recoded(cohort_result):
    row = cohort_result.cohort.set_index("icustay_id").loc[401]

    assert row["age"] == pytest.approx(91.4)
    assert row["age_over_89"] == 1


def test_age_is_whole_years_before_and_after_a_birthday():
    dob = pd.Series(pd.to_datetime(["2100-06-15", "2100-06-15"]))
    at = pd.Series(pd.to_datetime(["2150-06-14", "2150-06-15"]))

    assert cohort.age_in_years(dob, at).tolist() == [49.0, 50.0]


def test_cohort_has_no_outcome_or_diagnosis_columns_other_than_the_label(cohort_result):
    leaked = set(cohort_result.cohort.columns) & (config.POST_INDEX_COLUMNS - {"label"})

    assert leaked == set()


def test_admission_from_another_patient_is_rejected(mimic_dir):
    from mimic_iii import ingestion

    icu = ingestion.read_table(mimic_dir, "ICUSTAYS")
    icu.loc[icu["icustay_id"] == 101, "subject_id"] = 2  # stay 101 now claims subject 2

    with pytest.raises(IntegrityError, match="subject_id different"):
        cohort.build_cohort(
            ingestion.read_table(mimic_dir, "PATIENTS"),
            ingestion.read_table(mimic_dir, "ADMISSIONS"),
            icu, ingestion.read_table(mimic_dir, "DIAGNOSES_ICD"))


def test_orphan_icu_stay_is_rejected(mimic_dir):
    from mimic_iii import ingestion

    adm = ingestion.read_table(mimic_dir, "ADMISSIONS")
    adm = adm[adm["hadm_id"] != 21]

    with pytest.raises(IntegrityError, match="missing from ADMISSIONS"):
        cohort.build_cohort(
            ingestion.read_table(mimic_dir, "PATIENTS"), adm,
            ingestion.read_table(mimic_dir, "ICUSTAYS"),
            ingestion.read_table(mimic_dir, "DIAGNOSES_ICD"))


def test_duplicate_admission_key_is_rejected_at_the_join(mimic_dir):
    from mimic_iii import ingestion

    adm = ingestion.read_table(mimic_dir, "ADMISSIONS")
    adm = pd.concat([adm, adm.iloc[[0]]], ignore_index=True)

    with pytest.raises(IntegrityError, match="not unique"):
        cohort.build_cohort(
            ingestion.read_table(mimic_dir, "PATIENTS"), adm,
            ingestion.read_table(mimic_dir, "ICUSTAYS"),
            ingestion.read_table(mimic_dir, "DIAGNOSES_ICD"))


def test_invalid_mortality_flag_is_rejected(tmp_path, tables):
    tables["ADMISSIONS"].loc[0, "hospital_expire_flag"] = 2
    root = write_database(tmp_path / "db", tables)

    with pytest.raises(SchemaError, match="0/1"):
        load_cohort(root)


def test_icu_stay_ending_before_it_starts_is_rejected(tmp_path, tables):
    icu = tables["ICUSTAYS"]
    icu.loc[icu["icustay_id"] == 101, "outtime"] = "2149-12-31 00:00:00"
    root = write_database(tmp_path / "db", tables)

    with pytest.raises(IntegrityError, match="end before they start"):
        load_cohort(root)


def test_hospital_discharge_before_index_is_excluded_and_noted(tmp_path, tables):
    """The real Demo has one such stay: discharged 2 min before the index time."""
    adm = tables["ADMISSIONS"]
    adm.loc[adm["hadm_id"] == 11, "dischtime"] = (T0 + pd.Timedelta(hours=23)).strftime("%Y-%m-%d %H:%M:%S")
    root = write_database(tmp_path / "db", tables)

    result = load_cohort(root)

    assert 101 not in set(result.cohort["icustay_id"])
    assert any("discharge before the index time" in n for n in result.notes)


def test_all_stays_not_only_first_when_configured(mimic_dir):
    result = load_cohort(mimic_dir, config.TaskSpec(first_stay_only=False))

    # 102 is subject 1's second stay. It died at day 152, i.e. after its index time.
    assert {101, 102} <= set(result.cohort["icustay_id"])
