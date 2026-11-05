"""What the chart agent reads, and what it must never read."""

from __future__ import annotations

from agentic import chart_data


def test_outcome_is_never_loaded(chart_sources):
    cols = set(chart_sources.table.columns) | set(chart_sources.cohort.columns)
    assert not cols & {"label", "hospital_expire_flag", "deathtime", "dischtime",
                       "discharge_location"}


def test_admission_reasons(chart_sources):
    assert chart_sources.admission_reason[11] == "CONGESTIVE HEART FAILURE"
    assert chart_sources.admission_reason[21] == "CHEST PAIN"


def test_medications_use_only_days_before_the_index_day(chart_sources):
    meds = chart_sources.medications
    assert sorted(meds.loc[meds["hadm_id"] == 11, "drug"]) == ["Amiodarone"]
    assert sorted(meds.loc[meds["hadm_id"] == 21, "drug"]) == ["Acetaminophen",
                                                               "Potassium Chloride"]


def test_missing_prescriptions_table_gives_no_medications(tmp_path, chart_sources):
    meds = chart_data.read_medications(tmp_path, chart_sources.cohort)
    assert meds.empty and list(meds.columns) == ["hadm_id", "drug", "startdate"]


def test_last_lab_matches_the_feature_table(chart_sources, chart_env):
    table = chart_env["table"].set_index("icustay_id")
    labs = chart_sources.last_labs
    assert len(labs) > 0
    for row in labs.itertuples(index=False):
        assert row.value == table.loc[row.icustay_id, f"{row.variable}_last"]


def test_last_lab_keeps_the_hospital_flag(chart_sources):
    labs = chart_sources.last_labs
    creat = labs[(labs["icustay_id"] == 101) & (labs["variable"] == "creatinine")].iloc[0]
    assert creat["value"] == 2.0 and creat["flag"] == "abnormal"
    assert labs[(labs["icustay_id"] == 101) & (labs["variable"] == "potassium")].empty  # wrong unit


def test_split_of_covers_every_cohort_subject(chart_sources):
    assert set(chart_sources.split_of) == set(chart_sources.cohort["subject_id"].astype(int))
    assert set(chart_sources.split_of.values()) <= {"train", "val", "test"}
