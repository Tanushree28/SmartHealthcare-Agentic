"""Reading tables: both on-disk forms, malformed input, chunking."""

from __future__ import annotations

import pandas as pd
import pytest

from mimic_iii import config, ingestion
from mimic_iii.errors import DataSourceMissingError, SchemaError

from .synthetic import write_database


def test_upper_gz_and_lower_csv_read_identically(mimic_dir, demo_style_dir):
    for name in ("PATIENTS", "ADMISSIONS", "ICUSTAYS", "DIAGNOSES_ICD"):
        a = ingestion.read_table(mimic_dir, name)
        b = ingestion.read_table(demo_style_dir, name)
        pd.testing.assert_frame_equal(a, b)
        assert all(c == c.lower() for c in a.columns)


def test_dates_are_parsed(mimic_dir):
    icu = ingestion.read_table(mimic_dir, "ICUSTAYS")

    assert pd.api.types.is_datetime64_any_dtype(icu["intime"])
    assert pd.api.types.is_datetime64_any_dtype(icu["outtime"])


def test_icd9_codes_keep_leading_zeros(tmp_path, tables):
    tables["DIAGNOSES_ICD"].loc[0, "icd9_code"] = "0389"
    root = write_database(tmp_path / "db", tables)

    dx = ingestion.read_table(root, "DIAGNOSES_ICD")

    assert "0389" in set(dx["icd9_code"])


def test_missing_table_is_named(tmp_path, tables):
    del tables["ICUSTAYS"]
    root = write_database(tmp_path / "db", tables)

    with pytest.raises(SchemaError, match="ICUSTAYS"):
        ingestion.read_table(root, "ICUSTAYS")


def test_missing_column_is_named(tmp_path, tables):
    tables["ADMISSIONS"] = tables["ADMISSIONS"].drop(columns="hospital_expire_flag")
    root = write_database(tmp_path / "db", tables)

    with pytest.raises(SchemaError, match="hospital_expire_flag"):
        ingestion.read_table(root, "ADMISSIONS")


def test_duplicate_primary_key_is_rejected(tmp_path, tables):
    icu = tables["ICUSTAYS"]
    tables["ICUSTAYS"] = pd.concat([icu, icu.iloc[[0]]], ignore_index=True)
    root = write_database(tmp_path / "db", tables)

    with pytest.raises(SchemaError, match="not unique"):
        ingestion.read_table(root, "ICUSTAYS")


def test_unparseable_timestamp_is_rejected(tmp_path, tables):
    tables["ICUSTAYS"].loc[0, "intime"] = "yesterday-ish"
    root = write_database(tmp_path / "db", tables)

    with pytest.raises(SchemaError, match="intime"):
        ingestion.read_table(root, "ICUSTAYS")


def test_non_numeric_event_value_is_rejected(tmp_path, tables):
    tables["CHARTEVENTS"]["valuenum"] = tables["CHARTEVENTS"]["valuenum"].astype(object)
    tables["CHARTEVENTS"].loc[3, "valuenum"] = "high"
    root = write_database(tmp_path / "db", tables)

    with pytest.raises(SchemaError, match="valuenum"):
        ingestion.read_events(root, "CHARTEVENTS", config.itemids("chart"), "icustay_id", [101])


def test_ambiguous_table_is_rejected(tmp_path, tables):
    root = write_database(tmp_path / "db", tables, gz=False)
    nested = root / "copy"
    nested.mkdir()
    (root / "PATIENTS.csv").rename(nested / "PATIENTS.csv")
    other = root / "other"
    other.mkdir()
    (other / "PATIENTS.csv").write_text((nested / "PATIENTS.csv").read_text())

    with pytest.raises(SchemaError, match="ambiguous"):
        ingestion.find_table(root, "PATIENTS")


def test_chunk_size_does_not_change_the_result(mimic_dir):
    """Memory-safe reading must be exactly equivalent to reading whole."""
    args = (mimic_dir, "CHARTEVENTS", config.itemids("chart"), "icustay_id", [101, 201, 10000])

    small, s_small = ingestion.read_events(*args, chunksize=3)
    whole, s_whole = ingestion.read_events(*args, chunksize=1_000_000)

    pd.testing.assert_frame_equal(small, whole)
    assert s_small.chunks > s_whole.chunks == 1
    assert s_small.rows_kept == s_whole.rows_kept


def test_event_row_accounting_reconciles(mimic_dir):
    df, stats = ingestion.read_events(
        mimic_dir, "CHARTEVENTS", config.itemids("chart"), "icustay_id", [101], chunksize=5)

    assert stats.reconciles()
    assert stats.rows_error_flag == 1  # the error-flagged heart rate
    assert stats.rows_missing_value == 1  # the SpO2 row with no value
    assert set(df["icustay_id"]) == {101}
    assert "error" not in df.columns


def test_unwanted_itemids_are_never_kept(mimic_dir):
    df, _ = ingestion.read_events(mimic_dir, "CHARTEVENTS", [220045], "icustay_id", [101])

    assert set(df["itemid"]) == {220045}


def test_exact_duplicate_rows_are_dropped_and_counted(tmp_path, tables):
    ce = tables["CHARTEVENTS"]
    dup = ce.iloc[[1]].copy()
    dup["row_id"] = 9999  # row_id is not read, so the rows are otherwise identical
    tables["CHARTEVENTS"] = pd.concat([ce, dup], ignore_index=True)
    root = write_database(tmp_path / "db", tables)

    _, stats = ingestion.read_events(root, "CHARTEVENTS", config.itemids("chart"), "icustay_id", [101])

    assert stats.rows_duplicate == 1


def test_itemid_check_passes_and_catches_a_missing_item(mimic_dir, tmp_path, tables):
    labels = ingestion.check_itemids(mimic_dir)
    assert set(labels) == set(config.VARIABLE_NAMES)

    tables["D_ITEMS"] = tables["D_ITEMS"][tables["D_ITEMS"]["itemid"] != 220045]
    root = write_database(tmp_path / "db", tables)
    with pytest.raises(SchemaError, match="heart_rate:220045"):
        ingestion.check_itemids(root)


def test_validate_source_checks_every_table(mimic_dir):
    report = ingestion.validate_source(mimic_dir)

    assert set(report) == set(ingestion.REQUIRED_TABLES)


def test_missing_full_mimic_says_it_needs_credentialing(tmp_path):
    with pytest.raises(DataSourceMissingError, match="credentialing"):
        ingestion.require_source(tmp_path / "mimiciii", "full")


def test_missing_demo_says_where_to_get_it(tmp_path):
    with pytest.raises(DataSourceMissingError, match="mimiciii-demo"):
        ingestion.require_source(tmp_path / "nothing", "demo")
