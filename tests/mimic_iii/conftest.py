"""Fixtures for the MIMIC-III tests. Synthetic tables only; see ``synthetic.py``."""

from __future__ import annotations

import pytest

from mimic_iii import cohort, config, features, ingestion

from .synthetic import build_tables, write_database


@pytest.fixture
def tables():
    return build_tables()


@pytest.fixture
def mimic_dir(tmp_path, tables):
    """Full-release style: upper-case columns, gzipped."""
    return write_database(tmp_path / "mimiciii", tables, upper=True, gz=True)


@pytest.fixture
def demo_style_dir(tmp_path, tables):
    """Demo style: lower-case columns, plain CSV."""
    return write_database(tmp_path / "mimiciii-demo", tables, upper=False, gz=False)


def load_cohort(root, spec=config.TaskSpec()):
    return cohort.build_cohort(
        ingestion.read_table(root, "PATIENTS"),
        ingestion.read_table(root, "ADMISSIONS"),
        ingestion.read_table(root, "ICUSTAYS"),
        ingestion.read_table(root, "DIAGNOSES_ICD"),
        spec,
    )


@pytest.fixture
def cohort_result(mimic_dir):
    return load_cohort(mimic_dir)


@pytest.fixture
def feature_table(mimic_dir, cohort_result):
    events, _ = features.extract_events(mimic_dir, cohort_result.cohort, chunksize=7)
    return features.build_feature_table(cohort_result.cohort, events)
