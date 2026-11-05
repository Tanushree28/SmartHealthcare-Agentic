"""Facts about the real MIMIC-III Demo v1.4 on disk.

Deselected by default; run with ``pytest -m requires_data``. They pin the
cohort the pipeline produces on the Demo, so a change in the cohort rules or
the readers shows up as a changed number rather than passing silently.
"""

from __future__ import annotations

import pytest

from mimic_iii import config, dataset, features, ingestion

from .conftest import load_cohort

pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def demo():
    root = config.source_dir("demo")
    if not root.is_dir():
        pytest.skip(f"MIMIC-III Demo not at {root}; see COMMANDS.md section 8")
    return root


@pytest.fixture(scope="module")
def demo_cohort(demo):
    return load_cohort(demo)


def test_demo_has_every_required_table(demo):
    assert set(ingestion.validate_source(demo)) == set(ingestion.REQUIRED_TABLES)


def test_every_configured_itemid_exists_in_the_demo_dictionaries(demo):
    labels = ingestion.check_itemids(demo)

    assert "Heart Rate" in labels["heart_rate"]
    assert "Creatinine" in labels["creatinine"]


def test_demo_cohort_flow(demo_cohort):
    """100 patients -> 51 stays, 15 deaths (the 2026-10-14 exploration found 52;
    the pipeline additionally excludes one stay discharged before the index time)."""
    flow = [(r["step"], r["stays"]) for r in demo_cohort.flow]

    assert flow[0] == ("All ICU stays", 136)
    assert flow[-1][1] == 51
    assert int(demo_cohort.cohort["label"].sum()) == 15
    assert any("discharge before the index time" in n for n in demo_cohort.notes)


def test_demo_feature_table_and_split(demo, demo_cohort):
    events, stats = features.extract_events(demo, demo_cohort.cohort)
    table = features.build_feature_table(demo_cohort.cohort, events)
    split = dataset.patient_split(table)

    assert table.shape[0] == 51
    assert table["heart_rate_missing"].sum() == 0
    assert stats.converted["temp_c:fahrenheit_to_celsius"] > 0
    assert not (set(split.train) & set(split.test))
