"""The task definition: one source of truth, no silent drift."""

from __future__ import annotations

import pytest

from common import paths
from mimic_iii import config, train


def test_toml_task_equals_code_defaults():
    """configs/mimic_iii.toml and TaskSpec() must describe the same task."""
    assert config.load_task_spec() == config.TaskSpec()


def test_toml_model_section_equals_code_defaults():
    assert train.ModelSpec.from_dict(config.load_model_spec()) == train.ModelSpec()


def test_unknown_task_key_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[task]\nobservaton_hours = 12.0\n')

    with pytest.raises(ValueError, match="observaton_hours"):
        config.load_task_spec(path)


def test_missing_task_table_is_rejected(tmp_path):
    path = tmp_path / "empty.toml"
    path.write_text("[model]\n")

    with pytest.raises(ValueError, match=r"\[task\]"):
        config.load_task_spec(path)


@pytest.mark.parametrize("kwargs", [
    {"split_fractions": (0.5, 0.3, 0.3)},
    {"split_fractions": (1.0, 0.0, 0.0)},
    {"observation_hours": 0},
    {"lab_lookback_hours": -1},
    {"cardiac_icd9_prefixes": ("410", "999")},
])
def test_invalid_task_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        config.TaskSpec(**kwargs)


def test_no_itemid_maps_to_two_variables():
    for source in ("chart", "lab"):
        config.itemid_to_variable(source)  # raises on a duplicate


def test_every_variable_has_a_valid_range_and_lab_units():
    for v in config.VARIABLES:
        lo, hi = v.valid_range
        assert lo < hi
        if v.source == "lab":
            assert v.accepted_units, v.name
            assert all(u == u.lower() for u in v.accepted_units)


def test_post_index_columns_include_the_outcome_and_discharge_fields():
    for col in ("hospital_expire_flag", "deathtime", "dischtime", "los", "outtime",
                "icd9_code", "label"):
        assert col in config.POST_INDEX_COLUMNS


def test_describe_states_index_time_and_windows():
    d = config.TaskSpec().describe()

    assert d["index_time"] == "ICUSTAYS.intime + 24 h"
    assert d["vitals_window"] == "[intime, intime + 24 h)"
    assert d["labs_window"] == "[intime - 6 h, intime + 24 h)"


def test_common_data_root_agrees_with_mit_bih(monkeypatch, tmp_path):
    from mit_bih import config as mit_config

    assert paths.DATA_ROOT_ENV_VAR == mit_config.DATA_ROOT_ENV_VAR
    assert paths.DEFAULT_DATA_ROOT == mit_config.DEFAULT_DATA_ROOT
    monkeypatch.setenv(paths.DATA_ROOT_ENV_VAR, str(tmp_path))
    assert paths.data_root() == mit_config.data_root() == tmp_path


def test_source_paths_live_under_the_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.DATA_ROOT_ENV_VAR, str(tmp_path))

    assert config.source_dir("demo") == tmp_path / "mimiciii-demo"
    assert config.output_dir("full") == tmp_path / "processed" / "mimiciii"
    with pytest.raises(ValueError):
        config.source_dir("mimic-iv")
