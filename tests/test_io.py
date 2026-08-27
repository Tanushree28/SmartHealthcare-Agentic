"""Group F -- record and annotation I/O (PlanDoc/PLAN.md)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_ecg, write_record
from ecg import config
from ecg.errors import DataRootError, LeadNotFoundError, SamplingRateError
from ecg.io import available_records, load_annotations, load_record, select_lead


def test_f1_lead_selected_by_name_not_index(tmp_path):
    """F1: the exact bug name-based selection exists to prevent.

    Channel 0 is V5 and channel 1 is MLII. Index-based selection would return
    V5 and nothing else in the suite would notice.
    """
    mlii, _ = make_ecg(n_beats=5, qrs_amplitude=1.0)
    v5 = np.full_like(mlii, 0.25)  # constant, trivially distinguishable

    directory = tmp_path / "mitdb"
    write_record(directory, "rec", {"V5": v5, "MLII": mlii})

    loaded = load_record(directory, "rec")

    assert loaded.lead == "MLII"
    assert loaded.available_leads == ("V5", "MLII")
    assert loaded.signal.std() > 0.01, "returned the constant V5 channel"


def test_f2_missing_preferred_lead_falls_back_through_order(tmp_path):
    """F2: a record without MLII still loads, via the configured fallback."""
    sig, _ = make_ecg(n_beats=5)
    directory = tmp_path / "mitdb"
    write_record(directory, "rec", {"V1": sig, "V5": sig * 0.5})

    loaded = load_record(directory, "rec")

    # V5 precedes V1 in LEAD_PREFERENCE, so V5 wins.
    assert loaded.lead == "V5"
    assert config.LEAD_PREFERENCE.index("V5") < config.LEAD_PREFERENCE.index("V1")


def test_f3_no_configured_lead_present_raises_named_error(tmp_path):
    sig, _ = make_ecg(n_beats=5)
    directory = tmp_path / "mitdb"
    write_record(directory, "rec", {"ABC": sig, "XYZ": sig})

    with pytest.raises(LeadNotFoundError, match="ABC"):
        load_record(directory, "rec")


def test_f4_sampling_rate_mismatch_is_rejected(tmp_path):
    """F4: wrong-fs coefficients give a plausible but wrongly filtered signal."""
    sig, _ = make_ecg(n_beats=5)
    directory = tmp_path / "mitdb"
    write_record(directory, "rec", {"MLII": sig}, fs=250)

    with pytest.raises(SamplingRateError, match="250"):
        load_record(directory, "rec", expected_fs=360)

    # Explicitly opting out is allowed and reports the true rate.
    assert load_record(directory, "rec", expected_fs=None).fs == 250


def test_f5_missing_record_names_the_data_root_env_var(tmp_path):
    """F5: an actionable message, not a bare FileNotFoundError."""
    with pytest.raises(DataRootError) as excinfo:
        load_record(tmp_path / "nowhere", "100")

    message = str(excinfo.value)
    assert config.DATA_ROOT_ENV_VAR in message
    assert "download_mitbih.py" in message


def test_annotations_round_trip(synthetic_record):
    ann = load_annotations(synthetic_record["dir"], synthetic_record["name"])

    assert len(ann) == len(synthetic_record["ann_samples"])
    assert np.array_equal(ann.samples, synthetic_record["ann_samples"])
    assert list(ann.symbols) == synthetic_record["ann_symbols"]


def test_missing_annotation_file_is_reported(tmp_path):
    sig, _ = make_ecg(n_beats=5)
    directory = tmp_path / "mitdb"
    write_record(directory, "rec", {"MLII": sig})

    with pytest.raises(DataRootError, match=r"\.atr"):
        load_annotations(directory, "rec")


def test_select_lead_is_case_insensitive():
    assert select_lead(["mlii", "V5"]) == (0, "mlii")


def test_available_records_absent_dir_is_empty(tmp_path):
    assert available_records(tmp_path / "nope") == []


def test_available_records_lists_sorted(tiny_dataset):
    assert available_records(tiny_dataset["dir"]) == ["901", "902", "903"]
