"""The leakage tripwire guarding the training entry point.

Phase 1 z-scores beats using statistics fitted on *its* training records.
Training on a different split than the data was preprocessed for means the test
beats were scaled using statistics that already saw them. Nothing downstream
would reveal that, so it is refused rather than trusted to discipline.
"""

from __future__ import annotations

import json

import pytest

from ecg.baseline import NormalizationMismatchError, load_split_or_refuse


def write_run(directory, strategy, split_train, norm_train, val=(), test=()):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "splits.json").write_text(json.dumps({
        "strategy": strategy,
        "split_unit": "record",
        "train": list(split_train),
        "val": list(val),
        "test": list(test),
        "excluded": [],
    }))
    (directory / "normalization.json").write_text(json.dumps({
        "fitted_on": "train_records_only",
        "train_records": list(norm_train),
    }))
    return directory


def test_consistent_run_loads_fine(tmp_path):
    directory = write_run(tmp_path / "run", "ds1_ds2", ["101", "106"], ["101", "106"],
                          test=["100"])

    split = load_split_or_refuse(directory, required_strategy="ds1_ds2")

    assert split["train"] == ["101", "106"]


def test_refuses_when_normalization_was_fitted_on_different_records(tmp_path):
    """The exact bug: beats scaled for one split, trained on another."""
    directory = write_run(tmp_path / "run", "ds1_ds2",
                          split_train=["101", "106"],
                          norm_train=["101", "106", "100"])  # saw a test record

    with pytest.raises(NormalizationMismatchError, match="contaminated"):
        load_split_or_refuse(directory)


def test_refuses_when_the_strategy_does_not_match(tmp_path):
    """Pointing the trainer at a record_wise run while asking for ds1_ds2."""
    directory = write_run(tmp_path / "run", "record_wise", ["101"], ["101"])

    with pytest.raises(NormalizationMismatchError) as excinfo:
        load_split_or_refuse(directory, required_strategy="ds1_ds2")

    message = str(excinfo.value)
    assert "record_wise" in message
    assert "--strategy ds1_ds2" in message, "the error must say how to fix it"


def test_strategy_check_is_skipped_when_not_required(tmp_path):
    directory = write_run(tmp_path / "run", "record_wise", ["101"], ["101"])

    assert load_split_or_refuse(directory)["strategy"] == "record_wise"


@pytest.mark.parametrize("missing", ["splits.json", "normalization.json"])
def test_missing_artifact_names_the_preprocessing_script(tmp_path, missing):
    directory = write_run(tmp_path / "run", "ds1_ds2", ["101"], ["101"])
    (directory / missing).unlink()

    with pytest.raises(FileNotFoundError, match="preprocess_mitbih.py"):
        load_split_or_refuse(directory)
