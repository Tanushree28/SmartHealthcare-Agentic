"""The three MIMIC-III scripts, end to end on the synthetic database."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mimic_iii.errors import LeakageError

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "mimic_iii"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"mimic_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_features = _load("build_features")
make_splits = _load("make_splits")
train_script = _load("train")
show_predictions = _load("show_predictions")


@pytest.fixture
def pipeline_out(tmp_path, mimic_dir):
    out = tmp_path / "out"
    assert build_features.main(["--source", "demo", "--root", str(mimic_dir),
                                "--out-dir", str(out), "--chunksize", "11"]) == 0
    assert make_splits.main(["--out-dir", str(out)]) == 0
    assert train_script.main(["--out-dir", str(out)]) == 0
    return out


def test_scripts_write_every_artifact(pipeline_out):
    for name in ("cohort.csv.gz", "features.csv.gz", "build_summary.json", "splits.json",
                 "model.pkl", "model_meta.json", "report.json", "report.md"):
        assert (pipeline_out / name).exists(), name


def test_report_is_labelled_and_complete(pipeline_out):
    report = json.loads((pipeline_out / "report.json").read_text())

    assert "NOT CLINICAL PERFORMANCE" in report["banner"]
    assert set(report["metrics"]) == {"train", "val", "test"}
    assert report["task"]["index_time"] == "ICUSTAYS.intime + 24 h"
    assert report["provenance"]["inputs"]["features"]["sha256"]


def test_summary_counts_reconcile(pipeline_out):
    summary = json.loads((pipeline_out / "build_summary.json").read_text())

    for table in ("chart_ingest", "lab_ingest"):
        s = summary["events"][table]
        dropped = (s["rows_outside_cohort"] + s["rows_missing_key"] + s["rows_error_flag"]
                   + s["rows_missing_value"] + s["rows_duplicate"])
        assert s["rows_kept"] + dropped == s["rows_wanted_itemid"]
    assert summary["cohort"]["n_stays"] == summary["feature_table"]["rows"]
    assert summary["cohort"]["flow"][-1]["stays"] == summary["cohort"]["n_stays"]


def test_changed_feature_table_trips_the_split_tripwire(pipeline_out):
    path = pipeline_out / "features.csv.gz"
    import pandas as pd

    df = pd.read_csv(path)
    df.loc[0, "age"] = df.loc[0, "age"] + 1
    df.to_csv(path, index=False)

    with pytest.raises(LeakageError, match="different feature table"):
        train_script.main(["--out-dir", str(pipeline_out)])


def test_full_source_without_access_reports_blocked(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SMART_HEALTH_DATA", str(tmp_path))

    code = build_features.main(["--source", "full"])

    assert code == build_features.EXIT_BLOCKED
    assert "BLOCKED" in capsys.readouterr().err


def test_training_without_features_names_the_missing_step(tmp_path, capsys):
    assert train_script.main(["--out-dir", str(tmp_path)]) == 1
    assert "build_features.py" in capsys.readouterr().err


def test_malformed_source_fails_with_a_named_error(tmp_path, tables, capsys):
    from .synthetic import write_database

    tables["ICUSTAYS"] = tables["ICUSTAYS"].drop(columns="intime")
    root = write_database(tmp_path / "bad", tables)

    code = build_features.main(["--root", str(root), "--out-dir", str(tmp_path / "o")])

    assert code == 1
    assert "intime" in capsys.readouterr().err


def test_show_predictions_lists_every_test_patient(pipeline_out, capsys):
    import pandas as pd

    assert show_predictions.main(["--out-dir", str(pipeline_out)]) == 0

    out = capsys.readouterr().out
    split = json.loads((pipeline_out / "splits.json").read_text())
    table = pd.read_csv(pipeline_out / "features.csv.gz")
    n_test = int(table["subject_id"].isin(split["test"]).sum())
    assert out.count("\ntest ") == n_test
    assert f"of {n_test}." in out


def test_show_predictions_without_a_model_names_the_steps(tmp_path, capsys):
    assert show_predictions.main(["--out-dir", str(tmp_path)]) == 1
    assert "train.py" in capsys.readouterr().err

