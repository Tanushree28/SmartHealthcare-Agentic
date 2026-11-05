"""scripts/multimodal: the inference smoke test and the fusion gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from mimic_iii import config as mimic_config

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "multimodal"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"multimodal_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


infer = _load("infer")
train_fusion = _load("train_fusion")
demo_fusion = _load("demo_fusion_synthetic")


def test_fusion_training_reports_blocked_without_pairs(capsys):
    assert train_fusion.main([]) == train_fusion.EXIT_BLOCKED
    assert "BLOCKED" in capsys.readouterr().out


def _pairs_csv(tmp_path, ecg_source="mimic3wdb-matched"):
    rows = [{"subject_id": i, "ecg_subject_id": i, "clinical_subject_id": i,
             "ecg_source": ecg_source, "clinical_source": "mimiciii",
             "ecg_time": "2150-01-01 10:00:00", "index_time": "2150-01-02 08:00:00",
             "label": i % 2, "split": "train" if i < 4 else "test"} for i in range(6)]
    path = tmp_path / "pairs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_valid_pairs_are_accepted_but_training_stays_blocked(tmp_path, capsys):
    code = train_fusion.main(["--pairs", str(_pairs_csv(tmp_path))])

    out = capsys.readouterr().out
    assert code == train_fusion.EXIT_BLOCKED
    assert "Pairs valid: 6 rows" in out


def test_mitbih_paired_with_mimic_is_rejected(tmp_path, capsys):
    code = train_fusion.main(["--pairs", str(_pairs_csv(tmp_path, ecg_source="mitbih"))])

    assert code == 1
    assert "fabricate a patient" in capsys.readouterr().err


def test_infer_clinical_json_only(tmp_path, clinical_setup, capsys):
    row = clinical_setup["parts"]["test"].iloc[0]
    feats = {k: (None if pd.isna(v) else v) for k, v in row.items()
             if k not in (*mimic_config.ID_COLUMNS, mimic_config.LABEL_COLUMN)}
    path = tmp_path / "stay.json"
    path.write_text(json.dumps(feats, default=float))

    code = infer.main(["--clinical-json", str(path), "--mimic-dir", str(clinical_setup["dir"]),
                       "--out", str(tmp_path / "out.json")])

    result = json.loads((tmp_path / "out.json").read_text())
    assert code == 0
    assert result["clinical"]["status"] == "ok"
    assert result["ecg"]["status"] == "not_provided"
    assert result["fusion"]["status"] == "BLOCKED"


def test_infer_needs_at_least_one_input():
    with pytest.raises(SystemExit):
        infer.main([])


@pytest.mark.requires_data
def test_infer_on_a_real_mitbih_record(tmp_path):
    from mit_bih import config as ecg_config

    if not ecg_config.mitdb_dir().is_dir():
        pytest.skip("MIT-BIH not on disk")
    code = infer.main(["--ecg-record", "100", "--seconds", "20", "--out", str(tmp_path / "o.json")])

    result = json.loads((tmp_path / "o.json").read_text())
    assert code == 0
    assert result["ecg"]["model"]["ensemble"] == "D_all_four"
    assert result["ecg"]["summary"]["n_beats_classified"] > 15


def test_synthetic_fusion_demo_runs_and_refuses_real_pairing(capsys):
    assert demo_fusion.main(["--epochs", "3"]) == 0

    out = capsys.readouterr().out
    assert "SYNTHETIC DEMO" in out
    assert "REFUSED" in out and "fabricate a patient" in out

