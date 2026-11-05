"""Every new command-line entry point starts, and the setup check is honest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "scripts/validate_setup.py",
    "scripts/run_end_to_end.py",
    "scripts/make_integration_figures.py",
    "scripts/mimic_iii/build_features.py",
    "scripts/mimic_iii/make_splits.py",
    "scripts/mimic_iii/train.py",
    "scripts/mimic_iii/show_predictions.py",
    "scripts/mit_bih/export_inference_bundle.py",
    "scripts/mit_bih/evaluate_bundle.py",
    "scripts/mit_bih/record_learning_curve.py",
    "scripts/incart/download_incart.py",
    "scripts/incart/resampling_check.py",
    "scripts/incart/evaluate_on_incart.py",
    "scripts/multimodal/infer.py",
    "scripts/multimodal/train_fusion.py",
    "scripts/multimodal/demo_fusion_synthetic.py",
    "scripts/agentic/report.py",
    "scripts/agentic/compare_llms.py",
    "frontend/server.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_exits_cleanly(script):
    proc = subprocess.run([sys.executable, script, "--help"], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)

    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()


def test_validate_setup_on_an_empty_data_root_reports_missing_and_blocked(tmp_path):
    env = {"SMART_HEALTH_DATA": str(tmp_path), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run([sys.executable, "scripts/validate_setup.py"], cwd=ROOT,
                          capture_output=True, text=True, timeout=120, env=env)

    out = proc.stdout
    assert proc.returncode == 0, out + proc.stderr  # nothing is broken, only absent
    assert "MISSING  MIT-BIH raw records" in out
    assert "MISSING  MIMIC-III demo tables" in out
    assert "BLOCKED  MIMIC-III full tables" in out
    assert "BLOCKED  paired ECG + clinical dataset" in out
    assert "FAIL" not in out.split("{")[0]
