"""scripts/agentic/report.py end to end on synthetic data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "agentic"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"agentic_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load("report")


def _dirs(tiny_bundle, mitdb_dir, chart_env):
    return ["--bundle", str(tiny_bundle), "--mitdb-dir", str(mitdb_dir),
            "--mimic-dir", str(chart_env["out"]), "--mimic-raw-dir", str(chart_env["raw"]),
            "--llm", "off"]


def test_full_case_prints_stages_and_report(tiny_bundle, mitdb_dir, chart_env, capsys):
    code = report.main(["--ecg-record", "100", "--start", "2", "--stay", "101",
                        *_dirs(tiny_bundle, mitdb_dir, chart_env)])
    out = capsys.readouterr().out
    assert code == 0
    assert "① raw ecg: record 100 (held-out)" in out
    assert "display only" in out
    assert "SIMULATED CASE: ECG from MIT-BIH record 100" in out
    assert "Limitations" in out


def test_list_shows_records_and_stays(tiny_bundle, mitdb_dir, chart_env, capsys):
    assert report.main(["--list", *_dirs(tiny_bundle, mitdb_dir, chart_env)]) == 0
    out = capsys.readouterr().out
    assert "208  trained-on" in out and "101" in out and "CONGESTIVE HEART FAILURE" in out


def test_bad_window_exits_1(tiny_bundle, mitdb_dir, chart_env, capsys):
    code = report.main(["--ecg-record", "100", "--start", "30", "--seconds", "20",
                        *_dirs(tiny_bundle, mitdb_dir, chart_env)])
    assert code == 1 and "34.4 s long" in capsys.readouterr().err


def test_ask_without_llm_says_how_to_start_it(tiny_bundle, mitdb_dir, chart_env, capsys):
    code = report.main(["--stay", "101", "--ask", "What is a V beat?",
                        *_dirs(tiny_bundle, mitdb_dir, chart_env)])
    out = capsys.readouterr().out
    assert code == 0 and "Q: What is a V beat?" in out and "ollama serve" in out


def test_treatment_question_is_refused_even_without_llm(tiny_bundle, mitdb_dir, chart_env, capsys):
    report.main(["--stay", "101", "--ask", "What dose should we give?",
                 *_dirs(tiny_bundle, mitdb_dir, chart_env)])
    assert "can't suggest treatments" in capsys.readouterr().out


def test_missing_record_files_exit_1_with_a_message(tiny_bundle, tmp_path, chart_env, capsys):
    args = _dirs(tiny_bundle, tmp_path, chart_env)
    assert report.main(["--ecg-record", "100", *args]) == 1
    assert "no WFDB header" in capsys.readouterr().err


def test_llm_on_means_the_default_model():
    from agentic.llm import DEFAULT_MODEL

    assert report.llm_name("on") == DEFAULT_MODEL
    assert report.llm_name("off") is None
    assert report.llm_name("qwen2.5") == "qwen2.5"
