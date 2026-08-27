"""Group H -- data acquisition (PlanDoc/PLAN.md). Runs entirely offline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conftest import make_ecg, write_record
from ecg import acquire, config


@pytest.fixture
def fake_database(tmp_path):
    """Three records standing in for a complete download."""
    directory = tmp_path / "mitdb"
    names = ["100", "101", "102"]
    for name in names:
        sig, r_peaks = make_ecg(n_beats=6)
        write_record(
            directory,
            name,
            {"MLII": sig, "V5": sig * 0.5},
            ann_samples=r_peaks,
            ann_symbols=["N"] * len(r_peaks),
        )
    return directory, names


def test_h1_download_is_idempotent_and_makes_no_network_call(
    fake_database, monkeypatch, no_network
):
    """H1: a complete directory must short-circuit before any fetch."""
    directory, names = fake_database

    def explode(*args, **kwargs):
        raise AssertionError("dl_database called despite a complete directory")

    monkeypatch.setattr(acquire.wfdb, "dl_database", explode)

    assert acquire.missing_records(directory, names) == []
    checks = acquire.download(directory, records=names)
    assert all(c.ok for c in checks)


def test_h2_truncated_record_is_detected_and_named(fake_database):
    """H2: a partial download otherwise surfaces later as bizarre signal values."""
    directory, names = fake_database
    dat = directory / "101.dat"
    dat.write_bytes(dat.read_bytes()[: len(dat.read_bytes()) // 2])

    checks = {c.record: c for c in acquire.verify(directory, names)}

    assert checks["100"].ok
    assert not checks["101"].ok
    assert "truncated" in checks["101"].reason
    assert "101" in str(checks["101"])
    assert acquire.missing_records(directory, names) == ["101"]


def test_h2b_missing_annotation_file_fails_the_check(fake_database):
    directory, names = fake_database
    (directory / "102.atr").unlink()

    check = acquire.check_record(directory, "102")

    assert not check.ok
    assert "102.atr" in check.reason


def test_h2c_missing_record_entirely_fails_the_check(tmp_path):
    check = acquire.check_record(tmp_path, "100")

    assert not check.ok
    assert "100.hea" in check.reason


def test_h2d_unreadable_header_fails_the_check(fake_database):
    directory, names = fake_database
    (directory / "100.hea").write_text("not a wfdb header\n")

    check = acquire.check_record(directory, "100")

    assert not check.ok
    assert "header" in check.reason


def test_h2e_wrong_sampling_rate_fails_the_check(tmp_path):
    directory = tmp_path / "mitdb"
    sig, r_peaks = make_ecg(n_beats=6)
    write_record(
        directory, "100", {"MLII": sig}, fs=250,
        ann_samples=r_peaks, ann_symbols=["N"] * len(r_peaks),
    )

    check = acquire.check_record(directory, "100")

    assert not check.ok
    assert "250" in check.reason


def test_h3_full_suite_runs_without_network(fake_database, no_network):
    """H3: keeps the suite fast, offline and CI-safe."""
    directory, names = fake_database
    checks = acquire.verify(directory, names)
    assert all(c.ok for c in checks)


def test_download_raises_if_records_still_invalid_afterwards(
    fake_database, monkeypatch
):
    """A half-complete dataset must not be reported as success."""
    directory, names = fake_database
    (directory / "101.dat").unlink()

    monkeypatch.setattr(acquire.wfdb, "dl_database", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="failed verification"):
        acquire.download(directory, records=names)


def test_expected_dat_bytes_matches_a_real_written_record(fake_database):
    directory, _ = fake_database
    header = acquire.wfdb.rdheader(str(directory / "100"))

    predicted = acquire.expected_dat_bytes(header)

    assert predicted is not None
    assert (directory / "100.dat").stat().st_size >= predicted


def test_cli_verify_mode_reports_failures_without_network(
    fake_database, capsys, no_network
):
    import download_mitbih

    directory, _ = fake_database
    (directory / "100.hea").write_text("broken\n")

    exit_code = download_mitbih.main(["--dir", str(directory), "--verify"])

    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_verify_mode_succeeds_on_the_full_expected_record_set(tmp_path, no_network):
    """The CLI verifies all 48 expected records, not just what happens to exist."""
    import download_mitbih

    directory = tmp_path / "mitdb"
    for name in config.ALL_RECORDS:
        sig, r_peaks = make_ecg(n_beats=4)
        write_record(
            directory, name, {"MLII": sig, "V5": sig * 0.5},
            ann_samples=r_peaks, ann_symbols=["N"] * len(r_peaks),
        )

    assert download_mitbih.main(["--dir", str(directory), "--verify"]) == 0


def test_directory_size_of_absent_dir_is_zero(tmp_path):
    assert acquire.directory_size_bytes(tmp_path / "nope") == 0
