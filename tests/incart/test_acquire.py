"""Download, on-disk checks and the record-to-patient grouping. All offline."""

from __future__ import annotations

import pytest

from incart import config, io

from .synthetic import write_incart_record


def test_complete_directory_makes_no_network_call(incart_record, monkeypatch, no_network):
    monkeypatch.setattr(io.wfdb, "dl_database",
                        lambda *a, **k: pytest.fail("fetched despite a complete directory"))

    checks = io.download(incart_record["dir"], records=["I99"])

    assert [c.ok for c in checks] == [True]


def test_only_missing_records_are_fetched(tmp_path, monkeypatch):
    write_incart_record(tmp_path, "I97")
    calls = []

    def fake_dl_database(db_dir, dl_dir, records="all", **kwargs):
        calls.append((db_dir, list(records)))
        for name in records:
            write_incart_record(tmp_path, name)

    monkeypatch.setattr(io.wfdb, "dl_database", fake_dl_database)

    checks = io.download(tmp_path, records=["I97", "I98"])

    assert calls == [("incartdb", ["I98"])]
    assert all(c.ok for c in checks)


def test_download_that_leaves_a_bad_record_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(io.wfdb, "dl_database", lambda *a, **k: None)  # fetches nothing

    with pytest.raises(RuntimeError, match="I98"):
        io.download(tmp_path, records=["I98"])


def test_check_fails_without_lead_ii(tmp_path):
    write_incart_record(tmp_path, "I96", leads=("I", "III", "V1"))

    check = io.check_record(tmp_path, "I96")

    assert not check.ok and "lead II" in check.reason


def test_check_fails_at_another_rate(tmp_path):
    write_incart_record(tmp_path, "I95", fs=360)

    check = io.check_record(tmp_path, "I95")

    assert not check.ok and "257" in check.reason


def test_verify_names_a_missing_record(incart_record):
    checks = io.verify(incart_record["dir"], records=["I99", "I42"])

    assert [c.ok for c in checks] == [True, False]
    assert io.missing_records(incart_record["dir"], records=["I99", "I42"]) == ["I42"]


def test_patient_file_already_present_needs_no_network(tmp_path, monkeypatch, no_network):
    (tmp_path / config.PATIENT_FILE).write_text("I01\n")
    monkeypatch.setattr(io.wfdb, "dl_files", lambda *a, **k: pytest.fail("fetched"))

    assert io.fetch_patient_file(tmp_path) is True


def test_patient_file_download_failure_returns_false(tmp_path, monkeypatch):
    def raise_oserror(*args, **kwargs):
        raise OSError("404 Not Found")

    monkeypatch.setattr(io.wfdb, "dl_files", raise_oserror)

    assert io.fetch_patient_file(tmp_path) is False


def test_one_line_per_patient_groups_records():
    pm = io.build_patient_map("Patient 1: I01, I02\n\nPatient 2: I03\n", ("I01", "I02", "I03"), 2)

    assert pm.by_patient
    assert pm.patient_of == {"I01": "P01", "I02": "P01", "I03": "P02"}
    assert pm.groups == {"P01": ["I01", "I02"], "P02": ["I03"]}


def test_duplicate_ids_fall_back_to_records():
    # A diagnosis code such as ICD-10 "I02" would look like a record id.
    pm = io.build_patient_map("I01 I02\nI02 I03\n", ("I01", "I02", "I03"), 2)

    assert not pm.by_patient
    assert pm.patient_of == {"I01": "I01", "I02": "I02", "I03": "I03"}
    assert "understates" in pm.note


def test_wrong_patient_count_falls_back():
    pm = io.build_patient_map("I01 I02\nI03\n", ("I01", "I02", "I03"), 3)

    assert not pm.by_patient and "2" in pm.note


def test_missing_record_falls_back():
    pm = io.build_patient_map("I01 I02\n", ("I01", "I02", "I03"), 1)

    assert not pm.by_patient and "I03" in pm.note


def test_no_file_falls_back():
    pm = io.build_patient_map(None, ("I01",), 1)

    assert not pm.by_patient and pm.patient_of == {"I01": "I01"}


def test_load_patient_map_reads_the_file(tmp_path):
    (tmp_path / config.PATIENT_FILE).write_text("I01 I02\nI03\n")

    pm = io.load_patient_map(tmp_path, records=("I01", "I02", "I03"), expected_patients=2)

    assert pm.by_patient and pm.patient_of["I02"] == "P01"
