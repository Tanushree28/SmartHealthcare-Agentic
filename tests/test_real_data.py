"""Tests against the real MIT-BIH database.

Deselected by default (see ``addopts`` in pytest.ini) so the suite stays offline
and runnable on a fresh clone. Run them with::

    pytest -m requires_data

These assert facts published about MIT-BIH v1.0.0, so they catch a corrupted
download or a silently changed mapping -- things the synthetic fixtures cannot.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ecg import acquire, config, io

pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def mitdb():
    directory = config.mitdb_dir()
    if not directory.is_dir() or len(io.available_records(directory)) < 48:
        pytest.skip(f"MIT-BIH not present at {directory}; run scripts/download_mitbih.py")
    return directory


@pytest.fixture(scope="module")
def manifest():
    path = config.processed_dir() / "manifest.csv"
    if not path.exists():
        pytest.skip(f"no manifest at {path}; run scripts/preprocess_mitbih.py")
    return pd.read_csv(path, dtype={"record": str})


def test_all_48_records_present_and_verify(mitdb):
    checks = acquire.verify(mitdb)

    assert len(checks) == 48
    bad = [str(c) for c in checks if not c.ok]
    assert not bad, f"failed integrity check: {bad}"


def test_every_record_is_360_hz_two_channel(mitdb):
    """PLAN.md listed this as assumed; the pipeline asserts rather than assumes."""
    for name in config.ALL_RECORDS:
        header = acquire.wfdb.rdheader(str(mitdb / name))
        assert int(header.fs) == 360, f"{name} is {header.fs} Hz"
        assert header.n_sig == 2, f"{name} has {header.n_sig} channels"


def test_only_records_102_and_104_lack_mlii(manifest):
    """The documented exception that lead-by-name selection exists to handle."""
    non_mlii = set(manifest[manifest["lead"] != "MLII"]["record"])

    assert non_mlii == {"102", "104"}


def test_beat_annotation_total_matches_published_figure(manifest):
    """MIT-BIH v1.0.0: 112,647 annotations, of which 109,494 are beats."""
    assert manifest["n_annotations"].sum() == 112647

    beats = manifest["n_beats_kept"].sum() + manifest["n_dropped_boundary"].sum()
    assert beats == 109494


def test_no_annotation_symbol_was_unmapped(manifest):
    """A non-zero unknown count means the AAMI table has a gap."""
    assert manifest["n_unknown"].sum() == 0


def test_manifest_reconciles_across_all_48_records(manifest):
    accounted = manifest[
        ["n_beats_kept", "n_dropped_boundary", "n_excluded_nonbeat", "n_unknown"]
    ].sum().sum()

    assert accounted == manifest["n_annotations"].sum()


def test_all_records_processed_without_failure(manifest):
    assert (manifest["status"] == "ok").all()


def test_paced_records_excluded_from_every_split(manifest):
    payload = json.loads((config.processed_dir() / "splits.json").read_text())

    assigned = set(payload["train"]) | set(payload["val"]) | set(payload["test"])
    assert assigned & set(config.PACED_RECORDS) == set()
    assert set(payload["excluded"]) == set(config.PACED_RECORDS)
    assert len(assigned) == 44


def test_paced_beats_do_not_reach_the_training_class_counts():
    """Nearly every Q beat is a paced beat from an excluded record."""
    path = config.processed_dir() / "normalization.json"
    if not path.exists():
        pytest.skip("run scripts/preprocess_mitbih.py first")
    payload = json.loads(path.read_text())

    assert payload["all_processed_class_counts"]["Q"] > 7000
    assert payload["train_class_counts"]["Q"] < 100
