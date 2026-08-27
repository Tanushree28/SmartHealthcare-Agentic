"""Group G -- pipeline integration (PlanDoc/PLAN.md)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ecg import config
from ecg.errors import DataRootError
from ecg.pipeline import run


def npz_content_hash(path) -> str:
    """Hash the *arrays*, not the file.

    ``np.savez`` writes a zip archive whose member headers carry a wall-clock
    timestamp, so two identical runs produce different file bytes. Hashing the
    array payloads measures the determinism that actually matters.
    """
    digest = hashlib.sha256()
    with np.load(path) as data:
        for key in sorted(data.files):
            arr = data[key]
            digest.update(key.encode())
            digest.update(str(arr.dtype).encode())
            digest.update(str(arr.shape).encode())
            digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_g1_end_to_end_run_over_tiny_dataset(tiny_dataset, tmp_path):
    """G1: verifies the modules compose, which unit tests cannot show."""
    out = tmp_path / "processed"

    result = run(tiny_dataset["dir"], out, records=tiny_dataset["names"])

    assert result.exit_code == 0
    assert len(result.ok) == 3

    for name in tiny_dataset["names"]:
        path = out / "beats" / f"{name}.npz"
        assert path.exists()
        with np.load(path) as data:
            assert data["X"].ndim == 2
            assert data["X"].shape[1] == config.BEAT_LENGTH
            assert data["X"].dtype == np.float32
            assert data["y"].dtype == np.int8
            assert data["r_peaks"].dtype == np.int64
            assert len(data["X"]) == len(data["y"]) == len(data["r_peaks"])

    assert (out / "manifest.csv").exists()
    assert (out / "splits.json").exists()
    assert (out / "normalization.json").exists()


def test_g2_manifest_arithmetic_reconciles(tiny_dataset, tmp_path):
    """G2: an accounting identity that catches any silently discarded beat."""
    out = tmp_path / "processed"
    run(tiny_dataset["dir"], out, records=tiny_dataset["names"])

    manifest = pd.read_csv(out / "manifest.csv", dtype={"record": str})

    assert len(manifest) == 3
    for _, row in manifest.iterrows():
        accounted = (
            row["n_beats_kept"]
            + row["n_dropped_boundary"]
            + row["n_excluded_nonbeat"]
            + row["n_unknown"]
        )
        assert accounted == row["n_annotations"], f"record {row['record']} leaks beats"

        # Array length on disk must equal the manifest's kept count.
        with np.load(out / "beats" / f"{row['record']}.npz") as data:
            assert len(data["X"]) == row["n_beats_kept"]

        # Per-class counts must sum to the kept count.
        class_total = sum(row[f"n_class_{c}"] for c in config.AAMI_CLASSES)
        assert class_total == row["n_beats_kept"]


def test_g2b_non_beat_annotations_appear_in_the_manifest(tiny_dataset, tmp_path):
    """Each fixture record carries one '+' rhythm marker -- it must be counted."""
    out = tmp_path / "processed"
    run(tiny_dataset["dir"], out, records=tiny_dataset["names"])

    manifest = pd.read_csv(out / "manifest.csv", dtype={"record": str})

    assert (manifest["n_excluded_nonbeat"] == 1).all()
    assert all("nonbeat:+" in json.loads(r) for r in manifest["drop_reasons"])


def test_g3_rerunning_produces_identical_output(tiny_dataset, tmp_path):
    """G3: determinism is a precondition for trusting any later comparison."""
    first = tmp_path / "run1"
    second = tmp_path / "run2"

    run(tiny_dataset["dir"], first, records=tiny_dataset["names"])
    run(tiny_dataset["dir"], second, records=tiny_dataset["names"])

    for name in tiny_dataset["names"]:
        assert npz_content_hash(first / "beats" / f"{name}.npz") == npz_content_hash(
            second / "beats" / f"{name}.npz"
        )

    for artifact in ("manifest.csv", "splits.json", "normalization.json"):
        assert file_hash(first / artifact) == file_hash(second / artifact), artifact


def test_g4_one_corrupt_record_does_not_abort_the_run(tiny_dataset, tmp_path):
    """G4: a 48-record job must not lose 47 records' work to one bad file."""
    (tiny_dataset["dir"] / "902.hea").write_text("this is not a WFDB header\n")
    out = tmp_path / "processed"

    result = run(tiny_dataset["dir"], out, records=tiny_dataset["names"])

    assert len(result.ok) == 2
    assert len(result.failed) == 1
    assert result.failed[0].record == "902"
    assert result.exit_code != 0, "partial failure must not report success"

    assert (out / "beats" / "901.npz").exists()
    assert (out / "beats" / "903.npz").exists()
    assert not (out / "beats" / "902.npz").exists()

    manifest = pd.read_csv(out / "manifest.csv", dtype={"record": str})
    failed_row = manifest[manifest["record"] == "902"].iloc[0]
    assert failed_row["status"] == "failed"
    assert isinstance(failed_row["error"], str) and failed_row["error"]


def test_g5_normalization_statistics_derive_only_from_training_records(
    tiny_dataset, tmp_path
):
    """G5: end-to-end confirmation of B3 at the artifact level."""
    out = tmp_path / "processed"
    result = run(tiny_dataset["dir"], out, records=tiny_dataset["names"])

    payload = json.loads((out / "normalization.json").read_text())

    assert payload["fitted_on"] == "train_records_only"
    assert sorted(payload["train_records"]) == sorted(result.split.train)

    non_train = set(result.split.val) | set(result.split.test)
    assert non_train, "fixture must produce a non-empty held-out set"
    assert not (set(payload["train_records"]) & non_train)

    # Recomputing from the training beats alone must reproduce the stored stats.
    train_raw = []
    for name in result.split.train:
        record = next(r for r in result.ok if r.record == name)
        train_raw.append(record.beats.X)
    from ecg.normalize import fit

    expected = fit(train_raw)
    assert payload["stats"]["mean"] == pytest.approx(expected.mean, rel=1e-9)
    assert payload["stats"]["std"] == pytest.approx(expected.std, rel=1e-9)


def test_g5b_normalized_training_beats_are_standardized(tiny_dataset, tmp_path):
    out = tmp_path / "processed"
    result = run(tiny_dataset["dir"], out, records=tiny_dataset["names"])

    stacked = np.concatenate(
        [np.load(out / "beats" / f"{n}.npz")["X"].ravel() for n in result.split.train]
    )

    assert stacked.mean() == pytest.approx(0.0, abs=1e-3)
    assert stacked.std() == pytest.approx(1.0, abs=1e-2)


def test_g6_splits_json_records_record_wise_unit(tiny_dataset, tmp_path):
    out = tmp_path / "processed"
    run(tiny_dataset["dir"], out, records=tiny_dataset["names"])

    payload = json.loads((out / "splits.json").read_text())

    assert payload["split_unit"] == "record"
    assigned = payload["train"] + payload["val"] + payload["test"]
    assert len(assigned) == len(set(assigned)), "a record landed in two splits"


def test_g7_class_weights_are_reported_not_applied(tiny_dataset, tmp_path):
    out = tmp_path / "processed"
    run(tiny_dataset["dir"], out, records=tiny_dataset["names"])

    payload = json.loads((out / "normalization.json").read_text())

    assert payload["suggested_class_weights"]
    assert "not applied" in payload["note"]
    # The true distribution survives: no class was resampled to parity.
    counts = payload["train_class_counts"]
    assert counts["N"] != counts["V"]


def test_g8_empty_record_dir_names_the_env_var(tmp_path):
    with pytest.raises(DataRootError, match=config.DATA_ROOT_ENV_VAR):
        run(tmp_path / "empty", tmp_path / "out")


def test_g9_unknown_strategy_mentions_beat_splitting_is_unavailable(
    tiny_dataset, tmp_path
):
    with pytest.raises(ValueError, match="Beat-level splitting"):
        run(
            tiny_dataset["dir"],
            tmp_path / "out",
            records=tiny_dataset["names"],
            strategy="beat_wise",
        )


def test_g10_runs_offline(tiny_dataset, tmp_path, no_network):
    """The pipeline must never reach for the network."""
    result = run(tiny_dataset["dir"], tmp_path / "out", records=tiny_dataset["names"])
    assert result.exit_code == 0


def test_g11_train_class_counts_cover_training_records_only(tiny_dataset, tmp_path):
    """Class weights derived from val/test beats are leakage, however mild.

    Regression test: an earlier version summed class counts over every
    successfully processed record -- including the held-out splits and the
    excluded paced records -- and stored the total under 'train_class_counts'.
    """
    out = tmp_path / "processed"
    result = run(tiny_dataset["dir"], out, records=tiny_dataset["names"])

    payload = json.loads((out / "normalization.json").read_text())
    counts = payload["train_class_counts"]

    expected = 0
    for name in result.split.train:
        with np.load(out / "beats" / f"{name}.npz") as data:
            expected += len(data["y"])

    assert sum(counts.values()) == expected

    held_out = set(result.split.val) | set(result.split.test)
    held_out_beats = 0
    for name in held_out:
        with np.load(out / "beats" / f"{name}.npz") as data:
            held_out_beats += len(data["y"])
    assert held_out_beats > 0, "fixture must actually hold beats back"
    assert sum(counts.values()) != result.total_beats()


def test_g12_paced_records_are_processed_but_never_join_a_split(tmp_path):
    """Paced records still get written and counted; they just never join a split.

    Their beats must also stay out of the training class counts, or the
    suggested weights would be shaped by data no split ever sees.
    """
    from conftest import make_ecg, write_record

    directory = tmp_path / "mitdb"
    names = ["101", "106", "108", "102"]  # 102 is paced
    for i, name in enumerate(names):
        sig, r_peaks = make_ecg(n_beats=12 + i, seed=i)
        symbols = ["/"] * len(r_peaks) if name == "102" else ["N"] * len(r_peaks)
        write_record(
            directory, name, {"MLII": sig, "V1": sig * 0.5},
            ann_samples=r_peaks, ann_symbols=symbols,
        )

    out = tmp_path / "processed"
    result = run(directory, out, records=names)

    paced = next(r for r in result.records if r.record == "102")
    assert paced.status == "ok"
    assert paced.split == "excluded"
    assert (out / "beats" / "102.npz").exists(), "excluded != discarded"

    split = result.split
    assert "102" in split.excluded
    assert "102" not in set(split.train) | set(split.val) | set(split.test)

    # The paced record contributes only class Q; it must not reach train counts.
    payload = json.loads((out / "normalization.json").read_text())
    assert payload["train_class_counts"]["Q"] == 0
    assert payload["all_processed_class_counts"]["Q"] > 0


def test_g13_duplicate_record_ids_are_deduplicated(tiny_dataset, tmp_path):
    """A repeated id would otherwise land in two splits and read as leakage."""
    out = tmp_path / "processed"
    names = tiny_dataset["names"]

    result = run(tiny_dataset["dir"], out, records=names + names)

    assert len(result.records) == len(names)
    split = result.split
    assert len(set(split.train) | set(split.val) | set(split.test)) == len(names)
