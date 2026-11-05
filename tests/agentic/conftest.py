"""Fixtures for the agent tests: the knowledge file, a tiny ECG ensemble, a
synthetic MIT-BIH record and (Task 5) a synthetic MIMIC-III database with a
trained model. No real patient data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from agentic.chart_data import load_chart_sources  # noqa: E402
from agentic.knowledge import load_knowledge  # noqa: E402
from common import provenance  # noqa: E402
from mimic_iii import dataset, features, train  # noqa: E402
from mit_bih import config as ecg_config  # noqa: E402
from mit_bih import inference as ecg_inference  # noqa: E402
from mit_bih.nn import checkpoint  # noqa: E402
from mit_bih.nn.model import HybridBeatNet  # noqa: E402

from ..mimic_iii.conftest import load_cohort  # noqa: E402
from ..mimic_iii.synthetic import build_tables, write_database  # noqa: E402
from ..mit_bih.synthetic import make_ecg, write_record  # noqa: E402

#: Beats of the synthetic record "100" annotated V; every other beat is N.
V_BEATS = (10, 11, 12)


@pytest.fixture(scope="session")
def knowledge():
    return load_knowledge()


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory):
    """Two randomly initialised hybrid networks behind a real bundle file."""
    d = tmp_path_factory.mktemp("ecg")
    members = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        p = checkpoint.save_checkpoint(d / f"m{seed}.pt", HybridBeatNet().state_dict(),
                                       arm="test", use_rr=True)
        members.append({"path": str(p), "sha256": provenance.sha256_file(p)})
    bundle = ecg_inference.InferenceBundle(
        fs=ecg_config.FS, bandpass_low_hz=0.5, bandpass_high_hz=40.0, bandpass_order=3,
        notch=False, samples_before=100, samples_after=200, norm_mean=0.0, norm_std=0.4,
        rr_robust=False, classes=tuple(ecg_config.AAMI_CLASSES), ensemble="test",
        members=tuple(members))
    return ecg_inference.save_bundle(d / "bundle.json", bundle)


@pytest.fixture(scope="session")
def tiny_classifier(tiny_bundle):
    return ecg_inference.EcgClassifier.from_bundle(tiny_bundle)


@pytest.fixture(scope="session")
def mitdb_dir(tmp_path_factory):
    """Record "100" (a held-out name): 40 beats 300 samples apart (34.4 s)."""
    sig, peaks = make_ecg(n_beats=40, rr_samples=300)
    symbols = ["+"] + ["V" if i in V_BEATS else "N" for i in range(len(peaks))]
    samples = np.concatenate([[int(peaks[0]) - 50], peaks])
    directory = tmp_path_factory.mktemp("mitdb")
    write_record(directory, "100", {"MLII": sig, "V5": sig * 0.6},
                 ann_samples=samples, ann_symbols=symbols)
    return directory


@pytest.fixture(scope="session")
def chart_env(tmp_path_factory):
    """The synthetic MIMIC-III database plus what the agents need on top of it:
    admission reasons, prescriptions around the index day, hospital lab flags,
    and a low potassium for stay 201. Then cohort, features, split and model,
    written the way scripts/mimic_iii/*.py write them."""
    tables = build_tables()
    adm = tables["ADMISSIONS"]
    adm["diagnosis"] = "CONGESTIVE HEART FAILURE"
    adm.loc[adm["subject_id"] == 2, "diagnosis"] = "CHEST PAIN"

    labs = tables["LABEVENTS"]
    labs.loc[(labs["subject_id"] == 1) & (labs["itemid"] == 50912) & (labs["valuenum"] == 2.0),
             "flag"] = "abnormal"
    extra = pd.DataFrame([{"row_id": len(labs) + 1, "subject_id": 2, "hadm_id": 21,
                           "itemid": 50971, "charttime": "2150-01-01 12:00:00", "value": "3.1",
                           "valuenum": 3.1, "valueuom": "mEq/L", "flag": "abnormal"}])
    labs = pd.concat([labs, extra], ignore_index=True)
    labs["hadm_id"] = labs["hadm_id"].astype("Int64")
    tables["LABEVENTS"] = labs

    # Stays 101 and 201 start 2150-01-01 08:00, so the index day is 2150-01-02.
    rx = [(1, 11, 101, "2150-01-01 00:00:00", "Amiodarone"),          # day before: in
          (1, 11, 101, "2150-01-02 00:00:00", "Metoprolol"),          # index day: out
          (1, 11, 101, "2150-01-03 00:00:00", "Warfarin"),            # after: out
          (2, 21, 201, "2149-12-31 00:00:00", "Potassium Chloride"),  # in
          (2, 21, 201, "2149-12-31 00:00:00", "Acetaminophen")]       # in
    tables["PRESCRIPTIONS"] = pd.DataFrame(
        [{"row_id": i + 1, "subject_id": s, "hadm_id": h, "icustay_id": icu, "startdate": d,
          "enddate": "2150-01-09 00:00:00", "drug": name}
         for i, (s, h, icu, d, name) in enumerate(rx)])

    raw = write_database(tmp_path_factory.mktemp("mimic") / "db", tables)
    result = load_cohort(raw)
    events, _ = features.extract_events(raw, result.cohort)
    table = features.build_feature_table(result.cohort, events)
    split = dataset.patient_split(table)
    out = tmp_path_factory.mktemp("mimic_out")
    result.cohort.to_csv(out / "cohort.csv.gz", index=False)
    table.to_csv(out / "features.csv.gz", index=False)
    dataset.write_split(out / "splits.json", split, table)
    parts = dataset.split_frame(table, split)
    model, sel = train.select_and_fit(parts["train"], parts["val"])
    train.save_model(out, model, {
        "task": {"label": "ADMISSIONS.hospital_expire_flag", "index_time": "ICUSTAYS.intime + 24 h"},
        "data_source": "demo", "is_demo": True, "banner": "PIPELINE VERIFICATION ONLY",
        "threshold": sel["threshold"], "selection": sel})
    return {"raw": raw, "out": out, "table": table}


@pytest.fixture(scope="session")
def chart_sources(chart_env):
    return load_chart_sources(chart_env["out"], chart_env["raw"])
