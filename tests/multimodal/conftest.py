"""Fixtures: a tiny ECG ensemble, a clinical model trained on synthetic MIMIC
tables, and synthetic paired arrays for fusion. No real patient data."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common import provenance  # noqa: E402
from mimic_iii import dataset, features, train  # noqa: E402
from mit_bih import config as ecg_config  # noqa: E402
from mit_bih import inference as ecg_inference  # noqa: E402
from mit_bih.nn import checkpoint  # noqa: E402
from mit_bih.nn.model import HybridBeatNet  # noqa: E402
from multimodal.clinical_encoder import ClinicalEncoder  # noqa: E402
from multimodal.ecg_encoder import EcgEncoder  # noqa: E402
from multimodal.fusion import PairedArrays  # noqa: E402

from ..mimic_iii.conftest import load_cohort  # noqa: E402
from ..mimic_iii.synthetic import write_database  # noqa: E402


@pytest.fixture(scope="session")
def ecg_encoder(tmp_path_factory):
    """Two randomly initialised hybrid networks behind a real bundle."""
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
    path = ecg_inference.save_bundle(d / "bundle.json", bundle)
    return EcgEncoder.from_bundle(path)


@pytest.fixture(scope="session")
def clinical_setup(tmp_path_factory):
    """A clinical model trained on the synthetic MIMIC database, plus its table."""
    root = write_database(tmp_path_factory.mktemp("mimic") / "db")
    result = load_cohort(root)
    events, _ = features.extract_events(root, result.cohort)
    table = features.build_feature_table(result.cohort, events)
    parts = dataset.split_frame(table, dataset.patient_split(table))
    model, sel = train.select_and_fit(parts["train"], parts["val"])
    out = tmp_path_factory.mktemp("model")
    train.save_model(out, model, {
        "task": {"label": "ADMISSIONS.hospital_expire_flag (death before hospital discharge)",
                 "index_time": "ICUSTAYS.intime + 24 h", "horizon": "index time to hospital discharge"},
        "data_source": "demo", "is_demo": True, "banner": "PIPELINE VERIFICATION ONLY",
        "threshold": sel["threshold"], "selection": sel})
    return {"dir": out, "table": table, "parts": parts}


@pytest.fixture(scope="session")
def clinical_encoder(clinical_setup):
    return ClinicalEncoder.from_dir(clinical_setup["dir"])


def make_paired(n=600, ecg_dim=13, clinical_dim=10, seed=0, p_missing=0.0) -> PairedArrays:
    """Synthetic pairs whose label depends on BOTH modalities."""
    rng = np.random.default_rng(seed)
    ecg = rng.normal(size=(n, ecg_dim))
    clin = rng.normal(size=(n, clinical_dim))
    logit = 2.5 * ecg[:, 0] + 2.5 * clin[:, 0]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    ep = np.ones(n, dtype=bool)
    cp = np.ones(n, dtype=bool)
    if p_missing:
        drop = rng.random(n) < p_missing
        which = rng.random(n) < 0.5
        ep[drop & which] = False
        cp[drop & ~which] = False
        ecg[~ep] = 0.0
        clin[~cp] = 0.0
    return PairedArrays(ecg=ecg, clinical=clin, ecg_present=ep, clinical_present=cp,
                        y=y, subject_id=np.arange(n) + seed * 100_000)
