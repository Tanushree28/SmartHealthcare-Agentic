"""The shared inference interface: either modality, both, labelled, never merged."""

from __future__ import annotations

import json

import numpy as np
import pytest

from multimodal import fusion
from multimodal.clinical_encoder import ClinicalInput
from multimodal.ecg_encoder import EcgInput
from multimodal.errors import InputValidationError
from multimodal.inference import InferenceRequest, run_inference, to_json

from ..mit_bih.synthetic import make_ecg
from .conftest import make_paired

TOP_LEVEL = {"disclaimer", "modalities_received", "patient_linkage", "ecg", "clinical", "fusion"}


@pytest.fixture
def ecg_in():
    sig, peaks = make_ecg(n_beats=20)
    return EcgInput(sig, 360, peaks, record_id="synthetic-1")


@pytest.fixture
def clin_in(clinical_setup):
    return ClinicalInput.from_feature_table_row(clinical_setup["parts"]["test"].iloc[0])


def test_ecg_only(ecg_encoder, ecg_in):
    out = run_inference(InferenceRequest(ecg=ecg_in), ecg_encoder=ecg_encoder)

    assert set(out) == TOP_LEVEL
    assert out["ecg"]["status"] == "ok"
    assert out["clinical"]["status"] == "not_provided"
    assert out["fusion"]["status"] == "BLOCKED"
    assert out["patient_linkage"] == "single modality"
    assert "ranking" in out["ecg"]["score_semantics"] or "Uncalibrated" in out["ecg"]["score_semantics"]
    assert out["ecg"]["summary"]["n_beats_classified"] == 18


def test_clinical_only_is_labelled_with_its_training_data(clinical_encoder, clin_in):
    out = run_inference(InferenceRequest(clinical=clin_in), clinical_encoder=clinical_encoder)

    c = out["clinical"]
    assert c["status"] == "ok"
    assert 0.0 <= c["risk_score"] <= 1.0
    assert c["is_demo_model"] is True
    assert "Demo" in c["training_data"]
    assert c["index_time"] == "ICUSTAYS.intime + 24 h"
    assert out["ecg"]["status"] == "not_provided"


def test_both_without_same_patient_stay_separate(ecg_encoder, clinical_encoder, ecg_in, clin_in):
    out = run_inference(InferenceRequest(ecg=ecg_in, clinical=clin_in),
                        ecg_encoder=ecg_encoder, clinical_encoder=clinical_encoder)

    assert out["ecg"]["status"] == out["clinical"]["status"] == "ok"
    assert out["patient_linkage"].startswith("not asserted")
    assert out["fusion"]["status"] == "BLOCKED"
    assert "score" not in out["fusion"]
    assert set(out) == TOP_LEVEL  # no combined score anywhere at the top level


def test_no_input_is_an_error():
    with pytest.raises(InputValidationError):
        run_inference(InferenceRequest())


def test_bad_ecg_does_not_stop_the_clinical_output(ecg_encoder, clinical_encoder, clin_in):
    sig, peaks = make_ecg()
    out = run_inference(InferenceRequest(ecg=EcgInput(sig, 250, peaks), clinical=clin_in),
                        ecg_encoder=ecg_encoder, clinical_encoder=clinical_encoder)

    assert out["ecg"]["status"] == "error"
    assert "250 Hz" in out["ecg"]["error"]
    assert out["clinical"]["status"] == "ok"


def test_leaky_clinical_input_is_reported_as_an_error(clinical_encoder):
    out = run_inference(InferenceRequest(clinical=ClinicalInput({"age": 60, "hospital_expire_flag": 1})),
                        clinical_encoder=clinical_encoder)

    assert out["clinical"]["status"] == "error"
    assert "hospital_expire_flag" in out["clinical"]["error"]


def test_missing_model_is_reported_not_faked(ecg_in):
    out = run_inference(InferenceRequest(ecg=ecg_in))

    assert out["ecg"]["status"] == "model_unavailable"


def test_beats_are_listed_on_request(ecg_encoder, ecg_in):
    out = run_inference(InferenceRequest(ecg=ecg_in, include_beats=True), ecg_encoder=ecg_encoder)

    beats = out["ecg"]["beats"]
    assert len(beats) == 18
    assert set(beats[0]["scores"]) == {"N", "S", "V", "F", "Q"}


def test_output_is_json_serialisable(ecg_encoder, clinical_encoder, ecg_in, clin_in):
    out = run_inference(InferenceRequest(ecg=ecg_in, clinical=clin_in, include_beats=True),
                        ecg_encoder=ecg_encoder, clinical_encoder=clinical_encoder)

    assert json.loads(to_json(out))["ecg"]["status"] == "ok"


@pytest.fixture
def synthetic_fusion(ecg_encoder, clinical_encoder):
    train = make_paired(seed=0, n=200, ecg_dim=ecg_encoder.output_dim,
                        clinical_dim=clinical_encoder.output_dim)
    val = make_paired(seed=1, n=100, ecg_dim=ecg_encoder.output_dim,
                      clinical_dim=clinical_encoder.output_dim)
    model, _ = fusion.train_fusion(train, val, fusion.FusionTrainConfig(epochs=3))
    return fusion.FusionModel(model, {"trained_on": fusion.TRAINED_ON_SYNTHETIC})


def test_synthetic_fusion_model_is_blocked_by_default(ecg_encoder, ecg_in, synthetic_fusion):
    out = run_inference(InferenceRequest(ecg=ecg_in), ecg_encoder=ecg_encoder, fusion=synthetic_fusion)

    assert out["fusion"]["status"] == "BLOCKED"
    assert "synthetic" in out["fusion"]["reason"]


def test_synthetic_fusion_runs_only_when_allowed_and_says_so(ecg_encoder, clinical_encoder, ecg_in,
                                                            clin_in, synthetic_fusion):
    out = run_inference(InferenceRequest(ecg=ecg_in, clinical=clin_in, same_patient=True),
                        ecg_encoder=ecg_encoder, clinical_encoder=clinical_encoder,
                        fusion=synthetic_fusion, allow_synthetic_fusion=True)

    f = out["fusion"]
    assert f["status"] == "SYNTHETIC_ONLY"
    assert "not a risk estimate" in f["warning"]
    assert f["modalities_used"] == ["ecg", "clinical"]
    assert 0.0 <= f["score"] <= 1.0


def test_fusion_refuses_two_modalities_not_asserted_as_one_patient(
        ecg_encoder, clinical_encoder, ecg_in, clin_in, synthetic_fusion):
    out = run_inference(InferenceRequest(ecg=ecg_in, clinical=clin_in),
                        ecg_encoder=ecg_encoder, clinical_encoder=clinical_encoder,
                        fusion=synthetic_fusion, allow_synthetic_fusion=True)

    assert out["fusion"]["status"] == "not_applicable"


def test_fusion_handles_a_single_modality(clinical_encoder, clin_in, synthetic_fusion):
    out = run_inference(InferenceRequest(clinical=clin_in), clinical_encoder=clinical_encoder,
                        fusion=synthetic_fusion, allow_synthetic_fusion=True)

    assert out["fusion"]["modalities_used"] == ["clinical"]
    assert np.isfinite(out["fusion"]["score"])
