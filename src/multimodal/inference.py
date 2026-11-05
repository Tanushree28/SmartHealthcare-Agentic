"""One entry point for ECG data, clinical data, or both.

``run_inference`` validates whatever it is given, runs the matching model, and
returns a dict in which every output says what task it answers, what data the
model was trained on, and how far its numbers can be trusted.

**What it never does: combine the two models' outputs into one score.** The ECG
model classifies individual heartbeats (MIT-BIH, 1970s ambulatory recordings);
the clinical model predicts in-hospital mortality (MIMIC-III ICU stays). They
answer different questions about different populations. Averaging, multiplying
or otherwise merging their outputs would produce a number that looks like a
combined risk and is not one. A combined output comes only from a fusion model
trained on genuinely paired data -- none exists yet, so ``fusion.status`` is
``BLOCKED``.

A modality that fails validation is reported as an error without stopping the
other one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from mimic_iii import config as mimic_config
from mit_bih import inference as ecg_inference

from . import fusion as fusion_mod
from .clinical_encoder import ClinicalEncoder, ClinicalInput
from .ecg_encoder import EcgEncoder, EcgInput
from .errors import InputValidationError

DISCLAIMER = (
    "Research artifact. Not clinically validated, not a medical device, not for "
    "patient care. The ECG and clinical outputs come from independently trained "
    "models on different patient populations and are not a combined risk score."
)
ECG_TASK = ("Beat-level arrhythmia class (AAMI N/S/V/F/Q) for each heartbeat, "
            "from a single-lead ECG at 360 Hz.")
FUSION_BLOCKED_REASON = (
    "No fusion model trained on genuinely paired ECG + clinical data exists. "
    "MIT-BIH and MIMIC-III contain different patients, so they cannot be paired; "
    "a paired source (MIMIC-III Waveform Matched Subset + credentialed MIMIC-III) "
    "is not available on this machine."
)


@dataclass
class InferenceRequest:
    ecg: EcgInput | None = None
    clinical: ClinicalInput | None = None
    #: The caller asserts the ECG and clinical data are the same patient at the
    #: same time. Only then can a fusion model be applied to both.
    same_patient: bool = False
    include_beats: bool = False


def _ecg_section(request: InferenceRequest, encoder: EcgEncoder | None):
    if request.ecg is None:
        return {"status": "not_provided"}, None
    if encoder is None:
        return {"status": "model_unavailable",
                "reason": "no ECG inference bundle loaded"}, None
    try:
        enc = encoder.encode(request.ecg)
    except InputValidationError as exc:
        return {"status": "error", "error": str(exc)}, None
    pred = enc.prediction
    bundle = encoder.classifier.bundle
    out = {
        "status": "ok",
        "task": ECG_TASK,
        "model": {"ensemble": bundle.ensemble, "n_models": len(bundle.members),
                  "trained_on": bundle.trained_on},
        "score_semantics": bundle.calibration,
        "record_id": request.ecg.record_id,
        "summary": pred.summary(),
        "summary_vector": dict(zip(enc.names, enc.summary.round(6).tolist())),
    }
    if request.include_beats:
        out["beats"] = [
            {"r_peak": int(p), "predicted": pred.classes[int(k)],
             "scores": {c: round(float(s), 6) for c, s in zip(pred.classes, row)}}
            for p, k, row in zip(pred.r_peaks, pred.predicted, pred.scores)
        ]
    return out, enc


def _clinical_section(request: InferenceRequest, encoder: ClinicalEncoder | None):
    if request.clinical is None:
        return {"status": "not_provided"}, None
    if encoder is None:
        return {"status": "model_unavailable",
                "reason": "no clinical model loaded (run scripts/mimic_iii/train.py)"}, None
    try:
        enc = encoder.encode(request.clinical)
    except InputValidationError as exc:
        return {"status": "error", "error": str(exc)}, None
    meta = encoder.meta
    source = meta.get("data_source", "unknown")
    out = {
        "status": "ok",
        "task": meta["task"]["label"],
        "index_time": meta["task"]["index_time"],
        "horizon": meta["task"]["horizon"],
        "training_data": mimic_config.SOURCE_DESCRIPTIONS.get(source, source),
        "is_demo_model": bool(meta.get("is_demo", source == "demo")),
        "banner": meta.get("banner"),
        "stay_id": request.clinical.stay_id,
        "risk_score": round(enc.risk_score, 6),
        "threshold": round(enc.threshold, 6),
        "above_threshold": bool(enc.risk_score >= enc.threshold),
        "imputed_fields": enc.missing_fields,
        "ignored_fields": enc.ignored_fields,
    }
    return out, enc


def _fusion_section(request, fusion: fusion_mod.FusionModel | None, ecg_enc, clin_enc,
                    allow_synthetic: bool) -> dict:
    if fusion is None:
        return {"status": "BLOCKED", "reason": FUSION_BLOCKED_REASON}
    if fusion.is_synthetic and not allow_synthetic:
        return {"status": "BLOCKED",
                "reason": "the supplied fusion model was trained on synthetic test data; "
                          "it demonstrates the mechanics only. " + FUSION_BLOCKED_REASON}
    if ecg_enc is None and clin_enc is None:
        return {"status": "not_applicable", "reason": "no valid modality to fuse"}
    if ecg_enc is not None and clin_enc is not None and not request.same_patient:
        return {"status": "not_applicable",
                "reason": "both modalities given but the caller did not assert they belong to "
                          "the same patient at the same time; fusing them would be meaningless"}

    m = fusion.model
    ecg = np.zeros((1, m.ecg_dim)) if ecg_enc is None else ecg_enc.summary[None, :]
    clin = np.zeros((1, m.clinical_dim)) if clin_enc is None else clin_enc.vector[None, :]
    if ecg.shape[1] != m.ecg_dim or clin.shape[1] != m.clinical_dim:
        return {"status": "error", "error": "fusion model input sizes do not match the encoders"}
    with torch.no_grad():
        logit = m(torch.as_tensor(ecg, dtype=torch.float32), torch.as_tensor(clin, dtype=torch.float32),
                  torch.tensor([ecg_enc is not None]), torch.tensor([clin_enc is not None]))
    out = {
        "status": "ok" if not fusion.is_synthetic else "SYNTHETIC_ONLY",
        "score": round(float(torch.sigmoid(logit)[0]), 6),
        "modalities_used": [n for n, e in (("ecg", ecg_enc), ("clinical", clin_enc)) if e is not None],
        "trained_on": fusion.meta.get("trained_on"),
    }
    if fusion.is_synthetic:
        out["warning"] = ("Trained on synthetic test data. This number is not a risk estimate "
                          "and must not be shown as one.")
    return out


def run_inference(
    request: InferenceRequest,
    ecg_encoder: EcgEncoder | None = None,
    clinical_encoder: ClinicalEncoder | None = None,
    fusion: fusion_mod.FusionModel | None = None,
    allow_synthetic_fusion: bool = False,
) -> dict:
    if request.ecg is None and request.clinical is None:
        raise InputValidationError("provide ECG data, clinical data, or both")

    ecg_out, ecg_enc = _ecg_section(request, ecg_encoder)
    clin_out, clin_enc = _clinical_section(request, clinical_encoder)
    received = [n for n, x in (("ecg", request.ecg), ("clinical", request.clinical)) if x is not None]
    if len(received) == 2:
        linkage = ("asserted by the caller: same patient, same time" if request.same_patient else
                   "not asserted: the ECG and clinical outputs below are independent and must "
                   "not be combined")
    else:
        linkage = "single modality"
    return {
        "disclaimer": DISCLAIMER,
        "modalities_received": received,
        "patient_linkage": linkage,
        "ecg": ecg_out,
        "clinical": clin_out,
        "fusion": _fusion_section(request, fusion, ecg_enc, clin_enc, allow_synthetic_fusion),
    }


def to_json(result: dict) -> str:
    return json.dumps(result, indent=2, sort_keys=False, default=str)


def load_default_encoders(clinical_model_dir=None) -> tuple[EcgEncoder | None, ClinicalEncoder | None, list[str]]:
    """Load whichever models are available; report the ones that are not."""
    notes = []
    ecg = None
    try:
        ecg = EcgEncoder.from_bundle(ecg_inference.DEFAULT_BUNDLE_PATH)
    except (FileNotFoundError, ValueError) as exc:
        notes.append(f"ECG model unavailable: {exc}")
    clinical = None
    if clinical_model_dir is not None:
        try:
            clinical = ClinicalEncoder.from_dir(clinical_model_dir)
        except (FileNotFoundError, ValueError) as exc:
            notes.append(f"clinical model unavailable: {exc}")
    return ecg, clinical, notes
