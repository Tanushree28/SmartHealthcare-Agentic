"""ECG and clinical encoders: shapes, validation, agreement with their models."""

from __future__ import annotations

import numpy as np
import pytest

from mimic_iii import dataset
from multimodal.clinical_encoder import ClinicalInput
from multimodal.ecg_encoder import SUMMARY_NAMES, EcgInput
from multimodal.errors import InputValidationError

from ..mit_bih.synthetic import make_ecg


def test_ecg_summary_has_fixed_named_length(ecg_encoder):
    short_sig, short_peaks = make_ecg(n_beats=8)
    long_sig, long_peaks = make_ecg(n_beats=60)

    a = ecg_encoder.encode(EcgInput(short_sig, 360, short_peaks))
    b = ecg_encoder.encode(EcgInput(long_sig, 360, long_peaks))

    assert a.summary.shape == b.summary.shape == (ecg_encoder.output_dim,)
    assert a.names == SUMMARY_NAMES
    fractions = [v for n, v in zip(a.names, a.summary) if n.startswith("fraction_predicted_")]
    assert sum(fractions) == pytest.approx(1.0)
    assert np.isfinite(a.summary).all()


@pytest.mark.parametrize("ecg", [
    EcgInput(np.zeros((2, 100)), 360),
    EcgInput(np.zeros(0), 360),
    EcgInput(make_ecg()[0], 0),
    EcgInput(make_ecg()[0], 360.5),
    EcgInput(make_ecg()[0], 360, np.array([200, 560])),
])
def test_malformed_ecg_is_rejected(ecg_encoder, ecg):
    with pytest.raises(InputValidationError):
        ecg_encoder.encode(ecg)


def test_ecg_errors_from_the_model_are_wrapped(ecg_encoder):
    sig, peaks = make_ecg()

    with pytest.raises(InputValidationError, match="250 Hz"):
        ecg_encoder.encode(EcgInput(sig, 250, peaks))


def test_clinical_score_equals_the_saved_pipeline(clinical_encoder, clinical_setup):
    test = clinical_setup["parts"]["test"]
    X, _ = dataset.xy(test)
    expected = clinical_encoder.model.predict_proba(X)[:, 1]

    got = [clinical_encoder.encode(ClinicalInput.from_feature_table_row(row)).risk_score
           for _, row in test.iterrows()]

    np.testing.assert_allclose(got, expected)


def test_clinical_vector_has_the_model_width(clinical_encoder, clinical_setup):
    row = clinical_setup["table"].iloc[0]

    enc = clinical_encoder.encode(ClinicalInput.from_feature_table_row(row))

    assert enc.vector.shape == (clinical_encoder.output_dim,)
    assert np.isfinite(enc.vector).all()


def test_missing_clinical_fields_are_imputed_and_listed(clinical_encoder):
    enc = clinical_encoder.encode(ClinicalInput({"age": 70, "gender": "F", "heart_rate_mean": 95.0,
                                                 "heart_rate_count": 4}))

    assert "lactate_mean" in enc.missing_fields
    assert "age" not in enc.missing_fields
    assert 0.0 <= enc.risk_score <= 1.0


def test_unknown_clinical_fields_are_ignored_and_listed(clinical_encoder):
    enc = clinical_encoder.encode(ClinicalInput({"age": 70, "favourite_colour": "blue"}))

    assert enc.ignored_fields == ["favourite_colour"]


@pytest.mark.parametrize("field", ["hospital_expire_flag", "los", "deathtime", "icd9_code",
                                   "icustay_id", "label"])
def test_post_index_or_identifier_fields_are_rejected(clinical_encoder, field):
    with pytest.raises(InputValidationError, match="not allowed"):
        clinical_encoder.encode(ClinicalInput({"age": 70, field: 1}))


def test_non_numeric_clinical_value_is_rejected(clinical_encoder):
    with pytest.raises(InputValidationError, match="numeric"):
        clinical_encoder.encode(ClinicalInput({"age": "seventy"}))


def test_empty_clinical_input_is_rejected(clinical_encoder):
    with pytest.raises(InputValidationError):
        clinical_encoder.encode(ClinicalInput({}))


def test_feature_table_row_never_passes_identifiers(clinical_setup):
    row = clinical_setup["table"].iloc[0]

    ci = ClinicalInput.from_feature_table_row(row)

    assert not {"subject_id", "hadm_id", "icustay_id", "label"} & set(ci.features)
    assert ci.stay_id == str(int(row["icustay_id"]))
