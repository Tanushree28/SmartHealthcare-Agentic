"""The fusion network on synthetic pairs, and the pairing rules.

These tests show the architecture works mechanically. They say nothing about
clinical performance: every row is synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from multimodal import fusion
from multimodal.errors import FusionBlockedError, PairingError

from .conftest import make_paired


def _batch(n=4, de=13, dc=10):
    return (torch.randn(n, de), torch.randn(n, dc), torch.ones(n, dtype=torch.bool),
            torch.ones(n, dtype=torch.bool))


def test_forward_shape():
    model = fusion.ConcatFusion(13, 10)

    assert model(*_batch()).shape == (4,)


def test_wrong_input_widths_are_rejected():
    model = fusion.ConcatFusion(13, 10)
    e, c, ep, cp = _batch()

    with pytest.raises(ValueError, match="ecg must be"):
        model(e[:, :5], c, ep, cp)
    with pytest.raises(ValueError, match="clinical must be"):
        model(e, c[:, :5], ep, cp)
    with pytest.raises(ValueError, match="masks"):
        model(e, c, ep[:2], cp)


def test_row_without_any_modality_is_rejected():
    model = fusion.ConcatFusion(13, 10)
    e, c, ep, cp = _batch()
    ep[1] = cp[1] = False

    with pytest.raises(ValueError, match="at least one modality"):
        model(e, c, ep, cp)


@pytest.mark.parametrize("absent", ["ecg", "clinical"])
def test_output_ignores_the_content_of_an_absent_modality(absent):
    model = fusion.ConcatFusion(13, 10).eval()
    e, c, ep, cp = _batch()
    if absent == "ecg":
        ep[:] = False
        other = model(torch.randn_like(e) * 100, c, ep, cp)
        nan = model(torch.full_like(e, float("nan")), c, ep, cp)
    else:
        cp[:] = False
        other = model(e, torch.randn_like(c) * 100, ep, cp)
        nan = model(e, torch.full_like(c, float("nan")), ep, cp)

    base = model(e, c, ep, cp)
    torch.testing.assert_close(base, other)
    torch.testing.assert_close(base, nan)


def test_fusion_uses_both_modalities_on_a_task_that_needs_both():
    train, val, test = make_paired(seed=0), make_paired(seed=1, n=300), make_paired(seed=2, n=1000)
    model, info = fusion.train_fusion(train, val, fusion.FusionTrainConfig(epochs=60, seed=0))

    both = roc_auc_score(test.y, fusion.predict(model, test))
    ecg_only = fusion.PairedArrays(test.ecg, test.clinical * 0, test.ecg_present,
                                   np.zeros_like(test.clinical_present), test.y, test.subject_id)
    clin_only = fusion.PairedArrays(test.ecg * 0, test.clinical, np.zeros_like(test.ecg_present),
                                    test.clinical_present, test.y, test.subject_id)

    auc_ecg = roc_auc_score(test.y, fusion.predict(model, ecg_only))
    auc_clin = roc_auc_score(test.y, fusion.predict(model, clin_only))
    assert both > max(auc_ecg, auc_clin) + 0.05
    assert min(auc_ecg, auc_clin) > 0.6  # modality dropout taught it to use each alone
    assert info["epochs_run"] >= 1


def test_training_is_deterministic_for_a_seed():
    train, val = make_paired(seed=0, n=200), make_paired(seed=1, n=100)
    cfg = fusion.FusionTrainConfig(epochs=5, seed=3)

    a, _ = fusion.train_fusion(train, val, cfg)
    b, _ = fusion.train_fusion(train, val, cfg)

    np.testing.assert_array_equal(fusion.predict(a, val), fusion.predict(b, val))


def test_training_with_missing_modalities_runs():
    train = make_paired(seed=0, n=300, p_missing=0.4)
    val = make_paired(seed=1, n=100, p_missing=0.4)

    model, _ = fusion.train_fusion(train, val, fusion.FusionTrainConfig(epochs=5))

    assert np.isfinite(fusion.predict(model, val)).all()


def test_standardisation_is_fitted_on_training_rows_only():
    train, val = make_paired(seed=0, n=200), make_paired(seed=1, n=100)
    val.ecg[:] += 50.0  # a shifted validation set must not move the statistics

    model, _ = fusion.train_fusion(train, val, fusion.FusionTrainConfig(epochs=1))

    np.testing.assert_allclose(model.ecg_mean.numpy(), train.ecg.mean(axis=0), atol=1e-5)


def test_subject_in_train_and_val_is_refused():
    train = make_paired(seed=0, n=50)
    val = make_paired(seed=0, n=50)  # same subject ids

    with pytest.raises(PairingError, match="both training and validation"):
        fusion.train_fusion(train, val)


def test_paired_arrays_reject_rows_with_no_modality():
    p = make_paired(n=10)
    p.ecg_present[3] = p.clinical_present[3] = False

    with pytest.raises(ValueError, match="neither modality"):
        fusion.PairedArrays(**{k: getattr(p, k) for k in
                               ("ecg", "clinical", "ecg_present", "clinical_present", "y", "subject_id")})


def test_save_load_round_trip_and_labels(tmp_path):
    train, val = make_paired(seed=0, n=200), make_paired(seed=1, n=100)
    model, _ = fusion.train_fusion(train, val, fusion.FusionTrainConfig(epochs=3))
    path = fusion.save_fusion(tmp_path / "f.pt", model, {"trained_on": fusion.TRAINED_ON_SYNTHETIC})

    loaded, meta = fusion.load_fusion(path)

    np.testing.assert_allclose(fusion.predict(loaded, val), fusion.predict(model, val), rtol=1e-6)
    assert fusion.FusionModel(loaded, meta).is_synthetic


def test_claiming_paired_training_without_validated_pairs_is_refused(tmp_path):
    model = fusion.ConcatFusion(13, 10)

    with pytest.raises(FusionBlockedError):
        fusion.save_fusion(tmp_path / "f.pt", model, {"trained_on": fusion.TRAINED_ON_PAIRED})
    with pytest.raises(ValueError):
        fusion.save_fusion(tmp_path / "f.pt", model, {"trained_on": "vibes"})


# ---------------------------------------------------------------------------
# Pairing rules
# ---------------------------------------------------------------------------


def _pairs(**overrides):
    rows = []
    for i in range(6):
        rows.append({
            "subject_id": i, "ecg_subject_id": i, "clinical_subject_id": i,
            "ecg_source": "mimic3wdb-matched", "clinical_source": "mimiciii",
            "ecg_time": pd.Timestamp("2150-01-01 10:00"), "index_time": pd.Timestamp("2150-01-02 08:00"),
            "label": i % 2, "split": ("train", "train", "train", "train", "val", "test")[i],
        })
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df.loc[0, k] = v
    return df


def test_valid_pairs_pass():
    assert len(fusion.validate_pairs(_pairs())) == 6


def test_mitbih_with_mimic_is_refused_as_fabrication():
    df = _pairs()
    df["ecg_source"] = "MITBIH"

    with pytest.raises(PairingError, match="fabricate a patient"):
        fusion.validate_pairs(df)


def test_different_subjects_are_refused():
    with pytest.raises(PairingError, match="different subjects"):
        fusion.validate_pairs(_pairs(clinical_subject_id=99))


def test_ecg_after_the_index_time_is_refused():
    with pytest.raises(PairingError, match="leakage"):
        fusion.validate_pairs(_pairs(ecg_time=pd.Timestamp("2150-01-02 09:00")))


def test_ecg_before_the_window_is_refused():
    with pytest.raises(PairingError, match="predate"):
        fusion.validate_pairs(_pairs(ecg_time=pd.Timestamp("2149-12-25 00:00")))


def test_subject_in_two_splits_is_refused():
    df = pd.concat([_pairs(), _pairs().iloc[[0]].assign(split="test")], ignore_index=True)

    with pytest.raises(PairingError, match="more than one split"):
        fusion.validate_pairs(df)


def test_missing_columns_are_named():
    with pytest.raises(PairingError, match="ecg_time"):
        fusion.validate_pairs(_pairs().drop(columns="ecg_time"))
