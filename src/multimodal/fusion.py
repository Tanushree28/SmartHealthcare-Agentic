"""Late fusion of an ECG summary and a clinical vector, plus pairing checks.

**What exists here and what does not.** The module defines a fusion network,
a training loop, and the checks a paired dataset must pass. It is unit-tested
on synthetic paired fixtures. It has **not** been trained on real data, because
no genuinely paired ECG + clinical dataset is available: MIT-BIH (48 recordings
from 47 patients, 1975-79) and MIMIC-III (ICU patients 2001-12) contain different people,
and pairing a record from one with a record from the other would fabricate a
patient. ``validate_pairs`` refuses exactly that.

A real paired source exists -- the MIMIC-III Waveform Database Matched Subset,
linked to MIMIC-III by ``subject_id`` -- but it needs the credentialed clinical
database and terabyte-scale storage. See ``PlanDoc``.

**Architecture.** Each modality is projected to a hidden vector; a missing
modality's projection is multiplied by zero, and two presence flags are
appended, so the head can tell "absent" from "present and small". The output
for a row cannot depend on the input of a modality marked absent (tested).
Training randomly hides one modality per row (modality dropout) so the model
learns to work with either alone.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import provenance

from .errors import FusionBlockedError, PairingError

TRAINED_ON_PAIRED = "paired_clinical_data"
TRAINED_ON_SYNTHETIC = "synthetic_test_data"

#: Data sources that are known to contain different patients from each other.
#: A row pairing one with the other is rejected.
UNLINKABLE_SOURCES = {frozenset({"mitbih", "mimiciii"})}


class ConcatFusion(nn.Module):
    """Project each modality, mask the absent one, concatenate, classify."""

    def __init__(self, ecg_dim: int, clinical_dim: int, hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        if ecg_dim < 1 or clinical_dim < 1:
            raise ValueError("both modality dimensions must be positive")
        self.ecg_dim, self.clinical_dim, self.hidden = ecg_dim, clinical_dim, hidden
        # Input standardisation, fitted on training rows where the modality is present.
        self.register_buffer("ecg_mean", torch.zeros(ecg_dim))
        self.register_buffer("ecg_std", torch.ones(ecg_dim))
        self.register_buffer("clinical_mean", torch.zeros(clinical_dim))
        self.register_buffer("clinical_std", torch.ones(clinical_dim))
        self.ecg_proj = nn.Sequential(nn.Linear(ecg_dim, hidden), nn.ReLU())
        self.clinical_proj = nn.Sequential(nn.Linear(clinical_dim, hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, ecg: torch.Tensor, clinical: torch.Tensor,
                ecg_present: torch.Tensor, clinical_present: torch.Tensor) -> torch.Tensor:
        n = ecg.shape[0]
        if ecg.shape != (n, self.ecg_dim):
            raise ValueError(f"ecg must be (batch, {self.ecg_dim}), got {tuple(ecg.shape)}")
        if clinical.shape != (n, self.clinical_dim):
            raise ValueError(f"clinical must be (batch, {self.clinical_dim}), got {tuple(clinical.shape)}")
        if ecg_present.shape != (n,) or clinical_present.shape != (n,):
            raise ValueError("presence masks must be (batch,)")
        e_mask = ecg_present.to(ecg.dtype)
        c_mask = clinical_present.to(clinical.dtype)
        if torch.any((e_mask + c_mask) == 0):
            raise ValueError("every row needs at least one modality present")

        # Zero absent inputs before anything touches them: NaN * 0 is NaN.
        ecg = torch.where(e_mask[:, None] > 0, ecg, torch.zeros_like(ecg))
        clinical = torch.where(c_mask[:, None] > 0, clinical, torch.zeros_like(clinical))
        h_e = self.ecg_proj((ecg - self.ecg_mean) / self.ecg_std) * e_mask[:, None]
        h_c = self.clinical_proj((clinical - self.clinical_mean) / self.clinical_std) * c_mask[:, None]
        z = torch.cat([h_e, h_c, e_mask[:, None], c_mask[:, None]], dim=1)
        return self.head(z).squeeze(1)


@dataclass
class PairedArrays:
    """Encoded, already-paired rows. Absent modalities are rows of zeros."""

    ecg: np.ndarray  # (n, ecg_dim)
    clinical: np.ndarray  # (n, clinical_dim)
    ecg_present: np.ndarray  # (n,) bool
    clinical_present: np.ndarray  # (n,) bool
    y: np.ndarray  # (n,) 0/1
    subject_id: np.ndarray  # (n,)

    def __post_init__(self) -> None:
        n = self.y.shape[0]
        for name in ("ecg", "clinical"):
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(f"{name} must be (n, d) with n = {n}, got {arr.shape}")
        for name in ("ecg_present", "clinical_present", "subject_id"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        if not np.all(np.isin(self.y, (0, 1))):
            raise ValueError("labels must be 0/1")
        if np.any(~(self.ecg_present.astype(bool) | self.clinical_present.astype(bool))):
            raise ValueError("a row has neither modality")

    def tensors(self):
        f = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt)  # noqa: E731
        return (f(self.ecg), f(self.clinical), f(self.ecg_present, torch.bool),
                f(self.clinical_present, torch.bool), f(self.y))


@dataclass(frozen=True)
class FusionTrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-4
    hidden: int = 32
    dropout: float = 0.1
    modality_dropout: float = 0.2
    patience: int = 15
    seed: int = 0


def _fit_standardisation(model: ConcatFusion, data: PairedArrays) -> None:
    for name, present in (("ecg", data.ecg_present), ("clinical", data.clinical_present)):
        rows = getattr(data, name)[present.astype(bool)]
        if rows.shape[0] == 0:
            continue
        mean = torch.as_tensor(rows.mean(axis=0), dtype=torch.float32)
        std = torch.as_tensor(rows.std(axis=0), dtype=torch.float32)
        getattr(model, f"{name}_mean").copy_(mean)
        getattr(model, f"{name}_std").copy_(torch.where(std > 1e-8, std, torch.ones_like(std)))


def train_fusion(train: PairedArrays, val: PairedArrays,
                 cfg: FusionTrainConfig = FusionTrainConfig()) -> tuple[ConcatFusion, dict]:
    """Fit on ``train``, early-stop on ``val`` loss. Subjects must not overlap."""
    overlap = set(np.asarray(train.subject_id).tolist()) & set(np.asarray(val.subject_id).tolist())
    if overlap:
        raise PairingError(f"subjects in both training and validation: {sorted(overlap)[:5]}")

    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = ConcatFusion(train.ecg.shape[1], train.clinical.shape[1], cfg.hidden, cfg.dropout)
    _fit_standardisation(model, train)  # training rows only
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    e, c, ep, cp, y = train.tensors()
    ve, vc, vep, vcp, vy = val.tensors()
    best, best_loss, stale, history = None, float("inf"), 0, []
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(y.shape[0], generator=gen)
        for start in range(0, y.shape[0], cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            bep, bcp = ep[idx].clone(), cp[idx].clone()
            if cfg.modality_dropout > 0:
                both = bep & bcp
                drop = (torch.rand(idx.shape[0], generator=gen) < cfg.modality_dropout) & both
                which = torch.rand(idx.shape[0], generator=gen) < 0.5
                bep = bep & ~(drop & which)
                bcp = bcp & ~(drop & ~which)
            opt.zero_grad()
            loss = loss_fn(model(e[idx], c[idx], bep, bcp), y[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(ve, vc, vep, vcp), vy))
        history.append({"epoch": epoch + 1, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best, best_loss, stale = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best)
    model.eval()
    return model, {"best_val_loss": best_loss, "epochs_run": len(history),
                   "history": history, "config": asdict(cfg)}


def predict(model: ConcatFusion, data: PairedArrays) -> np.ndarray:
    model.eval()
    e, c, ep, cp, _ = data.tensors()
    with torch.no_grad():
        return torch.sigmoid(model(e, c, ep, cp)).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Pairing validation
# ---------------------------------------------------------------------------

PAIR_COLUMNS: tuple[str, ...] = (
    "subject_id", "ecg_subject_id", "clinical_subject_id", "ecg_source", "clinical_source",
    "ecg_time", "index_time", "label", "split",
)


def validate_pairs(pairs: pd.DataFrame, observation_hours: float = 24.0) -> pd.DataFrame:
    """Check that each row really is one patient's ECG and clinical data.

    Rules, each a :class:`PairingError` when broken:

    1. Both sources are named, and are not a pair known to hold different
       patients (MIT-BIH with MIMIC-III).
    2. ``ecg_subject_id == clinical_subject_id == subject_id``.
    3. The ECG was recorded inside the observation window,
       ``[index_time - observation_hours, index_time)`` -- an ECG from after the
       prediction time is leakage, one from long before describes another day.
    4. Labels are 0/1; splits are train/val/test and assign each subject once.
    """
    missing = [c for c in PAIR_COLUMNS if c not in pairs]
    if missing:
        raise PairingError(f"pairs table is missing columns {missing}")
    if pairs.empty:
        raise PairingError("pairs table is empty")

    src = pairs[["ecg_source", "clinical_source"]].astype(str).apply(lambda s: s.str.strip().str.lower())
    for (e_src, c_src) in set(map(tuple, src.to_numpy())):
        if frozenset({e_src, c_src}) in UNLINKABLE_SOURCES:
            raise PairingError(
                f"ECG from {e_src!r} paired with clinical data from {c_src!r}: these datasets "
                f"contain different patients, so a pairing would fabricate a patient")

    same = (pairs["ecg_subject_id"] == pairs["clinical_subject_id"]) & \
           (pairs["ecg_subject_id"] == pairs["subject_id"])
    if not same.all():
        raise PairingError(f"{int((~same).sum())} rows pair different subjects' ECG and clinical data")

    ecg_time = pd.to_datetime(pairs["ecg_time"])
    index_time = pd.to_datetime(pairs["index_time"])
    late = ecg_time >= index_time
    if late.any():
        raise PairingError(f"{int(late.sum())} ECGs were recorded at or after the index time (leakage)")
    early = ecg_time < index_time - pd.Timedelta(hours=observation_hours)
    if early.any():
        raise PairingError(f"{int(early.sum())} ECGs predate the {observation_hours:g} h observation window")

    if not pairs["label"].isin((0, 1)).all():
        raise PairingError("labels must be 0/1")
    if not pairs["split"].isin(("train", "val", "test")).all():
        raise PairingError("split must be train, val or test")
    per_subject = pairs.groupby("subject_id")["split"].nunique()
    if (per_subject > 1).any():
        raise PairingError(f"subjects in more than one split: {per_subject[per_subject > 1].index[:5].tolist()}")
    return pairs


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_fusion(path: Path | str, model: ConcatFusion, meta: dict) -> Path:
    if meta.get("trained_on") not in (TRAINED_ON_PAIRED, TRAINED_ON_SYNTHETIC):
        raise ValueError(f"meta['trained_on'] must be {TRAINED_ON_PAIRED!r} or {TRAINED_ON_SYNTHETIC!r}")
    if meta["trained_on"] == TRAINED_ON_PAIRED and not meta.get("pairs_validated"):
        raise FusionBlockedError("a fusion model claiming paired training data needs validated pairs")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full = {**meta, "ecg_dim": model.ecg_dim, "clinical_dim": model.clinical_dim,
            "hidden": model.hidden, "provenance": provenance.provenance()}
    torch.save({"meta": json.loads(json.dumps(full, default=str)), "state_dict": model.state_dict()}, path)
    return path


def load_fusion(path: Path | str) -> tuple[ConcatFusion, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    meta = payload["meta"]
    model = ConcatFusion(meta["ecg_dim"], meta["clinical_dim"], meta["hidden"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, meta


@dataclass
class FusionModel:
    """A loaded fusion network with its metadata."""

    model: ConcatFusion
    meta: dict = field(default_factory=dict)

    @property
    def is_synthetic(self) -> bool:
        return self.meta.get("trained_on") != TRAINED_ON_PAIRED
