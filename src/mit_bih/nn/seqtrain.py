"""Training loop for the rhythm-context model.

Separate from ``train.py`` because the batch shape differs -- windows of beats
rather than single beats -- but every other choice is deliberately identical:
same losses, same optimiser, same early-stopping metric, same ``TrainConfig``.
Anything that differed here would confound a comparison against Phase 1d.

Windows are gathered from indices at batch time. Materialising a 5-wide window
over the 50,955 DS1 beats would cost 306 MB; this costs 2 MB and a gather.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .. import config, evaluate
from ..sequence import SequenceData
from . import losses as losses_module
from .seqmodel import RhythmContextNet
from .train import TrainConfig, TrainResult, _seed_everything, resolve_device


def build_model(cfg: TrainConfig, state_dict: dict | None = None) -> RhythmContextNet:
    """Construct the model, optionally loading weights into it."""
    model = RhythmContextNet(dropout=cfg.dropout)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _batch(data: SequenceData, rows: np.ndarray, device) -> tuple:
    beats, rr = data.gather(rows)
    return (
        torch.from_numpy(beats).float().to(device),
        torch.from_numpy(rr).float().to(device),
        torch.from_numpy(data.y[rows]).long().to(device),
    )


def predict_proba_sequence(
    model: nn.Module,
    data: SequenceData,
    batch_size: int = 512,
    device=None,
) -> np.ndarray:
    """Class probabilities for every centre beat, in input order."""
    target = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    model = model.to(target)
    model.eval()

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, data.n, batch_size):
            rows = np.arange(start, min(start + batch_size, data.n))
            beats, rr, _ = _batch(data, rows, target)
            chunks.append(torch.softmax(model(beats, rr), dim=1).cpu().numpy())

    if not chunks:
        return np.zeros((0, len(config.AAMI_CLASSES)), dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def train_sequence_model(
    train_data: SequenceData,
    val_data: SequenceData,
    loss_kind: str,
    cfg: TrainConfig = TrainConfig(),
    seed: int | None = None,
) -> TrainResult:
    """Train the rhythm-context model, early-stopping on validation macro-F1."""
    if train_data.n == 0:
        raise ValueError("empty training set")
    if val_data.n == 0:
        raise ValueError("empty validation set")

    seed = cfg.seed if seed is None else seed
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    # Training split only -- Phase 1b Problem #6 was weights computed over
    # held-out beats.
    counts = np.bincount(
        train_data.y, minlength=len(config.AAMI_CLASSES)
    ).astype(np.int64)

    device = resolve_device(cfg.device)
    model = RhythmContextNet(dropout=cfg.dropout).to(device)
    loss_fn = losses_module.build_loss(
        loss_kind, counts, cap=cfg.weight_cap, gamma=cfg.gamma
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.oversample:
        per_class = np.zeros(len(counts), dtype=np.float64)
        present = counts > 0
        per_class[present] = 1.0 / counts[present]
        sample_p = per_class[train_data.y]
        sample_p = sample_p / sample_p.sum()
    else:
        sample_p = None

    best_score = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_proba: np.ndarray | None = None
    history: list[dict] = []
    since_improved = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if sample_p is None:
            order = rng.permutation(train_data.n)
        else:
            order = rng.choice(train_data.n, size=train_data.n, p=sample_p)

        running = 0.0
        for start in range(0, train_data.n, cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            beats, rr, y = _batch(train_data, rows, device)
            optimizer.zero_grad()
            loss = loss_fn(model(beats, rr), y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(rows)

        proba = predict_proba_sequence(model, val_data, device=device)
        scored = evaluate.evaluate(val_data.y, proba.argmax(axis=1))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / train_data.n,
                "val_macro_f1": scored.macro_f1,
                "val_macro_f1_headline": scored.macro_f1_of(config.HEADLINE_CLASSES),
                "val_macro_recall": scored.macro_recall,
            }
        )

        if scored.macro_f1 > best_score:
            best_score = scored.macro_f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            best_proba = proba
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= cfg.patience:
                break

    assert best_proba is not None

    return TrainResult(
        state_dict=best_state,
        best_epoch=best_epoch,
        best_val_macro_f1=float(best_score),
        history=tuple(history),
        val_proba=best_proba,
        val_y=val_data.y.copy(),
        val_record=val_data.record.copy(),
        train_counts=counts,
    )


def fit_final_sequence(
    train_data: SequenceData,
    loss_kind: str,
    cfg: TrainConfig,
    epochs: int,
) -> dict:
    """Fit on all of DS1 for a fixed epoch budget -- no held-out set remains."""
    if train_data.n == 0:
        raise ValueError("empty training set")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    _seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    counts = np.bincount(
        train_data.y, minlength=len(config.AAMI_CLASSES)
    ).astype(np.int64)

    device = resolve_device(cfg.device)
    model = RhythmContextNet(dropout=cfg.dropout).to(device)
    loss_fn = losses_module.build_loss(
        loss_kind, counts, cap=cfg.weight_cap, gamma=cfg.gamma
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(train_data.n)
        for start in range(0, train_data.n, cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            beats, rr, y = _batch(train_data, rows, device)
            optimizer.zero_grad()
            loss_fn(model(beats, rr), y).backward()
            optimizer.step()

    return copy.deepcopy(
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    )
