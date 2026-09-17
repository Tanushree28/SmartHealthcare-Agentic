"""Training loop for the Phase 1d arms.

Every arm shares this loop. What differs between arms is the loss object and two
flags -- ``oversample`` and ``use_rr``. Keeping the loop itself identical is what
makes the comparison a comparison: if arms also differed in their optimizer or
epoch budget, no row could be attributed to its one named change.

Selection is on validation **macro-F1**, never macro recall. Macro recall
rewards over-predicting a rare class with no offsetting penalty, and ranked the
worst of six Phase 1b models first -- see
``results/2026-08-27-baseline-ds1_ds2.md`` section 3.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .. import config, evaluate
from ..dataset import BeatData, class_counts
from . import losses as losses_module
from .model import HybridBeatNet


@dataclass(frozen=True)
class TrainConfig:
    """Everything held constant across arms, plus the two flags that are not."""

    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 256
    patience: int = 8
    seed: int = config.PHASE_1D_SEED
    dropout: float = 0.3
    weight_cap: float = losses_module.DEFAULT_WEIGHT_CAP
    gamma: float = 2.0
    oversample: bool = False  # arm 4
    use_rr: bool = True  # arm 5 sets this False
    device: str = "auto"  # "auto" prefers MPS; "cpu" forces the portable path


@dataclass(frozen=True)
class TrainResult:
    """The trained model plus everything needed to score and audit it."""

    state_dict: dict
    best_epoch: int
    best_val_macro_f1: float
    history: tuple[dict, ...]
    val_proba: np.ndarray
    val_y: np.ndarray
    val_record: np.ndarray
    train_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(len(config.AAMI_CLASSES), dtype=np.int64)
    )


def _cpu_state(model: nn.Module) -> dict:
    """Checkpoint on CPU, so a model trained on MPS loads anywhere."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _tensors(data: BeatData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beat = torch.from_numpy(np.ascontiguousarray(data.X)).unsqueeze(1).float()
    rr = torch.from_numpy(np.ascontiguousarray(data.rr)).float()
    y = torch.from_numpy(np.ascontiguousarray(data.y)).long()
    return beat, rr, y


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick the compute device.

    ``auto`` prefers Apple's Metal backend when present: measured on this
    machine at 5.7 s/epoch against 51 s on CPU over the 50,955 DS1 beats, which
    is the difference between a 90-minute study and a 15-hour one. Falls back to
    CPU wherever MPS is absent, so the code still runs anywhere.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def enable_determinism(warn_only: bool = True) -> None:
    """Ask torch for reproducible kernels.

    CUDA is **not** deterministic by default: some kernels accumulate with
    atomics whose order varies between launches. Measured on a Colab T4, two
    identical runs differed by 4.3e-4 in output probabilities. CPU and MPS were
    already bit-identical without this.

    ``CUBLAS_WORKSPACE_CONFIG`` must be set in the environment **before** the
    CUDA context is created -- setting it here is too late if torch has already
    touched the GPU, so the Colab notebook sets it in its first cell.

    ``warn_only=True`` means an op with no deterministic implementation warns
    instead of raising, so a run is never blocked outright; the determinism
    check in the notebook is what reports whether it actually worked.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception:  # pragma: no cover -- older/newer torch may not accept it
        pass


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    enable_determinism()


def _loader(data: BeatData, cfg: TrainConfig, generator: torch.Generator) -> DataLoader:
    beat, rr, y = _tensors(data)
    tensor_set = TensorDataset(beat, rr, y)

    if not cfg.oversample:
        return DataLoader(
            tensor_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

    # Arm 4: sample rare classes more often. Note this duplicates beats from the
    # same patients, which is close to the patient-memorisation failure recorded
    # as Problem #9 -- a win here is to be checked, not celebrated.
    counts = np.bincount(data.y, minlength=len(config.AAMI_CLASSES)).astype(np.float64)
    per_class = np.zeros_like(counts)
    present = counts > 0
    per_class[present] = 1.0 / counts[present]
    sample_weights = torch.from_numpy(per_class[data.y]).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(tensor_set),
        replacement=True,
        generator=generator,
    )
    return DataLoader(tensor_set, batch_size=cfg.batch_size, sampler=sampler)


def predict_proba(
    model: nn.Module,
    data: BeatData,
    batch_size: int = 512,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Class probabilities for every beat, in input order.

    ``device`` moves **the model as well as the data**. A checkpoint reloaded by
    ``checkpoint.load_checkpoint`` arrives on CPU, so scoring it on an
    accelerator without moving it raises "Input type and weight type should be
    the same" -- which is exactly what happened in the Phase 1d smoke run.
    """
    beat, rr, _ = _tensors(data)
    target = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    model = model.to(target)
    model.eval()

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, beat.shape[0], batch_size):
            stop = start + batch_size
            logits = model(
                beat[start:stop].to(target),
                rr[start:stop].to(target) if model.use_rr else None,
            )
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())

    if not chunks:
        return np.zeros((0, len(config.AAMI_CLASSES)), dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def train_model(
    train_data: BeatData,
    val_data: BeatData,
    loss_kind: str,
    cfg: TrainConfig = TrainConfig(),
    seed: int | None = None,
) -> TrainResult:
    """Train one arm on one fold, early-stopping on validation macro-F1."""
    if train_data.n_kept == 0:
        raise ValueError("empty training set")
    if val_data.n_kept == 0:
        raise ValueError("empty validation set")

    seed = cfg.seed if seed is None else seed
    _seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)

    # Counts come from the training split alone. Phase 1b Problem #6 was class
    # weights computed over held-out beats -- mild leakage that propagated
    # silently into the loss.
    counts = class_counts(train_data)

    device = resolve_device(cfg.device)
    model = HybridBeatNet(dropout=cfg.dropout, use_rr=cfg.use_rr).to(device)
    loss_fn = losses_module.build_loss(
        loss_kind, counts, cap=cfg.weight_cap, gamma=cfg.gamma
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = _loader(train_data, cfg, generator)

    best_score = -1.0
    best_state = _cpu_state(model)
    best_epoch = 0
    best_proba: np.ndarray | None = None
    history: list[dict] = []
    since_improved = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for beat, rr, y in loader:
            beat, rr, y = beat.to(device), rr.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(beat, rr if cfg.use_rr else None), y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * y.shape[0]

        proba = predict_proba(model, val_data, device=device)
        scored = evaluate.evaluate(val_data.y, proba.argmax(axis=1))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / train_data.n_kept,
                "val_macro_f1": scored.macro_f1,
                "val_macro_f1_headline": scored.macro_f1_of(config.HEADLINE_CLASSES),
                "val_macro_recall": scored.macro_recall,
            }
        )

        if scored.macro_f1 > best_score:
            best_score = scored.macro_f1
            best_state = _cpu_state(model)
            best_epoch = epoch
            best_proba = proba
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= cfg.patience:
                break

    assert best_proba is not None  # at least one epoch always runs

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


def fit_final(
    train_data: BeatData,
    loss_kind: str,
    cfg: TrainConfig,
    epochs: int,
) -> dict:
    """Fit on all of DS1 for a fixed number of epochs.

    There is no held-out set left at this point, so there is nothing to
    early-stop against. ``epochs`` is the median stopping epoch across the arm's
    folds. **This substitution must be stated in the results, not glossed** --
    see the spec, section 6 step 5.
    """
    if train_data.n_kept == 0:
        raise ValueError("empty training set")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    _seed_everything(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    device = resolve_device(cfg.device)
    model = HybridBeatNet(dropout=cfg.dropout, use_rr=cfg.use_rr).to(device)
    loss_fn = losses_module.build_loss(
        loss_kind, class_counts(train_data), cap=cfg.weight_cap, gamma=cfg.gamma
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = _loader(train_data, cfg, generator)

    for _ in range(epochs):
        model.train()
        for beat, rr, y in loader:
            beat, rr, y = beat.to(device), rr.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(beat, rr if cfg.use_rr else None), y).backward()
            optimizer.step()

    return _cpu_state(model)
