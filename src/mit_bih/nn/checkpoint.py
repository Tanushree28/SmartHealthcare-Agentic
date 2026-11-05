"""Saving and reloading trained weights.

A bare ``state_dict`` is not enough to rebuild this model. ``use_rr`` changes the
architecture -- the waveform-only ablation has no RR branch and a smaller
classifier head -- so weights alone do not say what to load them into. Every
checkpoint therefore carries the settings needed to reconstruct its model, plus
enough provenance to say where it came from.

Checkpoints are written on CPU so a model trained on a Colab GPU opens on a
laptop with no GPU at all.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from .. import config
from .model import HybridBeatNet


def save_checkpoint(
    path,
    state_dict: dict,
    arm: str,
    use_rr: bool,
    extra: dict | None = None,
) -> Path:
    """Write weights plus the metadata needed to reload them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "arm": arm,
        "use_rr": bool(use_rr),
        "classes": list(config.AAMI_CLASSES),
        "beat_length": config.BEAT_LENGTH,
        "phase": "1d",
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }

    torch.save(
        {
            "meta": meta,
            "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        },
        path,
    )
    return path


def load_checkpoint(path) -> tuple[HybridBeatNet, dict]:
    """Rebuild the model this checkpoint came from, and return it with its metadata.

    Raises rather than loading partially: a waveform-only checkpoint opened as a
    hybrid would otherwise leave the RR branch at random initialisation and
    quietly produce wrong numbers.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)

    if "meta" not in payload or "state_dict" not in payload:
        raise ValueError(
            f"{path} is not a Phase 1d checkpoint; expected 'meta' and "
            f"'state_dict' keys, found {sorted(payload)}"
        )

    meta = payload["meta"]
    model = HybridBeatNet(use_rr=meta["use_rr"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, meta
