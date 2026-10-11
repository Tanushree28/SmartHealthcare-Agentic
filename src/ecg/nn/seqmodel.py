"""The rhythm-context classifier: a CNN per beat, a BiLSTM across the window.

Phase 1d's model saw one beat and had to infer from its RR numbers what rhythm
it was early *against*. It made 3,015 N-called-S errors on DS2 -- 98% of that
arm's false positives, and the reason S precision stalled at 0.29.

Here every beat in a window is encoded by the same CNN, its RR features are
attached, and a bidirectional LSTM reads the window in both directions before
the centre beat is classified. The model can therefore compare a beat against
its actual neighbours rather than against a record-level average.

The CNN is deliberately the same stack as ``model.HybridBeatNet`` so that a
difference in results is attributable to the added context, not to a different
feature extractor.
"""

from __future__ import annotations

import torch
from torch import nn

from .. import config, features
from .model import _conv_block


class RhythmContextNet(nn.Module):
    """Classify the centre beat of a window, using its neighbours as context."""

    def __init__(
        self,
        n_classes: int = len(config.AAMI_CLASSES),
        n_rr: int = len(features.RR_FEATURE_NAMES),
        hidden: int = 64,
        dropout: float = 0.3,
        beat_length: int = config.BEAT_LENGTH,
    ) -> None:
        super().__init__()
        self.beat_length = beat_length

        # Identical to HybridBeatNet's waveform branch, so any difference in
        # results is the context, not the encoder.
        self.encoder = nn.Sequential(
            _conv_block(1, 32, 7, pool=True),
            _conv_block(32, 64, 5, pool=True),
            _conv_block(64, 64, 3, pool=True),
            _conv_block(64, 128, 3, pool=False),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.rr_branch = nn.Sequential(
            nn.Linear(n_rr, 32),
            nn.ReLU(inplace=True),
        )

        self.rhythm = nn.LSTM(
            input_size=128 + 32,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # The centre beat's own representation is concatenated with the rhythm
        # summary, so context supplements the beat rather than replacing it.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 128 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, beats: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        if beats.ndim != 3:
            raise ValueError(
                f"expected beats of shape (batch, width, {self.beat_length}), "
                f"got {tuple(beats.shape)}"
            )

        batch, width, length = beats.shape

        if length != self.beat_length:
            raise ValueError(f"beat length {length} != expected {self.beat_length}")
        if width % 2 == 0:
            raise ValueError(
                f"window width must be odd so there is a single centre beat, "
                f"got {width}"
            )
        if rr.shape[:2] != (batch, width):
            raise ValueError(
                f"rr shape {tuple(rr.shape)} does not match beats "
                f"{(batch, width)}"
            )

        # Encode every beat in the window with the shared CNN.
        flat = beats.reshape(batch * width, 1, length)
        encoded = self.encoder(flat).reshape(batch, width, -1)

        per_beat = torch.cat([encoded, self.rr_branch(rr)], dim=2)

        rhythm, _ = self.rhythm(per_beat)

        centre = width // 2
        return self.head(torch.cat([rhythm[:, centre], per_beat[:, centre]], dim=1))


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
