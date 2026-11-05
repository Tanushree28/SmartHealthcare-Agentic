"""The two-branch beat classifier.

One branch reads shape, the other reads timing.

**Why two branches and not one input vector.** Phase 1b diagnosed the S-class
failure as feature dilution: 6 RR columns competing against 30 morphology
columns inside one model, with ``max_features='sqrt'`` offering a tree ~6 of 36
columns per split, so the RR features were rarely available near the root. Here
they cannot be crowded out -- they arrive on their own path and are concatenated
at full width with the waveform representation.

**Why a waveform-only CNN would not do.** ``config.SAMPLES_BEFORE/AFTER`` crops
every beat around its R peak, so each beat is centred and the inter-beat timing
is destroyed by the cropping. A premature beat and an on-time beat produce the
identical window. S is defined by prematurity, so a waveform-only model is blind
to the feature that identifies it. ``use_rr=False`` builds exactly that model,
so the claim is measured (arm 5) rather than asserted.
"""

from __future__ import annotations

import torch
from torch import nn

from .. import config, features


def _conv_block(in_ch: int, out_ch: int, kernel: int, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class HybridBeatNet(nn.Module):
    """1D CNN over the beat waveform, plus a dense branch over the RR features."""

    def __init__(
        self,
        n_classes: int = len(config.AAMI_CLASSES),
        n_rr: int = len(features.RR_FEATURE_NAMES),
        use_rr: bool = True,
        dropout: float = 0.3,
        beat_length: int = config.BEAT_LENGTH,
    ) -> None:
        super().__init__()
        self.use_rr = use_rr
        self.beat_length = beat_length

        self.waveform = nn.Sequential(
            _conv_block(1, 32, 7, pool=True),
            _conv_block(32, 64, 5, pool=True),
            _conv_block(64, 64, 3, pool=True),
            _conv_block(64, 128, 3, pool=False),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        head_in = 128

        if use_rr:
            self.rr_branch = nn.Sequential(
                nn.Linear(n_rr, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )
            head_in += 32
        else:
            self.rr_branch = None

        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, beat: torch.Tensor, rr: torch.Tensor | None) -> torch.Tensor:
        if beat.ndim != 3 or beat.shape[1] != 1:
            raise ValueError(
                f"expected beat of shape (batch, 1, {self.beat_length}), got "
                f"{tuple(beat.shape)}"
            )
        if beat.shape[2] != self.beat_length:
            raise ValueError(
                f"beat length {beat.shape[2]} != expected {self.beat_length}"
            )

        shape = self.waveform(beat)

        if not self.use_rr:
            return self.head(shape)

        if rr is None:
            raise ValueError("this model requires rr features; got None")
        return self.head(torch.cat([shape, self.rr_branch(rr)], dim=1))


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
