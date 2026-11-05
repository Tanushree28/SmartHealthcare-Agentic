"""Typed failures for the multimodal layer."""

from __future__ import annotations


class MultimodalError(Exception):
    """Base class."""


class InputValidationError(MultimodalError):
    """A request's ECG or clinical input is malformed or unusable."""


class PairingError(MultimodalError):
    """A claimed ECG + clinical pairing is not a valid same-patient pairing."""


class FusionBlockedError(MultimodalError):
    """Fusion training or use was requested without valid paired data."""
