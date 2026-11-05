"""Fixtures for the INCART tests. Synthetic records only; see ``synthetic.py``."""

from __future__ import annotations

import pytest

from mit_bih import config as mit_config
from mit_bih import inference, normalize
from tests.mit_bih.conftest import no_network  # noqa: F401  (re-exported fixture)

from .synthetic import write_incart_record


@pytest.fixture
def incart_record(tmp_path):
    """One 12-lead record at 257 Hz: 20 beats cycling N, N, V, and one ``+``."""
    return write_incart_record(tmp_path / "incartdb", "I99")


@pytest.fixture
def bundle():
    """An inference bundle with the real preprocessing settings and no models.

    ``prepare_beats`` needs only the preprocessing fields, so no checkpoint is
    loaded. Normalization statistics are arbitrary but fixed.
    """
    stats = normalize.NormStats(mean=0.01, std=0.4)
    return inference.InferenceBundle(
        fs=mit_config.FS,
        bandpass_low_hz=mit_config.BANDPASS_LOW_HZ,
        bandpass_high_hz=mit_config.BANDPASS_HIGH_HZ,
        bandpass_order=mit_config.BANDPASS_ORDER,
        notch=False,
        samples_before=mit_config.SAMPLES_BEFORE,
        samples_after=mit_config.SAMPLES_AFTER,
        norm_mean=stats.mean,
        norm_std=stats.std,
        rr_robust=False,
        classes=tuple(mit_config.AAMI_CLASSES),
        ensemble="test",
        members=(),
    )
