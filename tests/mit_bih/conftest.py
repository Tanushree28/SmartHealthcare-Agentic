"""Shared fixtures for the MIT-BIH tests.

Hard constraint from PlanDoc/PLAN.md: the whole suite runs offline with no real
dataset present. These fixtures synthesize ECG-like signals and write *real*
WFDB records into ``tmp_path``, so the wfdb reading path is genuinely exercised
without a download. The synthesis functions live in ``synthetic.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from .synthetic import make_ecg, write_record


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_ecg():
    """A clean 20-beat signal and its true R-peak indices."""
    return make_ecg()


@pytest.fixture
def synthetic_record(tmp_path):
    """Two-channel record (MLII ch0, V5 ch1) with beat and non-beat symbols.

    Annotations: a rhythm marker ``+`` first, then alternating N / V / A beats,
    with a signal-quality marker ``~`` in the middle. This mix is what the
    label-exclusion tests need.
    """
    sig, r_peaks = make_ecg(n_beats=20)
    other = sig * 0.6

    symbols: list[str] = []
    samples: list[int] = []

    # Non-beat rhythm marker before the first beat.
    samples.append(int(r_peaks[0]) - 50)
    symbols.append("+")

    beat_cycle = ["N", "N", "V", "N", "A"]
    for i, peak in enumerate(r_peaks):
        samples.append(int(peak))
        symbols.append(beat_cycle[i % len(beat_cycle)])
        if i == 10:
            samples.append(int(peak) + 60)
            symbols.append("~")  # signal-quality marker

    order = np.argsort(np.asarray(samples, dtype=np.int64), kind="stable")
    samples_arr = np.asarray(samples, dtype=np.int64)[order]
    symbols_sorted = [symbols[i] for i in order]

    directory = tmp_path / "mitdb"
    write_record(
        directory,
        "sim001",
        {"MLII": sig, "V5": other},
        ann_samples=samples_arr,
        ann_symbols=symbols_sorted,
    )

    return {
        "dir": directory,
        "name": "sim001",
        "signal": sig,
        "r_peaks": r_peaks,
        "ann_samples": samples_arr,
        "ann_symbols": symbols_sorted,
        "n_beats": len(r_peaks),
        "n_nonbeat": 2,
    }


@pytest.fixture
def tiny_dataset(tmp_path):
    """Three synthetic records -- enough to exercise splitting and the manifest."""
    directory = tmp_path / "mitdb"
    names = ["901", "902", "903"]

    for i, name in enumerate(names):
        sig, r_peaks = make_ecg(n_beats=12 + i, seed=i)
        symbols = ["N" if j % 3 else "V" for j in range(len(r_peaks))]
        samples = list(map(int, r_peaks))

        # One rhythm marker per record so manifest arithmetic has a non-beat term.
        samples.append(int(r_peaks[0]) - 40)
        symbols.append("+")

        order = np.argsort(np.asarray(samples), kind="stable")
        write_record(
            directory,
            name,
            {"MLII": sig, "V1": sig * 0.5},
            ann_samples=np.asarray(samples)[order],
            ann_symbols=[symbols[k] for k in order],
        )

    return {"dir": directory, "names": names}


@pytest.fixture
def no_network(monkeypatch):
    """Make any socket creation raise, so a test proves it stays offline."""
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during offline test")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    return True
