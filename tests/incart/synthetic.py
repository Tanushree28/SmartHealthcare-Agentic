"""Synthetic INCART-style WFDB records: 12 leads at 257 Hz with beat labels.

Real WFDB files are written into ``tmp_path`` so the wfdb reading path is
exercised without the real database.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.mit_bih.synthetic import make_ecg, write_record

LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

#: AAMI class index of each beat symbol used below (N/S/V/F/Q order).
CLASS_INDEX = {"N": 0, "V": 2, "/": 4}


def write_incart_record(
    directory: Path,
    name: str = "I99",
    fs: int = 257,
    n_beats: int = 20,
    cycle: tuple[str, ...] = ("N", "N", "V"),
    leads: tuple[str, ...] = LEADS_12,
) -> dict:
    """Write one record; lead II is the clean signal, other leads are scaled copies.

    Annotations: a ``+`` rhythm marker 50 samples before the first beat, then
    beats cycling through ``cycle``.
    """
    sig, peaks = make_ecg(n_beats=n_beats, fs=fs, rr_samples=fs)
    signals = {lead: sig * (0.3 + 0.1 * i) for i, lead in enumerate(leads)}
    if "II" in signals:
        signals["II"] = sig

    beats = [cycle[i % len(cycle)] for i in range(n_beats)]
    samples = np.concatenate([[int(peaks[0]) - 50], peaks]).astype(np.int64)
    symbols = ["+"] + beats
    write_record(directory, name, signals, fs=fs, ann_samples=samples, ann_symbols=symbols)

    return {
        "dir": directory,
        "name": name,
        "lead_ii": sig,
        "r_peaks": peaks,
        "n_beats": n_beats,
        "labels": np.array([CLASS_INDEX[b] for b in beats], dtype=np.int64),
        "n_excluded": 1,
    }
