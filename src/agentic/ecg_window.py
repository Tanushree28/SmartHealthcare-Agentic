"""Which MIT-BIH records the demo offers, and cutting a window from one.

Every offered record is tagged. **Held-out** records (DS2) were never used to
train the ECG model, so results on them are an honest view. **Trained-on**
records (DS1) were, so results on them look better than they are. Records 201
and 202 are the same person, which the notes say. The four paced records are
not offered: the model was never meant for paced rhythms.

The window's beat positions are the cardiologist's annotations, as in
training. The annotation *labels* are kept as ``truth`` for display only; the
orchestrator never gives them to an agent or to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mit_bih import config as ecg_config
from mit_bih import io as ecg_io
from mit_bih import labels as ecg_labels
from mit_bih import splits

#: Shortest window offered: one beat window is 0.83 s, a heart rate needs two beats.
MIN_SECONDS = 2.0

NOTES = {"201": "same person as record 202 (held-out)",
         "202": "same person as record 201 (trained-on)"}


class WindowError(ValueError):
    pass


@dataclass(frozen=True)
class RecordInfo:
    name: str
    tag: str  # "held-out" | "trained-on"
    note: str = ""


def offered_records() -> list[RecordInfo]:
    out = []
    for name in ecg_config.ALL_RECORDS:
        if name in ecg_config.PACED_RECORDS:
            continue
        tag = "held-out" if name in splits.DS2_RECORDS else "trained-on"
        out.append(RecordInfo(name, tag, NOTES.get(name, "")))
    return out


@dataclass(frozen=True)
class EcgWindow:
    record: RecordInfo
    lead: str
    fs: int
    start_s: float
    signal: np.ndarray  # the window only
    r_peaks: np.ndarray  # annotated beat positions, relative to the window start
    truth: tuple[str, ...]  # AAMI class per r_peak -- DISPLAY ONLY


def load_window(record: str, start_s: float, seconds: float,
                record_dir: Path | str | None = None) -> EcgWindow:
    record = str(record)
    info = {r.name: r for r in offered_records()}.get(record)
    if info is None:
        if record in ecg_config.PACED_RECORDS:
            raise WindowError(f"record {record} is paced; paced records are outside what the "
                              f"model was built for")
        raise WindowError(f"unknown record {record!r}; see --list")
    if start_s < 0:
        raise WindowError(f"start must be 0 s or later, got {start_s:g}")
    if seconds <= 0:
        raise WindowError(f"seconds must be positive, got {seconds:g}")
    if seconds < MIN_SECONDS:
        raise WindowError(f"a window must be at least {MIN_SECONDS:g} s long to hold enough "
                          f"beats; got {seconds:g} s")

    directory = Path(record_dir or ecg_config.mitdb_dir()).expanduser()
    rec = ecg_io.load_record(directory, record)
    lo = int(round(start_s * rec.fs))
    hi = lo + int(round(seconds * rec.fs))
    if hi > rec.n_samples:
        raise WindowError(f"the window ends at {hi / rec.fs:.1f} s but record {record} is "
                          f"{rec.duration_s:.1f} s long")

    ann = ecg_io.load_annotations(directory, record)
    maps = ecg_labels.map_symbols(ann.symbols)
    keep = [i for i, m in enumerate(maps) if m.is_beat and lo <= ann.samples[i] < hi]
    peaks = (ann.samples[keep] - lo).astype(np.int64)
    truth = tuple(maps[i].aami_class for i in keep)
    return EcgWindow(info, rec.lead, rec.fs, float(start_s), rec.signal[lo:hi].copy(),
                     peaks, truth)
