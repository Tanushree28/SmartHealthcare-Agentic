"""Heart agent: the ECG model's beat-by-beat calls, turned into rhythm findings.

Input is :class:`BeatCalls` -- plain arrays -- so this module needs no torch
and can be tested with hand-made calls. Beats are dropped only at the window
edges (no full window, or no RR neighbour), so the classified beats are
consecutive and a "run" is simply consecutive calls of one class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .findings import Finding, FindingList

CLASS_NAMES = {"N": "normal", "S": "supraventricular ectopic", "V": "ventricular ectopic",
               "F": "fusion", "Q": "unclassifiable"}
MIN_RUN = 3
LOW_CONFIDENCE = 0.5
Q_SHARE_FLAG = 5.0  # percent
MODEL_CAVEAT = "classified by the ECG model (scores uncalibrated), not by a cardiologist"


@dataclass(frozen=True)
class BeatCalls:
    classes: tuple[str, ...]
    predicted: np.ndarray  # (n,) class index
    scores: np.ndarray  # (n, k)
    r_peaks: np.ndarray  # (n,) sample index
    fs: int
    n_input: int
    n_dropped: int
    source: str  # "annotation" | "detected"

    @classmethod
    def from_prediction(cls, pred, fs: int, source: str) -> "BeatCalls":
        p = pred.prepared
        return cls(tuple(pred.classes), np.asarray(pred.predicted, dtype=np.int64),
                   np.asarray(pred.scores), np.asarray(pred.r_peaks, dtype=np.int64), int(fs),
                   int(p.n_input), int(p.n_dropped_boundary + p.n_dropped_edge), source)


def runs(predicted: np.ndarray, target: int, min_len: int = MIN_RUN) -> list[tuple[int, int]]:
    """Inclusive (first, last) index of each run of ``target`` at least ``min_len`` long."""
    out, start = [], None
    for i, p in enumerate([*np.asarray(predicted).tolist(), -1]):
        if p == target and start is None:
            start = i
        elif p != target and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def _rate(peaks: np.ndarray, fs: int) -> int:
    return int(round(60.0 / (np.diff(peaks).mean() / fs)))


def analyze(calls: BeatCalls) -> list[Finding]:
    fl = FindingList("heart")
    n = int(calls.predicted.size)
    if n == 0:
        fl.add("beats_classified", "", "no beats could be classified in this window", value=0,
               unit="beats", evidence={"n_input": calls.n_input, "n_dropped": calls.n_dropped},
               terms=("beat",))
        return fl.items

    fl.add("beats_classified", "",
           f"{n} beats classified ({calls.n_dropped} at the window edges could not be)",
           value=n, unit="beats", evidence={"n_input": calls.n_input, "n_dropped": calls.n_dropped},
           terms=("beat",))

    counts = np.bincount(calls.predicted, minlength=len(calls.classes))
    for i, c in enumerate(calls.classes):
        if counts[i] == 0:
            continue
        share = round(100.0 * counts[i] / n, 1)
        fl.add("class_share", c,
               f"{share:.1f}% of beats ({int(counts[i])} of {n}) were classified {c} "
               f"({CLASS_NAMES.get(c, c)})",
               value=share, unit="%", evidence={"count": int(counts[i]), "total": n},
               terms=(f"beat:{c}",), caveat=MODEL_CAVEAT)

    if n >= 2:
        rr = np.diff(calls.r_peaks) / calls.fs
        hr, lo, hi = (int(round(60.0 / x)) for x in (rr.mean(), rr.max(), rr.min()))
        fl.add("heart_rate", "",
               f"average heart rate {hr} beats/min in this window (range {lo} to {hi})",
               value=hr, unit="beats/min", evidence={"min": lo, "max": hi}, terms=("heart_rate",))

    for c, kind in (("V", "v_run"), ("S", "s_run")):
        if c not in calls.classes:
            continue
        found = runs(calls.predicted, calls.classes.index(c))
        if not found:
            continue
        lengths = [b - a + 1 for a, b in found]
        rates = [_rate(calls.r_peaks[a:b + 1], calls.fs) for a, b in found]
        fl.add(kind, c,
               f"{len(found)} run(s) of {MIN_RUN} or more consecutive {c} beats "
               f"(longest {max(lengths)} beats, fastest {max(rates)} beats/min)",
               value=len(found), unit="runs",
               evidence={"lengths": lengths, "rates_bpm": rates, "max_rate_bpm": max(rates),
                         "beat_indices": [[int(a), int(b)] for a, b in found]},
               terms=(f"beat:{c}", "run"), caveat=MODEL_CAVEAT)

    low = int((calls.scores.max(axis=1) < LOW_CONFIDENCE).sum())
    if low:
        fl.add("low_confidence_beats", "",
               f"{low} of {n} beats had a top score below {LOW_CONFIDENCE} (the model was unsure)",
               value=low, unit="beats", terms=("beat",), caveat=MODEL_CAVEAT)

    if "Q" in calls.classes:
        q = round(100.0 * counts[calls.classes.index("Q")] / n, 1)
        if q >= Q_SHARE_FLAG:
            fl.add("q_share_high", "Q",
                   f"{q:.1f}% of beats were unclassifiable (Q); signal quality may limit these "
                   f"findings", value=q, unit="%", terms=("beat:Q",))

    fl.add("beat_source", "",
           "beat positions came from the cardiologist's annotations"
           if calls.source == "annotation" else "beat positions were detected automatically (XQRS)")
    return fl.items
