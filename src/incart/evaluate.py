"""Beats at the model rate, the beat-position check, metrics and decision rules.

Beats are prepared by ``mit_bih.inference.prepare_beats`` -- the same filter,
window, normalization and RR features the networks were trained with -- after
the signal and its annotation positions are moved to the model's 360 Hz.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mit_bih import config as mit_config
from mit_bih import evaluate as mit_evaluate
from mit_bih import filters, inference
from mit_bih.errors import EcgError

from . import config, resample
from .io import LabelledRecord

# ---------------------------------------------------------------------------
# Beats at the model rate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordBeats:
    """One record's beats ready to score, and where every annotated beat went."""

    name: str
    X: np.ndarray
    rr: np.ndarray
    y: np.ndarray  # int64, true AAMI class index
    r_peaks: np.ndarray  # int64, at the model rate
    n_annotated: int  # beat annotations in the record file
    n_out_of_range: int  # moved outside the resampled signal
    n_collided: int  # refined variant: two marks refined onto one peak
    n_dropped_boundary: int  # window crosses the signal edge
    n_dropped_edge: int  # no neighbour on one side, so no RR features

    @property
    def n_beats(self) -> int:
        return int(self.r_peaks.size)

    def reconciles(self) -> bool:
        return (self.n_beats + self.n_out_of_range + self.n_collided
                + self.n_dropped_boundary + self.n_dropped_edge) == self.n_annotated

    def counts(self) -> dict:
        return {
            "annotated": self.n_annotated,
            "classified": self.n_beats,
            "out_of_range": self.n_out_of_range,
            "collided": self.n_collided,
            "dropped_boundary": self.n_dropped_boundary,
            "dropped_no_rr_neighbour": self.n_dropped_edge,
        }


def band_passed(signal, bundle) -> np.ndarray:
    """The signal filtered exactly as the bundle's preprocessing filters it."""
    return filters.denoise(np.asarray(signal, dtype=np.float64), fs=bundle.fs,
                           low=bundle.bandpass_low_hz, high=bundle.bandpass_high_hz,
                           order=bundle.bandpass_order, use_notch=bundle.notch)


def _window_values(filtered, marks, fs: int, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    """``|filtered|`` in a +/- window around each mark; outside the signal = -1."""
    f = np.abs(np.asarray(filtered, dtype=np.float64))
    half = int(round(window_s * fs))
    rel = np.arange(-half, half + 1)
    idx = np.asarray(marks, dtype=np.int64)[:, None] + rel[None, :]
    inside = (idx >= 0) & (idx < f.size)
    values = np.where(inside, f[np.clip(idx, 0, f.size - 1)], -1.0)
    return values, rel


def position_offsets(filtered, marks, fs: int,
                     window_s: float = inference.PEAK_REFINE_SECONDS) -> np.ndarray:
    """Signed samples from each mark to the largest ``|filtered|`` within the window.

    The same refinement ``mit_bih.inference.detect_r_peaks`` applies; ties go
    to the earliest sample, as ``np.argmax`` does there.
    """
    if np.asarray(marks).size == 0:
        return np.zeros(0, dtype=np.int64)
    values, rel = _window_values(filtered, marks, fs, window_s)
    return rel[np.argmax(values, axis=1)].astype(np.int64)


def peak_amplitudes(filtered, marks, fs: int,
                    window_s: float = inference.PEAK_REFINE_SECONDS) -> np.ndarray:
    """The largest ``|filtered|`` within the window around each mark, in mV."""
    if np.asarray(marks).size == 0:
        return np.zeros(0, dtype=np.float64)
    values, _ = _window_values(filtered, marks, fs, window_s)
    return values.max(axis=1)


def refined_marks(filtered, marks, labels, fs: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Marks moved to the local maximum; a later mark landing on the same peak is dropped.

    Returns ``(marks, labels, n_collided)``. Refined positions never cross
    (each window's earliest maximum), so collisions are always adjacent.
    """
    marks = np.asarray(marks, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    moved = marks + position_offsets(filtered, marks, fs)
    keep = np.ones(moved.size, dtype=bool)
    keep[1:] = np.diff(moved) != 0
    return moved[keep], labels[keep], int(moved.size - np.count_nonzero(keep))


def at_model_rate(rec: LabelledRecord, bundle,
                  round_trip_via: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(signal, marks, labels, n_out_of_range)``, all at ``bundle.fs``.

    A record at another rate is resampled. ``round_trip_via`` sends a record
    already at the model rate to that rate and back (the resampling check).
    """
    signal, marks, labels = rec.signal, rec.beat_samples, rec.beat_labels
    if round_trip_via is not None:
        if rec.fs != bundle.fs:
            raise ValueError(f"round trip needs a record at {bundle.fs} Hz; {rec.name} is at {rec.fs} Hz")
        signal, moved = resample.round_trip(signal, marks, rec.fs, round_trip_via)
    elif rec.fs != bundle.fs:
        signal, moved = resample.resample_record(signal, marks, rec.fs, bundle.fs)
    else:
        return np.asarray(signal, dtype=np.float64), marks, labels, 0
    return signal, moved.samples, labels[moved.keep], moved.n_dropped


def record_beats(rec: LabelledRecord, bundle, *, round_trip_via: int | None = None,
                 refine: bool = False) -> RecordBeats:
    """Prepare one record's labelled beats for scoring."""
    signal, marks, labels, n_out = at_model_rate(rec, bundle, round_trip_via)
    n_collided = 0
    if refine:
        marks, labels, n_collided = refined_marks(band_passed(signal, bundle), marks, labels, bundle.fs)

    prepared = inference.prepare_beats(signal, bundle.fs, marks, bundle)
    label_at = dict(zip(marks.tolist(), labels.tolist()))
    y = np.array([label_at[int(p)] for p in prepared.r_peaks], dtype=np.int64)
    return RecordBeats(
        name=rec.name,
        X=prepared.X,
        rr=prepared.rr,
        y=y,
        r_peaks=prepared.r_peaks,
        n_annotated=rec.n_beats,
        n_out_of_range=n_out,
        n_collided=n_collided,
        n_dropped_boundary=prepared.n_dropped_boundary,
        n_dropped_edge=prepared.n_dropped_edge,
    )


# ---------------------------------------------------------------------------
# Beat-position check
# ---------------------------------------------------------------------------


def record_positions(rec: LabelledRecord, bundle) -> tuple[np.ndarray, np.ndarray]:
    """Offsets (samples at ``bundle.fs``) and peak amplitudes (mV), as marked."""
    signal, marks, _, _ = at_model_rate(rec, bundle)
    filtered = band_passed(signal, bundle)
    return position_offsets(filtered, marks, bundle.fs), peak_amplitudes(filtered, marks, bundle.fs)


def summarise_positions(offsets, peaks_mv, fs: int) -> dict:
    """Median and interquartile range of the offsets, and the median peak in mV."""
    off = np.asarray(offsets, dtype=np.float64)
    if off.size == 0:
        return {"n": 0, "median_samples": None, "median_ms": None, "q1_ms": None,
                "q3_ms": None, "median_peak_mv": None}
    q1, median, q3 = np.percentile(off, [25, 50, 75])
    to_ms = 1000.0 / fs
    return {
        "n": int(off.size),
        "median_samples": float(median),
        "median_ms": float(median * to_ms),
        "q1_ms": float(q1 * to_ms),
        "q3_ms": float(q3 * to_ms),
        "median_peak_mv": float(np.median(np.asarray(peaks_mv, dtype=np.float64))),
    }


# ---------------------------------------------------------------------------
# Metrics from confusion matrices, and the patient bootstrap
# ---------------------------------------------------------------------------

_HEADLINE_INDEX = [mit_config.AAMI_CLASSES.index(c) for c in mit_config.HEADLINE_CLASSES]


def confusion(y, pred, n_classes: int = len(mit_config.AAMI_CLASSES)) -> np.ndarray:
    """Counts, rows = true class, columns = predicted class."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if y.shape != pred.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {pred.shape}")
    return np.bincount(y * n_classes + pred, minlength=n_classes * n_classes) \
        .reshape(n_classes, n_classes).astype(np.int64)


def headline_f1_from_matrix(matrix):
    """Mean F1 over N/S/V/F from a confusion matrix, or a stack of them.

    An undefined precision, recall or F1 counts as 0, as in
    ``mit_bih.evaluate``; no division by zero is ever attempted.
    """
    m = np.asarray(matrix, dtype=np.float64)
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    support = m.sum(axis=-1)
    predicted = m.sum(axis=-2)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    out = f1[..., _HEADLINE_INDEX].mean(axis=-1)
    return float(out) if np.ndim(out) == 0 else out


def sum_by_group(matrices: Mapping[str, np.ndarray], group_of: Mapping[str, str]) -> dict[str, np.ndarray]:
    """Add each record's confusion matrix into its group's."""
    out: dict[str, np.ndarray] = {}
    for record, matrix in matrices.items():
        group = group_of[record]
        out[group] = out.get(group, 0) + np.asarray(matrix, dtype=np.int64)
    return out


def grouped_bootstrap_ci(matrices: Mapping[str, np.ndarray], n_boot: int = config.N_BOOT,
                         seed: int = config.SEED, alpha: float = config.ALPHA) -> tuple[float, float]:
    """Percentile interval for the headline macro-F1, resampling whole groups.

    Each draw picks as many groups as there are, with replacement, and pools
    all their beats. Beats from one patient are not independent, so resampling
    patients rather than beats is what makes the interval honest.
    """
    if not matrices:
        raise ValueError("cannot bootstrap without any groups")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    keys = sorted(matrices)
    stack = np.stack([np.asarray(matrices[k], dtype=np.float64) for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    scores = headline_f1_from_matrix(stack[draws].sum(axis=1))
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def record_row(name: str, y, pred) -> dict:
    """One row of the per-record table; a record with no beats has no metrics."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if y.size == 0:
        return {"record": name, "n_beats": 0, "headline_macro_f1": None, "metrics": None}
    result = mit_evaluate.evaluate(y, pred, label=name)
    return {
        "record": name,
        "n_beats": int(y.size),
        "headline_macro_f1": result.macro_f1_of(mit_config.HEADLINE_CLASSES),
        "metrics": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Decision rules, fixed before the run (spec section 3)
# ---------------------------------------------------------------------------


def holds_up(reference: float, ci: tuple[float, float]) -> bool:
    """The model holds up if the DS2 headline falls inside INCART's 95% interval."""
    low, high = ci
    return low <= reference <= high


def resampling_matters(original: float, round_trip: float,
                       threshold: float = config.RESAMPLING_THRESHOLD) -> bool:
    return abs(round_trip - original) > threshold


def position_rule_fires(incart_median_samples: float, ds2_median_samples: float,
                        threshold: int = config.POSITION_THRESHOLD_SAMPLES) -> bool:
    return abs(incart_median_samples - ds2_median_samples) > threshold


# ---------------------------------------------------------------------------
# Guards: the DS2 reference must exist, and INCART is scored once
# ---------------------------------------------------------------------------


class PreconditionError(EcgError):
    """A run was started before what it depends on, or would repeat a finished one."""


def read_resampling_verdict(path: Path | str, ensemble: str | None = None) -> dict:
    """The resampling check's result; refuse unless it reproduced 0.5554.

    With ``ensemble``, also refuse a check run on a different model: its DS2
    score would not be the reference for this one.
    """
    path = Path(path)
    if not path.is_file():
        raise PreconditionError(
            f"no resampling check at {path}. Run it first: "
            f"python scripts/incart/resampling_check.py")
    payload = json.loads(path.read_text())
    if payload.get("reproduces_published") is not True:
        published = inference.PUBLISHED_DS2_MACRO_F1[inference.SELECTED_ENSEMBLE]
        raise PreconditionError(
            f"the resampling check in {path} did not reproduce the published DS2 "
            f"macro-F1 {published:.4f}, so the DS2 reference is not trustworthy. "
            f"Find out why before scoring INCART.")
    if ensemble is not None and payload.get("ensemble") != ensemble:
        raise PreconditionError(
            f"the resampling check in {path} scored {payload.get('ensemble')!r}, not "
            f"{ensemble!r}. Re-run scripts/incart/resampling_check.py with the same bundle.")
    return payload


def refuse_to_overwrite(path: Path | str, overwrite: bool) -> None:
    """INCART is scored once: refuse to replace a finished run unless asked to."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise PreconditionError(
            f"{path} already exists. INCART is scored once; pass --overwrite only to "
            f"replace a broken run, and record why in the write-up.")
