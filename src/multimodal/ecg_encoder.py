"""ECG input -> per-beat scores and a fixed-length record summary.

Wraps :class:`mit_bih.inference.EcgClassifier`. The summary vector is what a
fusion model would consume: it has the same length for a 10-second strip and a
30-minute recording, and every element has a name a clinician can read.

The beat scores are **uncalibrated** (see ``mit_bih.inference``), so the
summary uses them as rankings and fractions, not as probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mit_bih import config as ecg_config
from mit_bih import inference as ecg_inference

from .errors import InputValidationError

MIN_BEATS = 3  # fewer leaves no beat with RR neighbours on both sides

SUMMARY_NAMES: tuple[str, ...] = (
    *(f"mean_score_{c}" for c in ecg_config.AAMI_CLASSES),
    *(f"fraction_predicted_{c}" for c in ecg_config.AAMI_CLASSES),
    "mean_rr_s",
    "sd_rr_s",
    "log_n_beats",
)


@dataclass
class EcgInput:
    """A single-lead ECG. ``r_peaks`` (sample indices) are optional."""

    signal: np.ndarray
    fs: int
    r_peaks: np.ndarray | None = None
    record_id: str | None = None

    def validate(self) -> None:
        sig = np.asarray(self.signal)
        if sig.ndim != 1 or sig.size == 0:
            raise InputValidationError(f"ECG signal must be a non-empty 1-D array, got shape {sig.shape}")
        if not isinstance(self.fs, (int, np.integer)) or self.fs <= 0:
            raise InputValidationError(f"ECG sampling rate must be a positive integer, got {self.fs!r}")
        if self.r_peaks is not None and np.asarray(self.r_peaks).size < MIN_BEATS:
            raise InputValidationError(f"at least {MIN_BEATS} R peaks are needed, got "
                                       f"{np.asarray(self.r_peaks).size}")


@dataclass
class EcgEncoding:
    summary: np.ndarray  # (len(SUMMARY_NAMES),)
    names: tuple[str, ...]
    prediction: ecg_inference.EcgPrediction
    notes: list[str] = field(default_factory=list)


class EcgEncoder:
    def __init__(self, classifier: ecg_inference.EcgClassifier) -> None:
        self.classifier = classifier

    @classmethod
    def from_bundle(cls, path: Path | str = ecg_inference.DEFAULT_BUNDLE_PATH) -> "EcgEncoder":
        return cls(ecg_inference.EcgClassifier.from_bundle(path))

    @property
    def output_dim(self) -> int:
        return len(SUMMARY_NAMES)

    def encode(self, ecg: EcgInput) -> EcgEncoding:
        ecg.validate()
        try:
            pred = self.classifier.predict_signal(ecg.signal, int(ecg.fs), ecg.r_peaks)
        except ValueError as exc:
            raise InputValidationError(f"ECG rejected: {exc}") from exc
        except Exception as exc:  # typed mit_bih errors: rate, NaN, too short
            if type(exc).__module__.startswith("mit_bih"):
                raise InputValidationError(f"ECG rejected: {exc}") from exc
            raise
        if pred.r_peaks.size == 0:
            raise InputValidationError("no classifiable beats in the ECG (too few R peaks)")
        return EcgEncoding(summary=summarize(pred), names=SUMMARY_NAMES, prediction=pred)


def summarize(pred: ecg_inference.EcgPrediction) -> np.ndarray:
    n = pred.r_peaks.size
    k = len(pred.classes)
    counts = np.bincount(pred.predicted, minlength=k)
    pre_rr = pred.prepared.rr[:, 0]  # features.RR_FEATURE_NAMES[0] == "pre_rr"
    return np.concatenate([
        pred.scores.mean(axis=0),
        counts / n,
        [float(pre_rr.mean()), float(pre_rr.std()), float(np.log(n))],
    ]).astype(np.float64)
