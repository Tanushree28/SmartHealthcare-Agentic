"""Beat classification from a raw ECG signal, with saved preprocessing.

The trained checkpoints in ``models/`` hold weights only. To classify a new
signal the network also needs exactly the preprocessing its training beats
went through: the band-pass settings, the window around each R peak, the
z-score statistics fitted on the training records, and the RR-feature
definition. Those statistics live in a processed-data folder outside the
repository, so a checkpoint on its own cannot be used.

An **inference bundle** (``models/mit_bih_inference_bundle.json``) records all
of it, plus the list of member checkpoints and their SHA-256 hashes.
``prepare_beats`` then reuses the *same functions* the Phase 1 pipeline used --
``filters.denoise``, ``segment.segment_beats``, ``normalize.apply``,
``features.rr_features`` -- rather than re-implementing them, and a
real-data test checks that beats prepared from the raw record are identical to
the beats the pipeline saved.

**The outputs are not probabilities.** Phase 1f measured the focal-loss
models' mean predicted P(S) at 3.2x the true rate and P(F) at 14.5x. Until a
calibration correction is fitted (Phase 1f Stage B, not started), the scores
rank beats; they do not estimate how likely a beat is to be S or F. The bundle
carries that statement and every prediction repeats it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from common import paths, provenance

from . import config, ensemble, features, filters, labels, normalize, segment
from .dataset import BeatData
from .errors import InvalidSignalError, SamplingRateError, SignalTooShortError

BUNDLE_VERSION = 1
DEFAULT_BUNDLE_PATH = paths.PROJECT_ROOT / "models" / "mit_bih_inference_bundle.json"

#: Ensembles that can be exported. ``D_all_four`` is the Phase 1g selection --
#: chosen on DS1 out-of-fold predictions before DS2 was scored -- and is the
#: reported model. The others are kept for comparison.
ENSEMBLES: dict[str, tuple[str, ...]] = {
    "D_all_four": tuple(
        f"models/mix_{loss}_fold{i}.pt"
        for loss in ("focal", "plain", "weighted", "oversample") for i in range(5)),
    "A_focal_only": tuple(f"models/mix_focal_fold{i}.pt" for i in range(5)),
    "focal_single": ("models/focal.pt",),
}
SELECTED_ENSEMBLE = "D_all_four"

#: Published DS2 headline macro-F1 (N/S/V/F) for each exportable ensemble,
#: from results/2026-09-17-ensemble-mix-ds1_ds2.json and the 1d results.
PUBLISHED_DS2_MACRO_F1: dict[str, float] = {
    "D_all_four": 0.5554, "A_focal_only": 0.6119, "focal_single": 0.5454,
}

CALIBRATION_NOTE = (
    "Uncalibrated. Phase 1f measured mean predicted P(S) at 3.2x and P(F) at "
    "14.5x the true class rates. Treat the scores as a ranking of beats, not as "
    "probabilities, until Phase 1f Stage B fits a correction."
)

#: Refinement window for detected R peaks: +/- 50 ms.
PEAK_REFINE_SECONDS = 0.05


@dataclass(frozen=True)
class InferenceBundle:
    fs: int
    bandpass_low_hz: float
    bandpass_high_hz: float
    bandpass_order: int
    notch: bool
    samples_before: int
    samples_after: int
    norm_mean: float
    norm_std: float
    rr_robust: bool
    classes: tuple[str, ...]
    ensemble: str
    members: tuple[dict, ...]  # {"path": project-relative, "sha256": ...}
    calibration: str = CALIBRATION_NOTE
    trained_on: str = "MIT-BIH Arrhythmia Database, DS1 records (de Chazal 2004 split)"
    normalization_source: dict = field(default_factory=dict)
    version: int = BUNDLE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "InferenceBundle":
        if payload.get("version") != BUNDLE_VERSION:
            raise ValueError(f"unsupported bundle version {payload.get('version')!r}")
        payload = dict(payload)
        payload["classes"] = tuple(payload["classes"])
        payload["members"] = tuple(payload["members"])
        return cls(**payload)

    @property
    def norm_stats(self) -> normalize.NormStats:
        return normalize.NormStats(mean=self.norm_mean, std=self.norm_std)


def build_bundle(processed_dir: Path | str, ensemble_name: str = SELECTED_ENSEMBLE) -> InferenceBundle:
    """Assemble a bundle from a processed folder's normalization and checkpoints."""
    from .nn import checkpoint

    if ensemble_name not in ENSEMBLES:
        raise ValueError(f"unknown ensemble {ensemble_name!r}; choose from {sorted(ENSEMBLES)}")
    processed_dir = Path(processed_dir).expanduser()
    norm_path = processed_dir / "normalization.json"
    norm = json.loads(norm_path.read_text())
    if norm.get("fitted_on") != "train_records_only":
        raise ValueError(f"{norm_path} was not fitted on training records only")
    if list(norm["aami_class_order"]) != list(config.AAMI_CLASSES):
        raise ValueError(f"{norm_path} class order differs from config.AAMI_CLASSES")

    members = []
    for rel in ENSEMBLES[ensemble_name]:
        path = paths.resolve_project_path(rel)
        _, meta = checkpoint.load_checkpoint(path)
        if list(meta["classes"]) != list(config.AAMI_CLASSES) or meta["beat_length"] != config.BEAT_LENGTH:
            raise ValueError(f"{rel} was trained for different classes or beat length")
        if not meta["use_rr"]:
            raise ValueError(f"{rel} is waveform-only; bundles hold hybrid models")
        members.append({"path": rel, "sha256": provenance.sha256_file(path),
                        "arm": meta.get("arm"), "phase": meta.get("phase")})

    return InferenceBundle(
        fs=config.FS,
        bandpass_low_hz=config.BANDPASS_LOW_HZ,
        bandpass_high_hz=config.BANDPASS_HIGH_HZ,
        bandpass_order=config.BANDPASS_ORDER,
        notch=config.NOTCH_ENABLED_BY_DEFAULT,
        samples_before=config.SAMPLES_BEFORE,
        samples_after=config.SAMPLES_AFTER,
        norm_mean=float(norm["stats"]["mean"]),
        norm_std=float(norm["stats"]["std"]),
        rr_robust=False,
        classes=tuple(config.AAMI_CLASSES),
        ensemble=ensemble_name,
        members=tuple(members),
        normalization_source={
            "file": "normalization.json of the DS1/DS2 preprocessing run",
            "sha256": provenance.sha256_file(norm_path),
            "train_records": list(norm["train_records"]),
        },
    )


def save_bundle(path: Path | str, bundle: InferenceBundle) -> Path:
    return provenance.write_json(path, bundle.to_dict())


def load_bundle(path: Path | str = DEFAULT_BUNDLE_PATH, verify: bool = True) -> InferenceBundle:
    """Read a bundle; with ``verify``, check every member checkpoint's hash."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no inference bundle at {path}. Run: python scripts/mit_bih/export_inference_bundle.py")
    bundle = InferenceBundle.from_dict(json.loads(path.read_text()))
    if verify:
        for m in bundle.members:
            member = paths.resolve_project_path(m["path"])
            if not member.exists():
                raise FileNotFoundError(f"bundle member {m['path']} is missing")
            if provenance.sha256_file(member) != m["sha256"]:
                raise ValueError(f"bundle member {m['path']} changed since the bundle was written")
    return bundle


# ---------------------------------------------------------------------------
# Signal -> beats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedBeats:
    X: np.ndarray  # (n, BEAT_LENGTH) float32, normalized
    rr: np.ndarray  # (n, 6) float32
    r_peaks: np.ndarray  # (n,) int64 -- sample index of each classified beat
    n_input: int  # R peaks supplied or detected
    n_dropped_boundary: int  # window crosses the signal edge
    n_dropped_edge: int  # no neighbour on one side, so no RR features

    @property
    def n_beats(self) -> int:
        return int(self.r_peaks.size)

    def reconciles(self) -> bool:
        return self.n_beats + self.n_dropped_boundary + self.n_dropped_edge == self.n_input


def _validate_signal(signal, fs: int, bundle: InferenceBundle) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 1:
        raise InvalidSignalError(f"expected a 1-D single-lead signal, got shape {sig.shape}")
    if int(fs) != bundle.fs:
        raise SamplingRateError(
            f"signal is sampled at {fs} Hz but the model was trained at {bundle.fs} Hz; "
            f"resample it before inference rather than feeding mismatched windows")
    if sig.size < bundle.samples_before + bundle.samples_after:
        raise SignalTooShortError(f"signal has {sig.size} samples; one beat window needs "
                                  f"{bundle.samples_before + bundle.samples_after}")
    if not np.all(np.isfinite(sig)):
        raise InvalidSignalError("signal contains NaN or inf")
    return sig


def _validate_peaks(r_peaks, n_samples: int) -> np.ndarray:
    peaks = np.asarray(r_peaks)
    if peaks.ndim != 1:
        raise ValueError(f"r_peaks must be 1-D, got shape {peaks.shape}")
    if peaks.size and not np.issubdtype(peaks.dtype, np.integer):
        if not np.all(np.equal(np.mod(peaks, 1), 0)):
            raise ValueError("r_peaks must be integer sample indices")
    peaks = peaks.astype(np.int64)
    if peaks.size and (peaks.min() < 0 or peaks.max() >= n_samples):
        raise ValueError(f"r_peaks fall outside the signal (0..{n_samples - 1})")
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("r_peaks must be strictly increasing")
    return peaks


def detect_r_peaks(signal, fs: int) -> np.ndarray:
    """R peaks by the wfdb XQRS detector, refined to the local maximum.

    The networks were trained on beats centred on *annotated* R peaks. A
    detector places QRS marks a few samples off, so each detection is moved to
    the largest band-passed value within +/-50 ms. Detection is still an
    approximation of the annotation, and the output says which was used.
    """
    from wfdb import processing

    sig = np.asarray(signal, dtype=np.float64)
    filtered = filters.denoise(sig, fs=fs)
    detected = processing.xqrs_detect(sig=filtered, fs=fs, verbose=False)
    half = int(round(PEAK_REFINE_SECONDS * fs))
    refined = []
    for p in np.asarray(detected, dtype=np.int64):
        lo, hi = max(p - half, 0), min(p + half + 1, filtered.size)
        refined.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))
    return np.unique(np.asarray(refined, dtype=np.int64))


def prepare_beats(signal, fs: int, r_peaks, bundle: InferenceBundle) -> PreparedBeats:
    """Filter, segment, normalize and attach RR features -- as Phase 1 did."""
    sig = _validate_signal(signal, fs, bundle)
    peaks = _validate_peaks(r_peaks, sig.size)

    filtered = filters.denoise(sig, fs=bundle.fs, low=bundle.bandpass_low_hz,
                               high=bundle.bandpass_high_hz, order=bundle.bandpass_order,
                               use_notch=bundle.notch)
    # Every supplied peak is a beat to classify; "N" here only routes it
    # through the same segmentation code. The label is never used.
    mappings = labels.map_symbols(["N"] * peaks.size)
    beats = segment.segment_beats(filtered, peaks, mappings,
                                  before=bundle.samples_before, after=bundle.samples_after)

    rr, keep = features.rr_features(beats.r_peaks, fs=bundle.fs, robust=bundle.rr_robust)
    X = normalize.apply(beats.X[keep], bundle.norm_stats) if keep.size else \
        np.zeros((0, bundle.samples_before + bundle.samples_after), dtype=np.float32)

    return PreparedBeats(
        X=X.astype(np.float32),
        rr=rr.astype(np.float32),
        r_peaks=beats.r_peaks[keep].astype(np.int64),
        n_input=int(peaks.size),
        n_dropped_boundary=int(beats.n_dropped_boundary),
        n_dropped_edge=int(beats.n_kept - keep.size),
    )


# ---------------------------------------------------------------------------
# Beats -> scores
# ---------------------------------------------------------------------------


@dataclass
class EcgPrediction:
    scores: np.ndarray  # (n, n_classes) ensemble-averaged softmax, uncalibrated
    r_peaks: np.ndarray
    classes: tuple[str, ...]
    r_peaks_source: str  # "provided" | "detected (wfdb XQRS, refined)"
    prepared: PreparedBeats
    ensemble: str
    calibration: str = CALIBRATION_NOTE

    @property
    def predicted(self) -> np.ndarray:
        return self.scores.argmax(axis=1) if self.scores.size else np.zeros(0, dtype=np.int64)

    def summary(self) -> dict:
        n = int(self.r_peaks.size)
        counts = np.bincount(self.predicted, minlength=len(self.classes)) if n else \
            np.zeros(len(self.classes), dtype=np.int64)
        return {
            "n_beats_classified": n,
            "n_input_peaks": self.prepared.n_input,
            "n_dropped_boundary": self.prepared.n_dropped_boundary,
            "n_dropped_no_rr_neighbour": self.prepared.n_dropped_edge,
            "predicted_class_counts": {c: int(k) for c, k in zip(self.classes, counts)},
            "mean_scores": {c: float(v) for c, v in
                            zip(self.classes, self.scores.mean(axis=0) if n else
                                np.zeros(len(self.classes)))},
            "r_peaks_source": self.r_peaks_source,
        }


class EcgClassifier:
    """An inference bundle with its member networks loaded."""

    def __init__(self, bundle: InferenceBundle, models: list) -> None:
        if not models:
            raise ValueError("an ensemble needs at least one model")
        self.bundle = bundle
        self.models = models

    @classmethod
    def from_bundle(cls, path: Path | str = DEFAULT_BUNDLE_PATH) -> "EcgClassifier":
        from .nn import checkpoint

        bundle = load_bundle(path, verify=True)
        models = [checkpoint.load_checkpoint(paths.resolve_project_path(m["path"]))[0]
                  for m in bundle.members]
        return cls(bundle, models)

    def score_beats(self, X: np.ndarray, rr: np.ndarray) -> np.ndarray:
        """Ensemble-averaged softmax for already-prepared beats."""
        from .nn.train import predict_proba

        if X.shape[0] == 0:
            return np.zeros((0, len(self.bundle.classes)), dtype=np.float64)
        data = BeatData(X=X, rr=rr, y=np.zeros(X.shape[0], dtype=np.int64),
                        record=np.full(X.shape[0], "", dtype=object),
                        n_input=X.shape[0], n_dropped_edge=0)
        return ensemble.average_proba([predict_proba(m, data, device="cpu") for m in self.models])

    def predict_signal(self, signal, fs: int, r_peaks=None) -> EcgPrediction:
        """Classify every beat of a single-lead signal.

        ``r_peaks`` are beat locations in samples (annotations, if available).
        When omitted, they are detected; detection is less exact than the
        annotations the networks were trained on.
        """
        sig = _validate_signal(signal, fs, self.bundle)
        if r_peaks is None:
            peaks = detect_r_peaks(sig, fs)
            source = "detected (wfdb XQRS, refined to local maximum)"
        else:
            peaks = r_peaks
            source = "provided"
        prepared = prepare_beats(sig, fs, peaks, self.bundle)
        return EcgPrediction(
            scores=self.score_beats(prepared.X, prepared.rr),
            r_peaks=prepared.r_peaks,
            classes=self.bundle.classes,
            r_peaks_source=source,
            prepared=prepared,
            ensemble=self.bundle.ensemble,
        )
