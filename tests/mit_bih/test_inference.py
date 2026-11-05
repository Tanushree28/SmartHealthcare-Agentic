"""Inference from a raw signal: same preprocessing as training, verified."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common import provenance  # noqa: E402
from mit_bih import config, dataset, features, inference, normalize, pipeline  # noqa: E402
from mit_bih.errors import InvalidSignalError, SamplingRateError, SignalTooShortError  # noqa: E402
from mit_bih.nn import checkpoint  # noqa: E402
from mit_bih.nn.model import HybridBeatNet  # noqa: E402

from .synthetic import make_ecg  # noqa: E402

STATS = normalize.NormStats(mean=0.01, std=0.4)


def _bundle(members=()) -> inference.InferenceBundle:
    return inference.InferenceBundle(
        fs=config.FS, bandpass_low_hz=config.BANDPASS_LOW_HZ,
        bandpass_high_hz=config.BANDPASS_HIGH_HZ, bandpass_order=config.BANDPASS_ORDER,
        notch=False, samples_before=config.SAMPLES_BEFORE, samples_after=config.SAMPLES_AFTER,
        norm_mean=STATS.mean, norm_std=STATS.std, rr_robust=False,
        classes=tuple(config.AAMI_CLASSES), ensemble="test", members=tuple(members))


@pytest.fixture
def tiny_bundle(tmp_path):
    """Two randomly initialised hybrid networks, saved as real checkpoints."""
    members = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        path = checkpoint.save_checkpoint(tmp_path / f"m{seed}.pt",
                                          HybridBeatNet().state_dict(), arm="test", use_rr=True)
        members.append({"path": str(path), "sha256": provenance.sha256_file(path)})
    bundle = _bundle(members)
    path = inference.save_bundle(tmp_path / "bundle.json", bundle)
    return path, bundle


def test_prepared_beats_reconcile_and_are_normalized():
    sig, peaks = make_ecg(n_beats=20)

    prep = inference.prepare_beats(sig, config.FS, peaks, _bundle())

    assert prep.reconciles()
    assert prep.n_dropped_edge == 2  # first and last beat have no RR neighbour
    assert prep.X.shape == (prep.n_beats, config.BEAT_LENGTH)
    assert prep.rr.shape == (prep.n_beats, len(features.RR_FEATURE_NAMES))
    assert prep.X.dtype == np.float32


def test_prepared_beats_equal_the_training_pipeline(synthetic_record):
    """The inference path must produce the beats Phase 1 + dataset.py produce."""
    directory, name = synthetic_record["dir"], synthetic_record["name"]
    result = pipeline.process_record(directory, name)
    beats = result.beats
    rr_ref, keep = features.rr_features(beats.r_peaks)
    X_ref = normalize.apply(beats.X, STATS)[keep]

    from mit_bih import io, labels
    rec = io.load_record(directory, name)
    ann = io.load_annotations(directory, name)
    is_beat = [m.outcome == "beat" for m in labels.map_symbols(ann.symbols)]
    peaks = np.asarray(ann.samples)[is_beat]

    prep = inference.prepare_beats(rec.signal, rec.fs, peaks, _bundle())

    np.testing.assert_array_equal(prep.X, X_ref)
    np.testing.assert_array_equal(prep.rr, rr_ref)
    np.testing.assert_array_equal(prep.r_peaks, beats.r_peaks[keep])


def test_beat_crossing_the_signal_edge_is_dropped_and_counted():
    sig, peaks = make_ecg(n_beats=10, first_peak=50)  # first window starts at -50

    prep = inference.prepare_beats(sig, config.FS, peaks, _bundle())

    assert prep.n_dropped_boundary == 1
    assert prep.reconciles()


@pytest.mark.parametrize("bad,exc", [
    (np.zeros((2, 4000)), InvalidSignalError),
    (np.full(4000, np.nan), InvalidSignalError),
    (np.zeros(100), SignalTooShortError),
])
def test_malformed_signals_are_rejected(bad, exc):
    with pytest.raises(exc):
        inference.prepare_beats(bad, config.FS, [], _bundle())


def test_wrong_sampling_rate_is_rejected_not_resampled():
    sig, peaks = make_ecg()

    with pytest.raises(SamplingRateError, match="250 Hz"):
        inference.prepare_beats(sig, 250, peaks, _bundle())


@pytest.mark.parametrize("peaks", [[500, 400, 900], [500, 500, 900], [-1, 500], [10**9], [1.5, 400]])
def test_malformed_r_peaks_are_rejected(peaks):
    sig, _ = make_ecg()

    with pytest.raises(ValueError):
        inference.prepare_beats(sig, config.FS, peaks, _bundle())


def test_detected_peaks_match_synthetic_r_peaks():
    sig, peaks = make_ecg(n_beats=40)

    detected = inference.detect_r_peaks(sig, config.FS)

    assert detected.size == peaks.size
    assert np.max(np.abs(detected - peaks)) <= 2


def test_classifier_scores_are_distributions_and_labelled(tiny_bundle):
    path, _ = tiny_bundle
    clf = inference.EcgClassifier.from_bundle(path)
    sig, peaks = make_ecg(n_beats=25)

    pred = clf.predict_signal(sig, config.FS, r_peaks=peaks)

    assert pred.scores.shape == (23, len(config.AAMI_CLASSES))
    np.testing.assert_allclose(pred.scores.sum(axis=1), 1.0, rtol=1e-6)
    assert pred.r_peaks_source == "provided"
    assert "not as probabilities" in pred.calibration
    s = pred.summary()
    assert sum(s["predicted_class_counts"].values()) == s["n_beats_classified"] == 23


def test_ensemble_is_the_mean_of_its_members(tiny_bundle):
    path, _ = tiny_bundle
    clf = inference.EcgClassifier.from_bundle(path)
    sig, peaks = make_ecg(n_beats=12)
    prep = inference.prepare_beats(sig, config.FS, peaks, clf.bundle)

    from mit_bih.nn.train import predict_proba
    data = dataset.BeatData(X=prep.X, rr=prep.rr, y=np.zeros(prep.n_beats, dtype=np.int64),
                            record=np.full(prep.n_beats, "", dtype=object),
                            n_input=prep.n_beats, n_dropped_edge=0)
    singles = [predict_proba(m, data) for m in clf.models]

    np.testing.assert_allclose(clf.score_beats(prep.X, prep.rr), np.mean(singles, axis=0))


def test_without_r_peaks_the_detector_is_used_and_named(tiny_bundle):
    path, _ = tiny_bundle
    clf = inference.EcgClassifier.from_bundle(path)
    sig, _ = make_ecg(n_beats=30)

    pred = clf.predict_signal(sig, config.FS)

    assert pred.r_peaks_source.startswith("detected")
    assert pred.prepared.n_input == 30


def test_bundle_round_trips(tiny_bundle):
    path, bundle = tiny_bundle

    assert inference.load_bundle(path) == bundle


def test_changed_member_checkpoint_is_refused(tiny_bundle):
    path, bundle = tiny_bundle
    with open(bundle.members[0]["path"], "ab") as handle:
        handle.write(b"x")

    with pytest.raises(ValueError, match="changed"):
        inference.load_bundle(path)


def test_unknown_bundle_version_is_refused(tmp_path):
    payload = _bundle().to_dict()
    payload["version"] = 99
    path = tmp_path / "b.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="version"):
        inference.load_bundle(path, verify=False)


def test_committed_bundle_is_the_selected_ensemble_and_verifies():
    """models/mit_bih_inference_bundle.json and models/*.pt are tracked files."""
    bundle = inference.load_bundle()

    assert bundle.ensemble == inference.SELECTED_ENSEMBLE == "D_all_four"
    assert len(bundle.members) == 20
    assert bundle.classes == config.AAMI_CLASSES
    assert bundle.norm_std > 0
    assert "Uncalibrated" in bundle.calibration


def test_smoke_real_ensemble_on_a_synthetic_signal():
    """The shipped 20-model ensemble runs end to end on a raw signal."""
    clf = inference.EcgClassifier.from_bundle()
    sig, peaks = make_ecg(n_beats=15)

    pred = clf.predict_signal(sig, config.FS, r_peaks=peaks)

    assert pred.scores.shape == (13, 5)
    assert np.isfinite(pred.scores).all()


def test_bundle_with_wrong_window_cannot_reuse_the_networks(tiny_bundle):
    path, bundle = tiny_bundle
    clf = inference.EcgClassifier.from_bundle(path)
    clf.bundle = replace(bundle, samples_after=150)
    sig, peaks = make_ecg(n_beats=10)

    with pytest.raises(ValueError, match="beat length"):
        clf.predict_signal(sig, config.FS, r_peaks=peaks)


@pytest.mark.requires_data
def test_raw_records_reproduce_the_saved_ds2_beats():
    """Real data: the bundle's preprocessing is the preprocessing behind the models.

    Exact against the Phase 1 pipeline run today. Against the beats saved on
    2026-08-27 the agreement is within one float32 step (<= 4.8e-7): the
    unmodified pipeline itself now differs from its saved output by that much
    (IMPLEMENTATION.md, Problem 18), so exact equality there is not expected.
    """
    import json

    from mit_bih import io, labels

    processed = config.data_root() / config.PROCESSED_DIRNAME / "mitdb-ds12"
    if not (processed / "beats").is_dir() or not config.mitdb_dir().is_dir():
        pytest.skip("needs raw MIT-BIH and the DS1/DS2 processed beats")
    bundle = inference.load_bundle()
    stats = normalize.NormStats.from_dict(
        json.loads((processed / "normalization.json").read_text())["stats"])
    for name in ("100", "208", "232"):
        saved = dataset.load_record(processed / "beats", name)
        today = pipeline.process_record(config.mitdb_dir(), name).beats
        rr_today, keep = features.rr_features(today.r_peaks)
        rec = io.load_record(config.mitdb_dir(), name)
        ann = io.load_annotations(config.mitdb_dir(), name)
        is_beat = [m.outcome == "beat" for m in labels.map_symbols(ann.symbols)]
        prep = inference.prepare_beats(rec.signal, rec.fs, np.asarray(ann.samples)[is_beat], bundle)

        np.testing.assert_array_equal(prep.X, normalize.apply(today.X, stats)[keep])
        np.testing.assert_array_equal(prep.rr, rr_today)
        np.testing.assert_array_equal(prep.rr, saved.rr)
        np.testing.assert_allclose(prep.X, saved.X, rtol=0, atol=1e-6)
