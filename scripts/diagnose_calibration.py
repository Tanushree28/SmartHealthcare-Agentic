#!/usr/bin/env python
"""Stage A of the calibration work: measure, decide nothing, fit nothing.

Two questions, answered from checkpoints that already exist:

1. **Is focal loss's Phase 1d win a better model or a moved threshold?**
   Scored by per-class average precision on DS2, which integrates over every
   threshold instead of picking one. If plain cross-entropy ranks S as well as
   focal does, focal bought an operating point, not a representation -- and for
   a consumer that reads probabilities rather than labels, the unweighted model
   is the better starting point.

2. **How far from calibrated are these probabilities?** Brier, NLL, ECE,
   per-class reliability, and prior drift, on the Phase 1d arms and on the
   5-fold ensemble.

Nothing here is fitted, so nothing here can leak. The correction comes next and
may only see pooled out-of-fold predictions.

Usage:
    PYTHONPATH=src python scripts/diagnose_calibration.py [--beats-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg import calibration, config, dataset, ensemble, evaluate, plots, ranking, splits  # noqa: E402
from ecg.nn import checkpoint, train  # noqa: E402

#: Phase 1d single-model arms, fitted on all of DS1. No out-of-fold predictions
#: exist for these (see ``final_fit_note`` in the Phase 1d results), so they can
#: answer question 1 on DS2 but can never be used to fit a correction.
ARMS = ("plain_ce", "weighted_ce", "focal", "oversample", "waveform_only")

#: Phase 1e fold models. These do have out-of-fold predictions.
FOLD_ARM = "beat_focal"
N_FOLDS = 5

#: The published DS2 headline macro-F1 for ``focal.pt``. Re-scoring must
#: reproduce it: the checkpoints were written by torch 2.11 on CUDA and are
#: being read by torch 2.14 on CPU. ``train.py`` documents ~4.3e-4 probability
#: drift across devices, so this is a tolerance, not an equality.
PUBLISHED_FOCAL_HEADLINE_F1 = 0.5454
REPRODUCTION_TOLERANCE = 0.005


def _default_beats_dir() -> Path:
    return config.data_root() / "processed" / "mitdb-ds12" / "beats"


def load_ds2(beats_dir: Path):
    records = [r for r in splits.DS2_RECORDS if r not in config.PACED_RECORDS]
    return dataset.load_records(beats_dir, records)


def reproduction_control(data, beats_dir: Path) -> dict:
    """Re-score the published winner and compare. Everything downstream rests here."""
    model, _ = checkpoint.load_checkpoint(Path("models") / "focal.pt")
    proba = train.predict_proba(model, data, device="cpu")
    scored = evaluate.evaluate(data.y, proba.argmax(axis=1), label="focal")
    got = scored.macro_f1_of(config.HEADLINE_CLASSES)
    delta = got - PUBLISHED_FOCAL_HEADLINE_F1

    result = {
        "published_headline_macro_f1": PUBLISHED_FOCAL_HEADLINE_F1,
        "rescored_headline_macro_f1": round(float(got), 6),
        "delta": round(float(delta), 6),
        "tolerance": REPRODUCTION_TOLERANCE,
        "n_test": int(data.n_kept),
        "passed": bool(abs(delta) <= REPRODUCTION_TOLERANCE),
    }
    if not result["passed"]:
        raise SystemExit(
            f"reproduction control FAILED: re-scored {got:.4f} against published "
            f"{PUBLISHED_FOCAL_HEADLINE_F1:.4f} (delta {delta:+.4f}, tolerance "
            f"{REPRODUCTION_TOLERANCE}). Every number downstream would describe a "
            f"different model than the one published. Stopping."
        )
    return result


def score_arm(name: str, data, models_dir: Path) -> np.ndarray:
    model, meta = checkpoint.load_checkpoint(models_dir / f"{name}.pt")
    return train.predict_proba(model, data, device="cpu"), meta


def measure(y_true: np.ndarray, proba: np.ndarray, label: str) -> dict:
    """Every Stage A number for one set of predictions."""
    rankings = ranking.rank_all(y_true, proba)
    drift = calibration.prior_drift(y_true, proba)
    hard = evaluate.evaluate(y_true, proba.argmax(axis=1), label=label)

    return {
        "label": label,
        "headline_macro_f1": round(float(hard.macro_f1_of(config.HEADLINE_CLASSES)), 4),
        "macro_average_precision": round(
            float(ranking.macro_average_precision(rankings)), 4
        ),
        "brier": round(calibration.brier_score(y_true, proba), 5),
        "nll": round(calibration.negative_log_likelihood(y_true, proba), 5),
        "ece": {
            str(b): round(calibration.expected_calibration_error(y_true, proba, b), 5)
            for b in calibration.DEFAULT_BIN_COUNTS
        },
        "classwise_ece": {
            str(b): round(
                calibration.classwise_calibration_error(y_true, proba, b), 5
            )
            for b in calibration.DEFAULT_BIN_COUNTS
        },
        "per_class": [
            {
                "class": r.name,
                "support": r.support,
                "prevalence": round(r.prevalence, 5),
                "average_precision": None if np.isnan(r.average_precision)
                else round(r.average_precision, 4),
                "roc_auc": None if np.isnan(r.roc_auc) else round(r.roc_auc, 4),
                "lift_over_chance": None if np.isnan(r.lift_over_chance)
                else round(r.lift_over_chance, 2),
                "mean_predicted": round(d.mean_predicted, 5),
                "prior_ratio": None if np.isnan(d.ratio) else round(d.ratio, 2),
            }
            for r, d in zip(rankings, drift)
        ],
    }


def pooled_oof(beats_dir: Path, models_dir: Path):
    """Out-of-fold predictions over all of DS1, one fold model per record.

    Each fold's model scores only the records held out from it, so no beat is
    ever scored by a model that trained on its patient. This is the only set a
    correction may later be fitted on.
    """
    fold_file = beats_dir.parent / "nn_folds.json"
    folds = json.loads(fold_file.read_text())["folds"]

    y_parts, p_parts, rec_parts = [], [], []
    for index, records in enumerate(folds):
        path = models_dir / f"{FOLD_ARM}_fold{index}.pt"
        model, meta = checkpoint.load_checkpoint(path)
        if meta.get("fold") != index:
            raise ValueError(
                f"{path} reports fold {meta.get('fold')}, expected {index}: "
                f"scoring a fold's held-out records with the wrong model is "
                f"exactly the leak the folds exist to prevent"
            )
        held_out = dataset.load_records(beats_dir, records)
        y_parts.append(held_out.y)
        p_parts.append(train.predict_proba(model, held_out, device="cpu"))
        rec_parts.append(held_out.record)

    return (
        np.concatenate(y_parts),
        np.concatenate(p_parts),
        np.concatenate(rec_parts),
        folds,
    )


def ensemble_proba(data, models_dir: Path) -> np.ndarray:
    members = []
    for index in range(N_FOLDS):
        model, _ = checkpoint.load_checkpoint(models_dir / f"{FOLD_ARM}_fold{index}.pt")
        members.append(train.predict_proba(model, data, device="cpu"))
    return ensemble.average_proba(members)


def draw(y_ds2, proba_by_arm, ensemble_p, oof_y, oof_p, fig_dir: Path, stamp: str):
    """Every figure. Returns the paths written."""
    written = []
    classes = config.AAMI_CLASSES

    hybrid = {k: v for k, v in proba_by_arm.items() if k != "waveform_only"}
    with_ens = {**hybrid, "ensemble": ensemble_p}

    for name in config.HEADLINE_CLASSES:
        index = classes.index(name)
        prevalence = float((y_ds2 == index).mean())

        curves = {}
        for arm, proba in with_ens.items():
            precision, recall, _ = ranking.pr_curve(y_ds2, proba, index)
            curves[arm] = (precision, recall)
        written.append(
            plots.save(
                plots.pr_curves(curves, name, prevalence),
                fig_dir / f"{stamp}-pr-{name}.png",
            )
        )

        rel = {
            arm: calibration.reliability_curve(y_ds2, proba, index, 15, name)
            for arm, proba in with_ens.items()
        }
        written.append(
            plots.save(
                plots.reliability(rel, name),
                fig_dir / f"{stamp}-reliability-{name}.png",
            )
        )

        written.append(
            plots.save(
                plots.confidence_histogram(with_ens, index, name),
                fig_dir / f"{stamp}-confidence-{name}.png",
            )
        )

    drift = {arm: calibration.prior_drift(y_ds2, p) for arm, p in with_ens.items()}
    written.append(
        plots.save(
            plots.prior_drift_bars(drift, config.HEADLINE_CLASSES),
            fig_dir / f"{stamp}-prior-drift.png",
        )
    )

    rc = {
        arm: calibration.risk_coverage_curve(y_ds2, p) for arm, p in with_ens.items()
    }
    written.append(
        plots.save(plots.risk_coverage(rc), fig_dir / f"{stamp}-risk-coverage.png")
    )

    for metric, lower_better in (("macro_average_precision", False),
                                 ("brier", True),
                                 ("classwise_ece", True)):
        values = {}
        for arm, proba in with_ens.items():
            if metric == "macro_average_precision":
                values[arm] = ranking.macro_average_precision(
                    ranking.rank_all(y_ds2, proba)
                )
            elif metric == "brier":
                values[arm] = calibration.brier_score(y_ds2, proba)
            else:
                values[arm] = calibration.classwise_calibration_error(y_ds2, proba, 15)
        written.append(
            plots.save(
                plots.metric_bars(values, metric, lower_is_better=lower_better),
                fig_dir / f"{stamp}-bars-{metric}.png",
            )
        )

    # Out-of-fold reliability for S: the set a correction will actually be fitted on.
    s = classes.index("S")
    written.append(
        plots.save(
            plots.reliability(
                {"oof-beat_focal": calibration.reliability_curve(oof_y, oof_p, s, 15, "S")},
                "S (pooled out-of-fold, DS1)",
            ),
            fig_dir / f"{stamp}-reliability-S-oof.png",
        )
    )
    return written


def _thin(values, limit: int = 400):
    """Evenly sample a long array down to ``limit`` points, endpoints kept.

    A precision-recall curve over 49,646 beats carries one point per distinct
    score. Storing all of them would put megabytes of near-duplicate numbers in
    the results file; 400 points redraw the same curve.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.size <= limit:
        return [round(float(v), 6) for v in a]
    idx = np.unique(np.linspace(0, a.size - 1, limit).astype(int))
    return [round(float(v), 6) for v in a[idx]]


def collect_curves(y_ds2, proba_by_arm, ensemble_p):
    """The arrays behind every figure, so a figure can be redrawn without re-scoring.

    ``report/`` builds its figures from committed artifacts in ``results/``
    rather than by loading checkpoints. Without these arrays the report could
    only show the scalar metrics, and the two most explanatory figures -- the
    flat precision-recall curve for S, and the reliability diagram -- could not
    be rebuilt at all.
    """
    classes = config.AAMI_CLASSES
    with_ens = {k: v for k, v in proba_by_arm.items() if k != "waveform_only"}
    with_ens["ensemble"] = ensemble_p

    pr, reliability = {}, {}
    for name in config.HEADLINE_CLASSES:
        index = classes.index(name)
        pr[name] = {}
        reliability[name] = {}
        for arm, proba in with_ens.items():
            precision, recall, _ = ranking.pr_curve(y_ds2, proba, index)
            pr[name][arm] = {
                "precision": _thin(precision),
                "recall": _thin(recall),
            }
            curve = calibration.reliability_curve(y_ds2, proba, index, 15, name)
            reliability[name][arm] = {
                "edges": [round(float(v), 6) for v in curve.edges],
                "mean_predicted": [
                    None if np.isnan(v) else round(float(v), 6)
                    for v in curve.mean_predicted
                ],
                "observed_rate": [
                    None if np.isnan(v) else round(float(v), 6)
                    for v in curve.observed_rate
                ],
                "counts": [int(v) for v in curve.counts],
            }

    risk = {}
    for arm, proba in with_ens.items():
        coverage, accuracy = calibration.risk_coverage_curve(y_ds2, proba)
        risk[arm] = {"coverage": _thin(coverage), "accuracy": _thin(accuracy)}

    return {"pr": pr, "reliability": reliability, "risk_coverage": risk}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    beats_dir = args.beats_dir or _default_beats_dir()
    stamp = date.today().isoformat()

    print(f"beats:  {beats_dir}")
    data = load_ds2(beats_dir)
    print(f"DS2:    {data.n_kept} beats, {data.n_dropped_edge} edge-dropped, "
          f"reconciles={data.reconciles()}")

    print("\n-- reproduction control --")
    control = reproduction_control(data, beats_dir)
    print(f"   focal.pt re-scored {control['rescored_headline_macro_f1']:.4f} "
          f"vs published {control['published_headline_macro_f1']:.4f} "
          f"(delta {control['delta']:+.5f})  PASS")

    print("\n-- scoring arms on DS2 --")
    proba_by_arm, meta_by_arm = {}, {}
    for arm in ARMS:
        proba_by_arm[arm], meta_by_arm[arm] = score_arm(arm, data, args.models_dir)
        print(f"   {arm}")

    print("   ensemble (5 x beat_focal folds)")
    ens = ensemble_proba(data, args.models_dir)

    print("\n-- pooled out-of-fold over DS1 --")
    oof_y, oof_p, oof_rec, folds = pooled_oof(beats_dir, args.models_dir)
    print(f"   {oof_y.size} beats from {len(set(oof_rec.tolist()))} records")

    measurements = {arm: measure(data.y, p, arm) for arm, p in proba_by_arm.items()}
    measurements["ensemble"] = measure(data.y, ens, "ensemble")
    measurements["oof_beat_focal"] = measure(oof_y, oof_p, "oof_beat_focal")

    print("\n-- figures --")
    fig_dir = args.out_dir / "figures"
    written = draw(data.y, proba_by_arm, ens, oof_y, oof_p, fig_dir, stamp)
    for path in written:
        print(f"   {path}")

    payload = {
        "phase": "1f",
        "stage": "A -- diagnosis only; nothing fitted",
        "command": " ".join(sys.argv),
        "generated": stamp,
        "reproduction_control": control,
        "ds2": {"n_beats": int(data.n_kept), "n_edge_dropped": int(data.n_dropped_edge)},
        "oof": {"n_beats": int(oof_y.size), "folds": folds},
        "checkpoint_meta": {k: v for k, v in meta_by_arm.items()},
        "measurements": measurements,
        "curves": collect_curves(data.y, proba_by_arm, ens),
        "figures": [str(p) for p in written],
        "environment": {
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    }
    out = args.out_dir / f"{stamp}-calibration-diagnosis-ds1_ds2.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
