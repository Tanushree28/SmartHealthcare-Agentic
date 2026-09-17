#!/usr/bin/env python
"""Run the Phase 1d imbalance study.

Six rows, one variable each:

    plain_ce        no imbalance handling at all -- the "before"
    weighted_ce     class-weighted cross-entropy, capped weights
    focal           focal loss, gamma=2
    tuned           the better of weighted_ce / focal, plus per-class thresholds
    oversample      rare beats resampled to balance the batches
    waveform_only   the RR branch removed, best imbalance setting

Protocol, from the spec section 6: every decision is made by cross-validation
over DS1. DS2 is scored exactly once, at the end, for reporting only.

Usage:
    ./venv/bin/python scripts/train_nn.py \
        --beats-dir ~/physionet-data/processed/mitdb-ds12/beats \
        --out-dir   ~/physionet-data/processed/mitdb-ds12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg import config, dataset, evaluate, folds, splits, thresholds  # noqa: E402
from ecg.nn import checkpoint as ckpt  # noqa: E402
from ecg.nn import train  # noqa: E402

#: Candidate caps for the class weights, selected by cross-validation on the
#: weighted arm and then reused. Uncapped, Q's 8 beats outweigh N's 35,501 by
#: the raw count ratio (~4,438x) and dominate the gradient.
WEIGHT_CAPS: tuple[float, ...] = (10.0, 25.0, 50.0)


@dataclass(frozen=True)
class Arm:
    """One row of the study. Exactly one thing differs from plain_ce."""

    name: str
    loss: str
    oversample: bool = False
    use_rr: bool = True
    tune: bool = False
    reuses: str | None = None  # set only for the threshold arm
    note: str = ""


ARMS: tuple[Arm, ...] = (
    Arm("plain_ce", "plain", note="no imbalance handling -- the before-picture"),
    Arm("weighted_ce", "weighted", note="class-weighted CE, capped weights"),
    Arm("focal", "focal", note="focal loss, gamma=2"),
    Arm(
        "tuned",
        "weighted",
        tune=True,
        reuses="best_of(weighted_ce, focal)",
        note="per-class thresholds on pooled out-of-fold predictions",
    ),
    Arm("oversample", "plain", oversample=True, note="rare beats resampled"),
    Arm(
        "waveform_only",
        "weighted",
        use_rr=False,
        note="RR branch removed -- isolates architecture from imbalance",
    ),
)


def run_cv(beats_dir, assignment, arm, cfg, label=None) -> dict:
    """Train one arm across every fold; return fold scores and pooled OOF output."""
    name = label or arm.name
    fold_rows: list[dict] = []
    oof_proba: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []

    for index in range(assignment.n_folds):
        train_records, val_records = assignment.train_val(index)
        train_data = dataset.load_records(beats_dir, train_records)
        val_data = dataset.load_records(beats_dir, val_records)

        started = time.time()
        result = train.train_model(train_data, val_data, arm.loss, cfg)
        scored = evaluate.evaluate(
            result.val_y, result.val_proba.argmax(axis=1), label=f"{name} fold{index}"
        )

        fold_rows.append(
            {
                "fold": index,
                "val_records": list(val_records),
                "best_epoch": result.best_epoch,
                "macro_f1": scored.macro_f1,
                "macro_f1_headline": scored.macro_f1_of(config.HEADLINE_CLASSES),
                "macro_recall": scored.macro_recall,
                "s_recall": scored.recall_of("S"),
                "v_recall": scored.recall_of("V"),
                "seconds": round(time.time() - started, 1),
            }
        )
        oof_proba.append(result.val_proba)
        oof_y.append(result.val_y)

        row = fold_rows[-1]
        print(
            f"  {name} fold {index}: macro-F1 {row['macro_f1_headline']:.3f} "
            f"S {row['s_recall']:.3f} (epoch {row['best_epoch']}, {row['seconds']}s)",
            flush=True,
        )

    headline = np.array([r["macro_f1_headline"] for r in fold_rows])
    return {
        "arm": name,
        "loss": arm.loss,
        "oversample": arm.oversample,
        "use_rr": arm.use_rr,
        "weight_cap": cfg.weight_cap,
        "note": arm.note,
        "folds": fold_rows,
        "mean_macro_f1": float(headline.mean()),
        "std_macro_f1": float(headline.std(ddof=1)) if headline.size > 1 else 0.0,
        "mean_s_recall": float(np.mean([r["s_recall"] for r in fold_rows])),
        "median_best_epoch": int(np.median([r["best_epoch"] for r in fold_rows])),
        "oof_proba": np.concatenate(oof_proba),
        "oof_y": np.concatenate(oof_y),
    }


#: Arrays too large to belong in the run JSON. Everything else is kept --
#: an earlier version filtered on the "oof_" prefix and silently dropped
#: ``oof_macro_f1`` and ``oof_s_recall`` along with the arrays, leaving the
#: tuned arm reading as 0.000 in the artifact while the log had the real value.
_BULKY = frozenset({"oof_proba", "oof_y"})


def _write_progress(out_dir: Path, payload: dict) -> None:
    """Persist what is known so far.

    A Colab session that drops at minute 50 must not cost the whole run, so this
    is called after every arm rather than once at the end.
    """
    (out_dir / "nn_cv.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_clean) + "\n"
    )


def _clean(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.PHASE_1D_SEED)
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument(
        "--skip-cap-sweep",
        action="store_true",
        help=f"use the default cap instead of selecting from {WEIGHT_CAPS}",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="where final weights are written (default: <out-dir>/checkpoints)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="run cross-validation only; do not touch DS2",
    )
    args = parser.parse_args(argv)

    beats_dir = Path(args.beats_dir).expanduser()
    if not beats_dir.is_dir():
        print(
            f"no beats at {beats_dir}\n"
            f"Run: ./venv/bin/python scripts/preprocess_mitbih.py "
            f"--strategy ds1_ds2 --out-dir ~/physionet-data/processed/mitdb-ds12",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else beats_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser()
        if args.checkpoint_dir
        else out_dir / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = train.resolve_device(args.device)
    print(
        f"[setup] device={device}  seed={args.seed}  epochs<={args.epochs}\n"
        f"[setup] checkpoints -> {checkpoint_dir}",
        flush=True,
    )

    assignment = folds.make_folds(n_folds=args.folds, seed=args.seed)
    assignment.write_json(out_dir / "nn_folds.json")

    def cfg_for(arm: Arm, cap: float) -> train.TrainConfig:
        return train.TrainConfig(
            epochs=args.epochs,
            seed=args.seed,
            oversample=arm.oversample,
            use_rr=arm.use_rr,
            weight_cap=cap,
            device=args.device,
        )

    progress: dict = {
        "phase": "1d",
        "status": "running",
        "seed": args.seed,
        "device": str(device),
        "device_note": (
            "Every arm ran on this device. CPU, MPS and CUDA kernels differ "
            "enough to shift probabilities by ~0.07, so arms are comparable "
            "only within one device."
        ),
        "folds": assignment.to_dict(),
        "cv": {},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
        },
        "command": " ".join(sys.argv),
    }

    weighted_arm = next(a for a in ARMS if a.name == "weighted_ce")

    # --- select the weight cap on the weighted arm, by cross-validation ------
    cap_runs: dict[float, dict] = {}
    if args.skip_cap_sweep:
        chosen_cap = train.TrainConfig().weight_cap
        print(f"[cap] sweep skipped; using default cap {chosen_cap}", flush=True)
    else:
        for cap in WEIGHT_CAPS:
            print(f"[cap] weighted_ce with cap={cap}", flush=True)
            cap_runs[cap] = run_cv(
                beats_dir, assignment, weighted_arm, cfg_for(weighted_arm, cap),
                label=f"weighted_ce(cap={cap})",
            )
            progress["cv"][f"weighted_ce(cap={cap})"] = {
                k: v for k, v in cap_runs[cap].items() if k not in _BULKY
            }
            _write_progress(out_dir, progress)
        chosen_cap = max(cap_runs, key=lambda c: cap_runs[c]["mean_macro_f1"])
        print(
            "[cap] selected "
            + f"{chosen_cap} from "
            + ", ".join(
                f"{c}:{cap_runs[c]['mean_macro_f1']:.3f}" for c in WEIGHT_CAPS
            ),
            flush=True,
        )

    # --- cross-validation: every selection decision happens here -------------
    cv: dict[str, dict] = {}
    for arm in ARMS:
        if arm.tune:
            continue  # handled below; trains nothing of its own
        if arm.name == "weighted_ce" and chosen_cap in cap_runs:
            cv[arm.name] = cap_runs[chosen_cap]
            cv[arm.name]["arm"] = "weighted_ce"
            continue
        print(f"[cv] {arm.name}: {arm.note}", flush=True)
        cv[arm.name] = run_cv(beats_dir, assignment, arm, cfg_for(arm, chosen_cap))
        progress["cv"][arm.name] = {
            k: v for k, v in cv[arm.name].items() if k not in _BULKY
        }
        _write_progress(out_dir, progress)

    # --- arm 3: thresholds on pooled out-of-fold predictions -----------------
    source = max(("weighted_ce", "focal"), key=lambda n: cv[n]["mean_macro_f1"])
    tuned_thresholds = thresholds.tune_thresholds(
        cv[source]["oof_y"], cv[source]["oof_proba"]
    )
    tuned_oof = evaluate.evaluate(
        cv[source]["oof_y"],
        thresholds.apply_thresholds(cv[source]["oof_proba"], tuned_thresholds),
    )
    cv["tuned"] = {
        "arm": "tuned",
        "reuses": source,
        "thresholds": tuned_thresholds.tolist(),
        "oof_macro_f1": tuned_oof.macro_f1_of(config.HEADLINE_CLASSES),
        "oof_s_recall": tuned_oof.recall_of("S"),
        "median_best_epoch": cv[source]["median_best_epoch"],
        "weight_cap": chosen_cap,
        "note": f"per-class thresholds tuned on {source}'s pooled OOF predictions",
    }
    print(
        f"[cv] tuned: reuses {source}, OOF macro-F1 "
        f"{cv['tuned']['oof_macro_f1']:.3f}, S {tuned_oof.recall_of('S'):.3f}, "
        f"thresholds {np.round(tuned_thresholds, 3).tolist()}",
        flush=True,
    )

    payload = {
        **progress,
        "status": "cv_complete",
        "weight_cap_selected": chosen_cap,
        "weight_cap_candidates": list(WEIGHT_CAPS),
        "weight_cap_cv": {str(c): cap_runs[c]["mean_macro_f1"] for c in cap_runs},
        "cv": {
            name: {k: v for k, v in row.items() if k not in _BULKY}
            for name, row in cv.items()
        },
    }
    _write_progress(out_dir, payload)
    print(f"[cv] wrote {out_dir / 'nn_cv.json'}", flush=True)

    if args.skip_test:
        print("[test] skipped by --skip-test; DS2 untouched", flush=True)
        return 0

    # --- final fits on all of DS1, then ONE pass over DS2 --------------------
    print("[final] fitting on all of DS1, then scoring DS2 once", flush=True)
    ds1 = dataset.load_records(beats_dir, splits.DS1_RECORDS)
    ds2 = dataset.load_records(beats_dir, splits.DS2_RECORDS)

    evaluations = []
    for arm in ARMS:
        train_arm = arm
        if arm.tune:
            # Reuses the source arm's configuration; only the decision rule
            # differs, so it must not be trained again with different settings.
            train_arm = next(a for a in ARMS if a.name == cv["tuned"]["reuses"])

        epochs = cv[train_arm.name]["median_best_epoch"]
        started = time.time()
        state = train.fit_final(
            ds1, train_arm.loss, cfg_for(train_arm, chosen_cap), epochs=epochs
        )

        ckpt.save_checkpoint(
            checkpoint_dir / f"{arm.name}.pt",
            state,
            arm=arm.name,
            use_rr=train_arm.use_rr,
            extra={
                "loss": train_arm.loss,
                "oversample": train_arm.oversample,
                "weight_cap": chosen_cap,
                "epochs": epochs,
                "seed": args.seed,
                "device": str(device),
                "thresholds": cv["tuned"]["thresholds"] if arm.tune else None,
                "trained_on": "DS1 (all 22 records)",
                "epochs_note": (
                    "Fixed budget -- the median stopping epoch across this "
                    "arm's CV folds. No held-out set remained to early-stop on."
                ),
            },
        )

        net, _ = ckpt.load_checkpoint(checkpoint_dir / f"{arm.name}.pt")
        proba = train.predict_proba(net, ds2, device=device)

        if arm.tune:
            y_pred = thresholds.apply_thresholds(
                proba, np.asarray(cv["tuned"]["thresholds"])
            )
        else:
            y_pred = proba.argmax(axis=1)

        scored = evaluate.evaluate(ds2.y, y_pred, label=arm.name)
        evaluations.append(scored)
        print(
            f"[final] {arm.name}: {epochs} epochs, {time.time() - started:.0f}s",
            flush=True,
        )
        print(evaluate.format_report(scored), flush=True)
        print("", flush=True)

    evaluate.write_json(
        evaluations,
        out_dir / "nn_metrics.json",
        extra={
            **payload,
            "test_records": list(splits.DS2_RECORDS),
            "n_test_beats": int(ds2.n_kept),
            "final_fit_note": (
                "Final models were fitted on all of DS1 for a fixed number of "
                "epochs -- the median stopping epoch across that arm's folds -- "
                "because no held-out set remains to early-stop against."
            ),
        },
    )
    print(f"[test] wrote {out_dir / 'nn_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
