#!/usr/bin/env python
"""Phase 1e -- the four improvements Phase 1d's results pointed to.

Phase 1d found S recall 0.693 (up from the Random Forest's 0.271) but S
precision of only 0.294: 3,015 normal beats were called S, 98% of that arm's
false positives. It also could not rank three of its six arms, because
fold-to-fold variance (std 0.13) dwarfed the differences between them (0.007).

This addresses both, in four changes:

    1. repeated CV over several seeds, with PAIRED per-fold comparison and
       bootstrap intervals -- so a real improvement can be told from noise
    2. a rhythm-context model: a BiLSTM reads a window of neighbouring beats,
       so "early" is judged against the surrounding rhythm rather than inferred
       from one beat's numbers
    3. a robust rhythm reference: rolling median instead of a record-level mean,
       which one arrhythmic episode cannot drag around
    4. ensembling the fold models, which costs no training and cuts variance

Arms, each differing from the reference in exactly one way:

    beat_focal              Phase 1d's winner, reproduced as the reference
    beat_robust_rr          + robust rhythm reference            (change 3)
    rhythm_context          + neighbouring beats                 (change 2)
    rhythm_context_robust   both                                 (2 and 3)

Usage:
    python scripts/train_nn_v2.py \
        --beats-dir ~/physionet-data/processed/mitdb-ds12/beats \
        --out-dir   ~/physionet-data/processed/mitdb-ds12 \
        --device cuda --seeds 3
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

from ecg import (  # noqa: E402
    config,
    dataset,
    ensemble,
    evaluate,
    folds,
    sequence,
    splits,
    stats,
    thresholds,
)
from ecg.nn import checkpoint as ckpt  # noqa: E402
from ecg.nn import seqtrain, train  # noqa: E402

#: Window width for the rhythm-context arms: the beat plus two either side.
CONTEXT: int = 2

_BULKY = frozenset({"oof_proba", "oof_y"})


@dataclass(frozen=True)
class ArmV2:
    """One row. Exactly one thing differs from the reference arm."""

    name: str
    model: str  # "beat" | "rhythm"
    robust_rr: bool = False
    loss: str = "focal"
    note: str = ""


ARMS: tuple[ArmV2, ...] = (
    ArmV2("beat_focal", "beat", note="Phase 1d winner, reproduced as reference"),
    ArmV2("beat_robust_rr", "beat", robust_rr=True, note="rolling-median rhythm reference"),
    ArmV2("rhythm_context", "rhythm", note="BiLSTM over +/-2 neighbouring beats"),
    ArmV2(
        "rhythm_context_robust",
        "rhythm",
        robust_rr=True,
        note="rhythm context AND robust reference",
    ),
)

REFERENCE = "beat_focal"


def _load(beats_dir, records, arm: ArmV2):
    data = dataset.load_records(beats_dir, records, robust_rr=arm.robust_rr)
    if arm.model == "rhythm":
        return sequence.make_sequences(data, context=CONTEXT)
    return data


def _train(train_data, val_data, arm: ArmV2, cfg):
    if arm.model == "rhythm":
        return seqtrain.train_sequence_model(train_data, val_data, arm.loss, cfg)
    return train.train_model(train_data, val_data, arm.loss, cfg)


def run_repeated_cv(beats_dir, arm: ArmV2, seeds, n_folds, epochs, device) -> dict:
    """Cross-validate this arm once per seed.

    Repeating over seeds is change 1. A single CV pass gave Phase 1d five
    numbers whose spread it could not separate from real effects; several seeds
    give a distribution, and the paired comparison below gives a verdict.
    """
    rows: list[dict] = []
    oof_by_seed: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for seed in seeds:
        assignment = folds.make_folds(n_folds=n_folds, seed=seed)
        cfg = train.TrainConfig(
            epochs=epochs, seed=seed, device=device, gamma=2.0
        )

        proba_parts, y_parts = [], []
        for index in range(assignment.n_folds):
            train_records, val_records = assignment.train_val(index)
            started = time.time()

            result = _train(
                _load(beats_dir, train_records, arm),
                _load(beats_dir, val_records, arm),
                arm,
                cfg,
            )
            scored = evaluate.evaluate(result.val_y, result.val_proba.argmax(axis=1))

            rows.append(
                {
                    "seed": seed,
                    "fold": index,
                    "macro_f1_headline": scored.macro_f1_of(config.HEADLINE_CLASSES),
                    "s_recall": scored.recall_of("S"),
                    "s_precision": next(
                        m.precision for m in scored.per_class if m.name == "S"
                    ),
                    "s_f1": next(m.f1 for m in scored.per_class if m.name == "S"),
                    "v_recall": scored.recall_of("V"),
                    "best_epoch": result.best_epoch,
                    "seconds": round(time.time() - started, 1),
                }
            )
            proba_parts.append(result.val_proba)
            y_parts.append(result.val_y)

            r = rows[-1]
            print(
                f"  {arm.name} seed {seed} fold {index}: "
                f"macroF1 {r['macro_f1_headline']:.3f} "
                f"S rec {r['s_recall']:.3f} prec {r['s_precision']:.3f} "
                f"({r['seconds']}s)",
                flush=True,
            )

        oof_by_seed[seed] = (
            np.concatenate(proba_parts),
            np.concatenate(y_parts),
        )

    headline = np.array([r["macro_f1_headline"] for r in rows])
    s_recall = np.array([r["s_recall"] for r in rows])
    s_f1 = np.array([r["s_f1"] for r in rows])

    return {
        "arm": arm.name,
        "model": arm.model,
        "robust_rr": arm.robust_rr,
        "loss": arm.loss,
        "note": arm.note,
        "runs": rows,
        "mean_macro_f1": float(headline.mean()),
        "std_macro_f1": float(headline.std(ddof=1)),
        "ci_macro_f1": stats.bootstrap_ci(headline),
        "mean_s_recall": float(s_recall.mean()),
        "mean_s_f1": float(s_f1.mean()),
        "ci_s_f1": stats.bootstrap_ci(s_f1),
        "median_best_epoch": int(np.median([r["best_epoch"] for r in rows])),
        "oof_proba": oof_by_seed[seeds[0]][0],
        "oof_y": oof_by_seed[seeds[0]][1],
    }


def _clean(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj))


def _save(out_dir: Path, payload: dict, name: str = "nn_v2_cv.json") -> None:
    (out_dir / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_clean) + "\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--seeds", type=int, default=3, help="how many CV repeats (change 1)"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args(argv)

    beats_dir = Path(args.beats_dir).expanduser()
    if not beats_dir.is_dir():
        print(
            f"no beats at {beats_dir}\nRun: python scripts/preprocess_mitbih.py "
            f"--strategy ds1_ds2 --out-dir <dir>",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else beats_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints_v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = train.resolve_device(args.device)
    seeds = [config.PHASE_1D_SEED + i for i in range(args.seeds)]

    print(
        f"[setup] device={device}  seeds={seeds}  folds={args.folds}  "
        f"epochs<={args.epochs}\n"
        f"[setup] {len(ARMS)} arms x {args.seeds} seeds x {args.folds} folds = "
        f"{len(ARMS) * args.seeds * args.folds} training runs",
        flush=True,
    )

    payload: dict = {
        "phase": "1e",
        "status": "running",
        "device": str(device),
        "seeds": seeds,
        "context": CONTEXT,
        "arms": {a.name: a.note for a in ARMS},
        "cv": {},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
        },
        "command": " ".join(sys.argv),
    }

    # --- change 1: repeated cross-validation --------------------------------
    cv: dict[str, dict] = {}
    for arm in ARMS:
        print(f"[cv] {arm.name}: {arm.note}", flush=True)
        cv[arm.name] = run_repeated_cv(
            beats_dir, arm, seeds, args.folds, args.epochs, args.device
        )
        payload["cv"][arm.name] = {
            k: v for k, v in cv[arm.name].items() if k not in _BULKY
        }
        _save(out_dir, payload)

    # --- change 1: paired comparison, the analysis Phase 1d lacked ----------
    print("\n[paired] every arm against the reference, run by run", flush=True)
    comparisons: dict[str, dict] = {}
    base = cv[REFERENCE]
    for arm in ARMS:
        if arm.name == REFERENCE:
            continue
        entry = {}
        for metric in ("macro_f1_headline", "s_f1", "s_recall", "s_precision"):
            a = np.array([r[metric] for r in base["runs"]])
            b = np.array([r[metric] for r in cv[arm.name]["runs"]])
            result = stats.paired_comparison(a, b)
            entry[metric] = result.to_dict()
            print(f"  {arm.name:<24}{metric:<18}{result}", flush=True)
        comparisons[arm.name] = entry
    payload["paired_vs_reference"] = comparisons
    _save(out_dir, payload)

    winner = max(ARMS, key=lambda a: cv[a.name]["mean_s_f1"])
    print(f"\n[select] best mean S-F1: {winner.name}", flush=True)

    payload["status"] = "cv_complete"
    payload["winner"] = winner.name
    _save(out_dir, payload)

    if args.skip_test:
        print("[test] skipped; DS2 untouched", flush=True)
        return 0

    # --- final fits, ensembling, and ONE pass over DS2 ----------------------
    print("\n[final] fitting on all of DS1, then scoring DS2 once", flush=True)
    evaluations = []
    extra_rows: dict[str, dict] = {}

    for arm in ARMS:
        cfg = train.TrainConfig(
            epochs=args.epochs, seed=seeds[0], device=args.device
        )
        ds1 = _load(beats_dir, splits.DS1_RECORDS, arm)
        ds2 = _load(beats_dir, splits.DS2_RECORDS, arm)
        epochs = cv[arm.name]["median_best_epoch"]

        # change 4: one model per fold, then average their predictions.
        members: list[np.ndarray] = []
        assignment = folds.make_folds(n_folds=args.folds, seed=seeds[0])
        for index in range(assignment.n_folds):
            train_records, _ = assignment.train_val(index)
            member_train = _load(beats_dir, train_records, arm)
            if arm.model == "rhythm":
                state = seqtrain.fit_final_sequence(
                    member_train, arm.loss, cfg, epochs=epochs
                )
                model = seqtrain.build_model(cfg, state)
                members.append(
                    seqtrain.predict_proba_sequence(model, ds2, device=device)
                )
            else:
                state = train.fit_final(member_train, arm.loss, cfg, epochs=epochs)
                model = ckpt.load_checkpoint(
                    ckpt.save_checkpoint(
                        checkpoint_dir / f"{arm.name}_fold{index}.pt",
                        state,
                        arm=arm.name,
                        use_rr=True,
                        extra={"fold": index, "epochs": epochs},
                    )
                )[0]
                members.append(train.predict_proba(model, ds2, device=device))
            print(f"  {arm.name} member {index} trained", flush=True)

        single = members[0]
        averaged = ensemble.average_proba(members)

        for label, proba in (
            (arm.name, single),
            (f"{arm.name}+ensemble", averaged),
        ):
            scored = evaluate.evaluate(ds2.y, proba.argmax(axis=1), label=label)
            evaluations.append(scored)
            low, high = stats.metric_ci(
                ds2.y, proba.argmax(axis=1), metric="macro_f1_headline", n_boot=200
            )
            extra_rows[label] = {"ci_macro_f1_headline": [low, high]}
            print(
                f"[final] {label}: macroF1 "
                f"{scored.macro_f1_of(config.HEADLINE_CLASSES):.3f} "
                f"[{low:.3f}, {high:.3f}]  S rec {scored.recall_of('S'):.3f}",
                flush=True,
            )

    evaluate.write_json(
        evaluations,
        out_dir / "nn_v2_metrics.json",
        extra={**payload, "status": "complete", "test_ci": extra_rows},
    )
    print(f"\n[test] wrote {out_dir / 'nn_v2_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
