#!/usr/bin/env python
"""Phase 1g -- does an ensemble of *different losses* beat five copies of one?

Phase 1f measured ranking quality free of any decision threshold and found that
the loss functions barely differ on it (macro average precision 0.558 to 0.567
across four losses, against a 0.379 to 0.545 spread on macro-F1). On class S
specifically the ordering inverted: ``oversample`` and ``plain``, which placed
4th and 5th in Phase 1d, rank S beats better than ``focal``, which placed 1st.

The current ensemble is five copies of ``focal`` -- the weakest S ranker of the
four. Members that make *different* mistakes are what an ensemble needs, so
this trains fold models for all four losses and asks whether mixing them helps.

**Why this cannot reuse the existing checkpoints.** ``plain_ce.pt``,
``weighted_ce.pt`` and ``oversample.pt`` were each fitted on all of DS1 (see
``final_fit_note`` in the Phase 1d results), so no out-of-fold predictions exist
for them. Choosing a composition by comparing candidates on DS2 would be
selecting on the test set, which is exactly what ``thresholds.py`` forbids and
what would invalidate every DS2 number this project has published. Fold models
are therefore retrained here so selection can happen on DS1 alone.

**The selection rule is fixed before the run, in ``COMPOSITIONS`` below.** The
winner is chosen by pooled out-of-fold macro average precision and only then
scored on DS2. Every candidate's DS2 score is also reported -- looking after the
choice is made costs nothing; choosing by those numbers would cost everything.

Usage:
    ./venv/bin/python scripts/train_ensemble_mix.py --device cuda
    ./venv/bin/python scripts/train_ensemble_mix.py --device cpu --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg import (  # noqa: E402
    calibration,
    config,
    dataset,
    ensemble,
    evaluate,
    folds,
    ranking,
    splits,
)
from ecg.nn import checkpoint as ckpt  # noqa: E402
from ecg.nn import train  # noqa: E402


@dataclass(frozen=True)
class LossArm:
    """One loss. Everything else is held identical to the reference arm."""

    name: str
    loss: str
    oversample: bool = False
    note: str = ""


#: The four Phase 1d losses, as fold models on one shared fold assignment.
ARMS: tuple[LossArm, ...] = (
    LossArm("focal", "focal", note="Phase 1d winner; the current ensemble is 5 of these"),
    LossArm("plain", "plain", note="no imbalance handling; ranks S 2nd best"),
    LossArm("weighted", "weighted", note="class-weighted CE, capped"),
    LossArm("oversample", "plain", oversample=True, note="rare beats resampled; ranks S best"),
)

#: Candidate ensembles, **fixed before the run**. ``E`` drops focal on purpose:
#: it is the weakest S ranker, so "leave it out" is a real hypothesis rather
#: than a courtesy option.
COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "A_focal_only": ("focal",),
    "B_focal_plain": ("focal", "plain"),
    "C_focal_plain_oversample": ("focal", "plain", "oversample"),
    "D_all_four": ("focal", "plain", "weighted", "oversample"),
    "E_plain_oversample": ("plain", "oversample"),
}

#: The metric the winner is chosen by, on pooled out-of-fold data only.
SELECTION_METRIC = "pooled-OOF macro average precision over N/S/V/F"

REFERENCE = "A_focal_only"


def train_arm(beats_dir: Path, arm: LossArm, assignment, cfg, device: str,
              checkpoint_dir: Path) -> dict:
    """Train one loss across every fold.

    Each fold model early-stops on the fold held out from it, and that fold's
    predictions become its share of the pooled out-of-fold output. This is the
    same construction ``train_nn.run_cv`` uses to produce the predictions
    ``thresholds.py`` tunes against, so the mild optimism of stopping on the
    held-out fold is present here exactly as it is there -- and, being identical
    across arms, it cannot favour one composition over another.
    """
    oof_proba, oof_y, oof_record, ds2_members, rows = [], [], [], [], []

    ds2_records = [r for r in splits.DS2_RECORDS if r not in config.PACED_RECORDS]
    ds2 = dataset.load_records(beats_dir, ds2_records)

    for index in range(assignment.n_folds):
        train_records, val_records = assignment.train_val(index)
        started = time.time()

        result = train.train_model(
            dataset.load_records(beats_dir, train_records),
            dataset.load_records(beats_dir, val_records),
            arm.loss,
            cfg,
        )

        path = ckpt.save_checkpoint(
            checkpoint_dir / f"mix_{arm.name}_fold{index}.pt",
            result.state_dict,
            arm=f"mix_{arm.name}",
            use_rr=True,
            extra={
                "fold": index,
                "loss": arm.loss,
                "oversample": arm.oversample,
                "best_epoch": result.best_epoch,
                "phase": "1g",
            },
        )
        model = ckpt.load_checkpoint(path)[0]

        oof_proba.append(result.val_proba)
        oof_y.append(result.val_y)
        oof_record.append(result.val_record)
        ds2_members.append(train.predict_proba(model, ds2, device=device))

        scored = evaluate.evaluate(result.val_y, result.val_proba.argmax(axis=1))
        rows.append(
            {
                "fold": index,
                "best_epoch": result.best_epoch,
                "macro_f1_headline": round(
                    scored.macro_f1_of(config.HEADLINE_CLASSES), 4
                ),
                "seconds": round(time.time() - started, 1),
            }
        )
        print(
            f"  {arm.name} fold {index}: macro-F1 "
            f"{rows[-1]['macro_f1_headline']:.3f} "
            f"(epoch {result.best_epoch}, {rows[-1]['seconds']:.0f}s)",
            flush=True,
        )

    return {
        "arm": arm.name,
        "loss": arm.loss,
        "oversample": arm.oversample,
        "note": arm.note,
        "folds": rows,
        "oof_proba": np.concatenate(oof_proba),
        "oof_y": np.concatenate(oof_y),
        "oof_record": np.concatenate(oof_record),
        "ds2_members": ds2_members,
    }


def score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Every number reported for one set of predictions."""
    rankings = ranking.rank_all(y_true, proba)
    hard = evaluate.evaluate(y_true, proba.argmax(axis=1))
    by_name = {r.name: r for r in rankings}

    return {
        "macro_average_precision": round(
            float(ranking.macro_average_precision(rankings)), 4
        ),
        "s_average_precision": round(float(by_name["S"].average_precision), 4),
        "v_average_precision": round(float(by_name["V"].average_precision), 4),
        "f_average_precision": round(float(by_name["F"].average_precision), 4),
        "headline_macro_f1": round(hard.macro_f1_of(config.HEADLINE_CLASSES), 4),
        "s_recall": round(hard.recall_of("S"), 4),
        "s_precision": round(
            next(m.precision for m in hard.per_class if m.name == "S"), 4
        ),
        "brier": round(calibration.brier_score(y_true, proba), 5),
        "classwise_ece": round(
            calibration.classwise_calibration_error(y_true, proba, 15), 5
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.PHASE_1D_SEED)
    args = parser.parse_args(argv)

    beats_dir = (
        Path(args.beats_dir) if args.beats_dir
        else config.data_root() / "processed" / "mitdb-ds12" / "beats"
    )
    out_dir = Path(args.out_dir) if args.out_dir else ROOT
    checkpoint_dir = out_dir / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    assignment = folds.make_folds(n_folds=args.folds, seed=args.seed)
    assignment.validate()

    cfg_base = dict(epochs=args.epochs, seed=args.seed, device=args.device, gamma=2.0)
    device = train.resolve_device(args.device)

    print(f"[setup] {len(ARMS)} losses x {args.folds} folds = "
          f"{len(ARMS) * args.folds} training runs on {device}", flush=True)
    print(f"[setup] selection metric: {SELECTION_METRIC}", flush=True)
    print(f"[setup] candidates fixed in advance: {list(COMPOSITIONS)}", flush=True)

    trained: dict[str, dict] = {}
    for arm in ARMS:
        print(f"\n[train] {arm.name}: {arm.note}", flush=True)
        cfg = train.TrainConfig(**cfg_base, oversample=arm.oversample)
        trained[arm.name] = train_arm(
            beats_dir, arm, assignment, cfg, device, checkpoint_dir
        )

    # --- the beats must line up before anything is averaged ----------------
    reference_y = trained[ARMS[0].name]["oof_y"]
    reference_record = trained[ARMS[0].name]["oof_record"]
    for arm in ARMS[1:]:
        if not np.array_equal(trained[arm.name]["oof_y"], reference_y):
            raise SystemExit(
                f"arm {arm.name} produced out-of-fold labels in a different order "
                f"than {ARMS[0].name}. Averaging misaligned members would mix "
                f"predictions for one beat with predictions for another."
            )
        if not np.array_equal(trained[arm.name]["oof_record"], reference_record):
            raise SystemExit(f"arm {arm.name} out-of-fold records are misaligned")

    ds2_records = [r for r in splits.DS2_RECORDS if r not in config.PACED_RECORDS]
    ds2 = dataset.load_records(beats_dir, ds2_records)

    # --- step 1: choose, on DS1 out-of-fold predictions only ---------------
    print(f"\n[select] scoring {len(COMPOSITIONS)} candidates on pooled "
          f"out-of-fold predictions ({reference_y.size} beats)", flush=True)

    oof_scores: dict[str, dict] = {}
    for name, members in COMPOSITIONS.items():
        pooled = ensemble.average_proba(
            [trained[m]["oof_proba"] for m in members]
        )
        oof_scores[name] = {
            "members": list(members),
            "n_models": len(members) * args.folds,
            **score(reference_y, pooled),
        }
        print(f"  {name:26s} macro AP {oof_scores[name]['macro_average_precision']:.4f}"
              f"   S AP {oof_scores[name]['s_average_precision']:.4f}", flush=True)

    selected = max(
        oof_scores, key=lambda k: oof_scores[k]["macro_average_precision"]
    )
    print(f"\n[select] winner by {SELECTION_METRIC}: {selected}", flush=True)
    print("[select] DS2 has not been touched yet.", flush=True)

    # --- step 2: only now, DS2 --------------------------------------------
    print("\n[test] scoring every candidate on DS2 (the choice is already made)",
          flush=True)
    ds2_scores: dict[str, dict] = {}
    for name, members in COMPOSITIONS.items():
        averaged = ensemble.average_proba(
            [p for m in members for p in trained[m]["ds2_members"]]
        )
        ds2_scores[name] = {
            "members": list(members),
            "n_models": len(members) * args.folds,
            **score(ds2.y, averaged),
        }
        mark = "  <-- selected" if name == selected else ""
        print(f"  {name:26s} macro AP {ds2_scores[name]['macro_average_precision']:.4f}"
              f"   S AP {ds2_scores[name]['s_average_precision']:.4f}"
              f"   F1 {ds2_scores[name]['headline_macro_f1']:.4f}{mark}", flush=True)

    payload = {
        "phase": "1g",
        "question": "does mixing losses beat five copies of one?",
        "command": " ".join(sys.argv),
        "generated": date.today().isoformat(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection": {
            "metric": SELECTION_METRIC,
            "candidates_fixed_in_advance": list(COMPOSITIONS),
            "selected": selected,
            "reference": REFERENCE,
            "note": "Selected on DS1 out-of-fold predictions before DS2 was scored.",
        },
        "arms": {
            a.name: {
                "loss": a.loss,
                "oversample": a.oversample,
                "note": a.note,
                "folds": trained[a.name]["folds"],
            }
            for a in ARMS
        },
        "oof": {
            "n_beats": int(reference_y.size),
            "n_folds": args.folds,
            "seed": args.seed,
            "scores": oof_scores,
        },
        "ds2": {"n_beats": int(ds2.n_kept), "scores": ds2_scores},
        "environment": {
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "device": str(device),
        },
    }

    try:
        import torch

        payload["environment"]["torch"] = torch.__version__
    except Exception:  # pragma: no cover
        pass

    out = out_dir / "ensemble_mix_metrics.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)

    ref_ds2 = ds2_scores[REFERENCE]
    win_ds2 = ds2_scores[selected]
    print(
        f"\n[result] {selected} against {REFERENCE} on DS2: "
        f"macro AP {win_ds2['macro_average_precision']:.4f} vs "
        f"{ref_ds2['macro_average_precision']:.4f} "
        f"({win_ds2['macro_average_precision'] - ref_ds2['macro_average_precision']:+.4f}), "
        f"S AP {win_ds2['s_average_precision']:.4f} vs "
        f"{ref_ds2['s_average_precision']:.4f} "
        f"({win_ds2['s_average_precision'] - ref_ds2['s_average_precision']:+.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
