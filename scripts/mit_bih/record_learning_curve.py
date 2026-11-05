#!/usr/bin/env python3
"""Re-train one cross-validation fold locally and save its per-epoch history.

    python scripts/mit_bih/record_learning_curve.py
    python scripts/mit_bih/record_learning_curve.py --arm focal --fold 0 --device cpu

The Phase 1d-1g Colab runs saved each fold's best epoch and final score, but
not the epoch-by-epoch history, so no learning curve of the published models
exists. This script re-runs one fold with the Phase 1d protocol (same folds,
seed, loss, weight cap, epoch limit and early stopping, via the same
``train.train_model``) and writes what the training loop records per epoch:
training loss and validation macro-F1. It does not record validation loss or
accuracy, because the training loop does not compute them.

The result is a re-run on this machine, not the published model. Its fold-0
score is compared with the published fold-0 score in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import paths, provenance  # noqa: E402
from mit_bih import config, dataset, folds  # noqa: E402
from mit_bih.nn import train  # noqa: E402

PUBLISHED = paths.PROJECT_ROOT / "results" / "2026-09-16-nn-ds1_ds2.json"
ARM_LOSS = {"focal": "focal", "weighted_ce": "weighted", "plain_ce": "plain"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beats-dir", type=Path,
                        default=config.data_root() / config.PROCESSED_DIRNAME / "mitdb-ds12" / "beats")
    parser.add_argument("--arm", choices=sorted(ARM_LOSS), default="focal")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40, help="Phase 1d limit (default 40)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path,
                        default=paths.PROJECT_ROOT / "PlanDoc" / "integration_results"
                        / "mitbih_learning_curve_fold0_focal.json")
    args = parser.parse_args(argv)

    published = json.loads(PUBLISHED.read_text())
    cap = float(published["cv"][args.arm]["weight_cap"])
    assignment = folds.make_folds(n_folds=5, seed=config.PHASE_1D_SEED)
    train_records, val_records = assignment.train_val(args.fold)
    published_fold = published["cv"][args.arm]["folds"][args.fold]
    if sorted(published_fold["val_records"]) != sorted(val_records):
        raise SystemExit("fold assignment differs from the published one; refusing to compare")

    cfg = train.TrainConfig(epochs=args.epochs, seed=config.PHASE_1D_SEED,
                            weight_cap=cap, device=args.device)
    train_data = dataset.load_records(args.beats_dir, train_records)
    val_data = dataset.load_records(args.beats_dir, val_records)
    print(f"{args.arm} fold {args.fold}: {train_data.n_kept:,} training beats "
          f"({len(train_records)} records), {val_data.n_kept:,} validation beats "
          f"({len(val_records)} records)", flush=True)

    t0 = time.time()
    result = train.train_model(train_data, val_data, ARM_LOSS[args.arm], cfg)
    seconds = round(time.time() - t0, 1)
    best = next(h for h in result.history if h["epoch"] == result.best_epoch)

    payload = {
        "what": (f"Local re-run of Phase 1d {args.arm}, cross-validation fold {args.fold}, "
                 f"to record per-epoch history. Not the published model."),
        "arm": args.arm,
        "fold": args.fold,
        "train_records": list(train_records),
        "val_records": list(val_records),
        "n_train_beats": int(train_data.n_kept),
        "n_val_beats": int(val_data.n_kept),
        "config": {"epochs_max": args.epochs, "patience": cfg.patience, "seed": cfg.seed,
                   "weight_cap": cap, "lr": cfg.lr, "batch_size": cfg.batch_size},
        "device": str(train.resolve_device(args.device)),
        "history": list(result.history),
        "best_epoch": int(result.best_epoch),
        "best_val_macro_f1_headline": float(best["val_macro_f1_headline"]),
        "published_fold": {"best_epoch": published_fold["best_epoch"],
                           "macro_f1_headline": published_fold["macro_f1_headline"],
                           "device": "cuda (Colab)"},
        "seconds": seconds,
        "provenance": provenance.provenance(),
    }
    provenance.write_json(args.out, payload)
    print(f"best epoch {result.best_epoch}, validation macro-F1 (N/S/V/F) "
          f"{best['val_macro_f1_headline']:.4f}; published fold: epoch "
          f"{published_fold['best_epoch']}, {published_fold['macro_f1_headline']:.4f}")
    print(f"{seconds}s on {payload['device']}. Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
