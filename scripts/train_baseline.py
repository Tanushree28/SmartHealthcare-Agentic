#!/usr/bin/env python3
"""Train the classical baseline on the Phase 1 output.

    python scripts/preprocess_mitbih.py --strategy ds1_ds2 --out-dir ~/physionet-data/processed/mitdb-ds12
    python scripts/train_baseline.py --processed-dir ~/physionet-data/processed/mitdb-ds12

This is a floor, not a final model: a number a later deep model has to beat
before it can claim to have learned anything. It also confirms the Phase 1
output is learnable at all.

The test split is evaluated **once**. Any tuning belongs on the validation
records.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg import baseline, config, evaluate, features  # noqa: E402


def load_features(beats_dir, records, feature_set: str):
    kwargs = baseline.FEATURE_SETS[feature_set]
    return features.extract_records(beats_dir, records, **kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=None,
                        help="a preprocessing output directory (default: the standard one)")
    parser.add_argument("--strategy", choices=("record_wise", "ds1_ds2"), default="ds1_ds2",
                        help="the split the processed data must have been built with")
    parser.add_argument("--classifiers", nargs="*", default=list(baseline.CLASSIFIERS))
    parser.add_argument("--feature-sets", nargs="*", default=list(baseline.FEATURE_SETS))
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SPLIT_SEED)
    parser.add_argument("--no-balance", action="store_true",
                        help="disable class weighting (the majority class will dominate)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write metrics.json (default: inside the processed dir)")
    args = parser.parse_args(argv)

    processed = (args.processed_dir or config.processed_dir()).expanduser()
    print(f"Processed data : {processed}")

    # --- Leakage tripwire -------------------------------------------------
    split = baseline.load_split_or_refuse(processed, required_strategy=args.strategy)
    train_records = split["train"]
    test_records = split["test"]
    val_records = split["val"]

    overlap = set(train_records) & set(test_records)
    if overlap:
        raise SystemExit(f"record leakage in splits.json: {sorted(overlap)}")

    print(f"Split          : {split['strategy']} "
          f"(train={len(train_records)} val={len(val_records)} test={len(test_records)})")
    print(f"Excluded       : {', '.join(split['excluded']) or 'none'}")
    print()

    beats_dir = processed / "beats"
    results = []
    summary_rows = []

    for feature_set in args.feature_sets:
        train = load_features(beats_dir, train_records, feature_set)
        test = load_features(beats_dir, test_records, feature_set)

        assert not set(train.record) & set(test.record), "record crossed the split"

        for kind in args.classifiers:
            label = f"{kind} / {feature_set}"
            started = time.perf_counter()

            pipeline = baseline.build_pipeline(
                kind,
                seed=args.seed,
                balanced=not args.no_balance,
                classes=np.unique(train.y),
            )
            pipeline.fit(train.X, train.y)
            predictions = pipeline.predict(test.X)

            elapsed = time.perf_counter() - started
            result = evaluate.evaluate(test.y, predictions, label=label)
            assert result.reconciles()
            results.append(result)

            print(evaluate.format_report(result))

            groups = baseline.grouped_importance(
                baseline.feature_importances(pipeline, train.feature_names)
            )
            if groups:
                shares = "  ".join(f"{k}={v:.1%}" for k, v in sorted(groups.items()))
                print(f"\nfeature importance by family: {shares}")

            print(f"\ntrain {train.n_kept:,} beats -> test {test.n_kept:,} beats "
                  f"in {elapsed:.1f}s "
                  f"({train.n_dropped_edge + test.n_dropped_edge} dropped: "
                  f"first/last beat of each record has no RR neighbour)")
            print("\n" + "-" * 72 + "\n")

            summary_rows.append((label, result.macro_recall, result.recall_of("S"),
                                 result.recall_of("V"), result.accuracy))

    print("SUMMARY (test split, evaluated once)")
    print(f"{'model / features':<34}{'macro rec':>10}{'S rec':>8}{'V rec':>8}{'acc':>8}")
    for label, macro, s, v, acc in summary_rows:
        print(f"{label:<34}{macro:>10.3f}{s:>8.3f}{v:>8.3f}{acc:>8.3f}")
    print("\nRead macro recall and the per-class columns. Accuracy is uninformative "
          "here:\nnormal beats are ~86% of the data.")

    out = args.out or (processed / "baseline_metrics.json")
    evaluate.write_json(results, out, extra={
        "split_strategy": split["strategy"],
        "train_records": train_records,
        "test_records": test_records,
        "balanced": not args.no_balance,
        "seed": args.seed,
    })
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
