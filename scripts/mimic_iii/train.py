#!/usr/bin/env python3
"""Train and evaluate the MIMIC-III clinical model.

    python scripts/mimic_iii/train.py                 # the Demo: pipeline check only
    python scripts/mimic_iii/train.py --source full

Fits imputation + scaling + one-hot encoding + logistic regression on the
training patients only, chooses C and the decision threshold on the validation
patients, then scores the test patients once. Writes, next to the features:

    model.pkl, model_meta.json   the fitted pipeline and its description
    report.json, report.md       metrics with bootstrap CIs, cohort, task

Refuses to run if features.csv.gz changed after splits.json was made.
On the Demo, the report is labelled as pipeline verification, not performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import provenance  # noqa: E402
from mimic_iii import config, dataset, evaluate, features, train  # noqa: E402
from mimic_iii.errors import LeakageError  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(config.SOURCES), default="demo")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    spec = config.load_task_spec(args.config)
    model_spec = train.ModelSpec.from_dict(config.load_model_spec(args.config))
    out_dir = (args.out_dir or config.output_dir(args.source)).expanduser()
    features_path = out_dir / "features.csv.gz"
    split_path = out_dir / "splits.json"
    summary_path = out_dir / "build_summary.json"
    for p, cmd in ((features_path, "build_features.py"), (split_path, "make_splits.py")):
        if not p.exists():
            print(f"missing {p}. Run: python scripts/mimic_iii/{cmd} --source {args.source}",
                  file=sys.stderr)
            return 1

    # --- tripwire: the split must belong to this exact feature table ---------
    split = dataset.read_split(split_path)
    actual = provenance.sha256_file(features_path)
    if split.feature_table_sha256 != actual:
        raise LeakageError(
            f"{split_path.name} was made for a different feature table (hash "
            f"{split.feature_table_sha256} != {actual}). Re-run: "
            f"python scripts/mimic_iii/make_splits.py --source {args.source}")

    table = features.read_feature_table(features_path)
    parts = dataset.split_frame(table, split)

    model, selection = train.select_and_fit(parts["train"], parts["val"], model_spec, seed=args.seed)
    threshold = selection["threshold"]
    results = {name: evaluate.evaluate_split(model, parts[name], threshold,
                                             n_boot=model_spec.n_bootstrap, seed=args.seed)
               for name in dataset.SPLIT_NAMES}

    cohort_summary = (json.loads(summary_path.read_text())["cohort"]
                      if summary_path.exists() else {"flow": [], "n_stays": len(table),
                                                     "n_positive": int(table["label"].sum())})
    split_counts = {n: {"patients": int(df["subject_id"].nunique()), "stays": int(len(df)),
                        "positive": int(df["label"].sum())} for n, df in parts.items()}
    report = evaluate.build_report(
        args.source, spec, cohort_summary, split_counts, selection, results,
        evaluate.top_coefficients(model),
        inputs={"features": features_path, "splits": split_path})

    meta = {
        "task": spec.describe(),
        "data_source": args.source,
        "is_demo": args.source == "demo",
        "banner": report["banner"],
        "selection": selection,
        "threshold": threshold,
        "train_subjects": len(split.train),
        "feature_table_sha256": actual,
        "provenance": report["provenance"],
    }
    saved = train.save_model(out_dir, model, meta)
    written = evaluate.write_report(out_dir, report)

    print(f"\n{report['banner']}\n")
    print(f"Selected C = {selection['selected_C']}  ({selection['selection_note']})")
    print(f"Threshold  = {threshold:.3f}  ({selection['threshold_note']})\n")
    print(f"{'split':6s} {'n':>5s} {'pos':>4s} {'AUROC':>7s} {'AUPRC':>7s} {'Brier':>7s}")
    for name, m in results.items():
        f = lambda v: "   n/a" if v is None else f"{v:7.3f}"  # noqa: E731
        print(f"{name:6s} {m['n']:>5d} {m['n_positive']:>4d} {f(m['auroc'])} {f(m['auprc'])} {f(m['brier'])}")
    t = results["test"]
    if t["auroc_ci"]["lower"] is not None:
        print(f"\nTest AUROC 95% CI: {t['auroc_ci']['lower']:.2f}-{t['auroc_ci']['upper']:.2f} "
              f"(n = {t['n']})")
    print(f"\nWrote {saved['model']}\n      {saved['meta']}\n      {written['json']}\n      {written['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
