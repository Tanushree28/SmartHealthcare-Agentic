#!/usr/bin/env python3
"""Show the clinical model's prediction for each patient next to what happened.

    python scripts/mimic_iii/show_predictions.py                 # the test patients
    python scripts/mimic_iii/show_predictions.py --split all     # every patient

For each ICU stay: the risk score, the model's call (died / survived, using
the threshold chosen on the validation patients), what actually happened, and
whether the call was right. The three features that pushed the score most are
listed, so you can see why the model said what it said.

Run build_features.py, make_splits.py and train.py first. On the MIMIC-III
Demo the test split is 7 patients, so the right/wrong count is a demonstration
of the pipeline, not a measure of how good the model is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mimic_iii import config, dataset, features, preprocessing, train  # noqa: E402


def reasons(model, X, k: int = 3) -> list[str]:
    """The k features with the largest push on each row's score, with direction."""
    names = preprocessing.output_feature_names(model)
    values = model.named_steps["pre"].transform(X)
    coef = model.named_steps["clf"].coef_.ravel()
    out = []
    for row in values * coef:
        top = np.argsort(-np.abs(row))[:k]
        out.append(", ".join(f"{'+' if row[i] > 0 else '-'}{names[i]}" for i in top))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(config.SOURCES), default="demo")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    args = parser.parse_args(argv)

    out_dir = (args.out_dir or config.output_dir(args.source)).expanduser()
    try:
        model, meta = train.load_model(out_dir)
    except FileNotFoundError as exc:
        print(f"{exc}\nRun: python scripts/mimic_iii/build_features.py, make_splits.py, "
              f"train.py --source {args.source}", file=sys.stderr)
        return 1
    table = features.read_feature_table(out_dir / "features.csv.gz")
    parts = dataset.split_frame(table, dataset.read_split(out_dir / "splits.json"))
    names = dataset.SPLIT_NAMES if args.split == "all" else (args.split,)
    threshold = float(meta["threshold"])

    print(f"\n{meta.get('banner', '')}\n")
    print(f"Question: will this ICU heart patient die before leaving hospital?")
    print(f"Data: first 24 h of the ICU stay. Rule: risk score >= {threshold:.2f} means "
          f"'predicts died'.\n")
    print(f"{'split':5s}  {'ICU stay':>8s}  {'age':>4s} {'sex':3s}  {'risk':>5s}  "
          f"{'model says':10s}  {'really':8s}  {'right?':6s}  biggest reasons (+ raises risk, - lowers)")

    total = right = caught = deaths = false_alarms = 0
    for name in names:
        df = parts[name]
        X, y = dataset.xy(df)
        risk = model.predict_proba(X)[:, 1]
        why = reasons(model, X)
        for i in range(len(df)):
            said = int(risk[i] >= threshold)
            ok = said == y[i]
            age = ">89" if df["age_over_89"].iloc[i] == 1 else f"{df['age'].iloc[i]:.0f}"
            print(f"{name:5s}  {int(df['icustay_id'].iloc[i]):>8d}  {age:>4s} {str(df['gender'].iloc[i]):3s}  "
                  f"{risk[i]:5.2f}  {'died' if said else 'survived':10s}  "
                  f"{'died' if y[i] else 'survived':8s}  {'yes' if ok else 'NO':6s}  {why[i]}")
            total += 1
            right += int(ok)
            deaths += int(y[i])
            caught += int(said and y[i])
            false_alarms += int(said and not y[i])

    print(f"\nRight: {right} of {total}.  Deaths caught: {caught} of {deaths}.  "
          f"False alarms (said died, survived): {false_alarms}.")
    if args.split in ("train", "all"):
        print("Note: the train rows are patients the model learned from, so getting them right "
              "proves little.")
    if meta.get("is_demo"):
        print("Note: MIMIC-III Demo. Too few patients for these counts to mean anything "
              "about real performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
