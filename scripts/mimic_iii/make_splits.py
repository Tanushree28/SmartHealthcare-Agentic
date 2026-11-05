#!/usr/bin/env python3
"""Split the MIMIC-III feature table by patient into train/validation/test.

    python scripts/mimic_iii/make_splits.py                 # the Demo
    python scripts/mimic_iii/make_splits.py --source full

Reads features.csv.gz written by build_features.py and writes splits.json next
to it. The split unit is SUBJECT_ID, stratified on the patient-level label, and
deterministic for the seed in configs/mimic_iii.toml. splits.json records the
SHA-256 of the feature table; train.py refuses to run if the table changes
afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import provenance  # noqa: E402
from mimic_iii import config, dataset, features  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(config.SOURCES), default="demo")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="directory holding features.csv.gz (default: the source's output dir)")
    parser.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    spec = config.load_task_spec(args.config)
    out_dir = (args.out_dir or config.output_dir(args.source)).expanduser()
    features_path = out_dir / "features.csv.gz"
    if not features_path.exists():
        print(f"no feature table at {features_path}. Run: "
              f"python scripts/mimic_iii/build_features.py --source {args.source}", file=sys.stderr)
        return 1

    table = features.read_feature_table(features_path)
    split = dataset.patient_split(
        table, fractions=spec.split_fractions, seed=spec.split_seed,
        feature_table_sha256=provenance.sha256_file(features_path))
    path = dataset.write_split(out_dir / "splits.json", split, table)

    parts = dataset.split_frame(table, split)
    print(f"Split by subject_id, seed {spec.split_seed}, fractions {spec.split_fractions}")
    for name, df in parts.items():
        print(f"  {name:5s} {df['subject_id'].nunique():>6} patients  {len(df):>6} stays  "
              f"{int(df[config.LABEL_COLUMN].sum()):>5} positive")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
