#!/usr/bin/env python3
"""Run the MIT-BIH preprocessing pipeline (PLAN.md Phase 1).

    python scripts/preprocess_mitbih.py
    python scripts/preprocess_mitbih.py --strategy ds1_ds2

Writes into ``$SMART_HEALTH_DATA/processed/mitbih/``:

    beats/<record>.npz    X (n, 300) float32, y (n,) int8, r_peaks (n,) int64
    manifest.csv          per-record accounting, including every dropped beat
    splits.json           record-wise train/val/test assignment
    normalization.json    z-score statistics fitted on training records only

Exit status is non-zero if any record failed, even when others succeeded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg import config, pipeline  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help=f"raw WFDB records (default: ${config.DATA_ROOT_ENV_VAR}/{config.MITDB_DIRNAME})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: <data root>/processed/mitbih)")
    parser.add_argument("--records", nargs="*", default=None,
                        help="subset of record ids (default: everything on disk)")
    parser.add_argument("--strategy", choices=("record_wise", "ds1_ds2"),
                        default="record_wise",
                        help="split strategy. Beat-level splitting is not available.")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SPLIT_SEED)
    parser.add_argument("--include-paced", action="store_true",
                        help="keep the four paced records (102, 104, 107, 217)")
    parser.add_argument("--notch", action="store_true",
                        help="enable the powerline notch (a no-op under the default 40 Hz low-pass)")
    parser.add_argument("--strict-symbols", action="store_true",
                        help="fail on any annotation symbol missing from the AAMI table")
    args = parser.parse_args(argv)

    data_dir = (args.data_dir or config.mitdb_dir()).expanduser()
    out_dir = (args.out_dir or config.processed_dir()).expanduser()

    print(f"Reading  : {data_dir}")
    print(f"Writing  : {out_dir}")
    print(f"Strategy : {args.strategy} (record-wise; beat-level splitting is unavailable)")

    result = pipeline.run(
        data_dir,
        out_dir,
        records=args.records,
        strategy=args.strategy,
        seed=args.seed,
        exclude_paced=not args.include_paced,
        use_notch=args.notch,
        strict_symbols=args.strict_symbols,
    )

    split = result.split
    print(
        f"\nRecords  : {len(result.ok)} ok, {len(result.failed)} failed "
        f"({len(split.excluded)} excluded: {', '.join(split.excluded) or 'none'})"
    )
    print(f"Splits   : train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    print(
        f"Beats    : {result.total_beats():,} kept overall "
        f"({result.total_beats('train'):,} in train)"
    )

    all_totals = result.class_totals()
    train_totals = result.class_totals("train")
    train_beats = max(result.total_beats("train"), 1)
    print(f"\n  {'class':<6}{'all':>10}{'train':>10}{'train %':>10}")
    for cls in config.AAMI_CLASSES:
        overall = all_totals.get(cls, 0)
        in_train = train_totals.get(cls, 0)
        share = 100 * in_train / train_beats
        note = "   <- absent from train" if in_train == 0 else ""
        print(f"  {cls:<6}{overall:>10,}{in_train:>10,}{share:>9.2f}%{note}")
    print(
        "\n  Class weights in normalization.json are fitted on the train split "
        "only,\n  and are reported for use at training time -- nothing here is "
        "resampled."
    )

    for record in result.failed:
        print(f"\nFAILED {record.record}: {record.error}", file=sys.stderr)

    if result.failed:
        print(
            f"\nExit 1: {len(result.failed)} record(s) failed. Successful records "
            f"were still written.",
            file=sys.stderr,
        )

    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
