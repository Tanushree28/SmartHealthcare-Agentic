#!/usr/bin/env python3
"""Train the fusion model -- only from validated, genuinely paired data.

    python scripts/multimodal/train_fusion.py                       # reports BLOCKED
    python scripts/multimodal/train_fusion.py --pairs pairs.csv     # validates a pairing manifest

Without a paired dataset this script exits with status 3 and says why. That is
the current state: MIT-BIH and MIMIC-III contain different patients, and the
genuinely paired source (MIMIC-III Waveform Database Matched Subset, linked to
credentialed MIMIC-III by subject_id) is not available here.

--pairs checks a manifest against multimodal.fusion.validate_pairs: same
subject on both sides, ECG recorded inside the observation window before the
index time, one split per subject, and no MIT-BIH-with-MIMIC-III rows. Loading
ECGs and clinical features for validated pairs depends on which paired dataset
is obtained, so training stays BLOCKED until then; the training loop itself
(multimodal.fusion.train_fusion) is implemented and tested on synthetic pairs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from multimodal import fusion  # noqa: E402
from multimodal.errors import PairingError  # noqa: E402
from multimodal.inference import FUSION_BLOCKED_REASON  # noqa: E402

EXIT_BLOCKED = 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, default=None,
                        help=f"CSV with columns {', '.join(fusion.PAIR_COLUMNS)}")
    parser.add_argument("--observation-hours", type=float, default=24.0)
    args = parser.parse_args(argv)

    if args.pairs is None:
        print(f"BLOCKED: {FUSION_BLOCKED_REASON}")
        return EXIT_BLOCKED

    pairs = pd.read_csv(args.pairs)
    try:
        fusion.validate_pairs(pairs, observation_hours=args.observation_hours)
    except PairingError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1
    counts = pairs.groupby("split")["subject_id"].nunique().to_dict()
    print(f"Pairs valid: {len(pairs)} rows, subjects per split {counts}")
    print("BLOCKED: loading the paired ECG and clinical features depends on the paired "
          "dataset that is obtained; see PlanDoc/specs/2026-10-14-mimic-iii-and-multimodal-design.md.")
    return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
