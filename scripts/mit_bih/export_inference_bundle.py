#!/usr/bin/env python3
"""Write the MIT-BIH inference bundle: preprocessing settings + checkpoint list.

    python scripts/mit_bih/export_inference_bundle.py
    python scripts/mit_bih/export_inference_bundle.py --ensemble A_focal_only --out /tmp/b.json

The checkpoints in models/ hold weights only. The bundle adds what inference
also needs -- band-pass settings, beat window, the z-score statistics fitted on
the training records, the RR-feature definition -- and the SHA-256 of every
member checkpoint. It contains no patient data: two aggregate statistics and a
list of training record numbers.

The default ensemble is D_all_four, the Phase 1g selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mit_bih import config, inference  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path,
                        default=config.data_root() / config.PROCESSED_DIRNAME / "mitdb-ds12",
                        help="the DS1/DS2 preprocessing output the checkpoints were trained on")
    parser.add_argument("--ensemble", choices=sorted(inference.ENSEMBLES),
                        default=inference.SELECTED_ENSEMBLE)
    parser.add_argument("--out", type=Path, default=inference.DEFAULT_BUNDLE_PATH)
    args = parser.parse_args(argv)

    norm = args.processed_dir.expanduser() / "normalization.json"
    if not norm.exists():
        print(f"no {norm}. Run: python scripts/mit_bih/preprocess_mitbih.py --strategy ds1_ds2 "
              f"--out-dir ~/physionet-data/processed/mitdb-ds12", file=sys.stderr)
        return 1

    bundle = inference.build_bundle(args.processed_dir, args.ensemble)
    path = inference.save_bundle(args.out, bundle)
    print(f"Ensemble      : {bundle.ensemble} ({len(bundle.members)} checkpoints)")
    print(f"Normalization : mean {bundle.norm_mean:.6f}, std {bundle.norm_std:.6f} "
          f"(fitted on {len(bundle.normalization_source['train_records'])} training records)")
    print(f"Calibration   : {bundle.calibration}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
