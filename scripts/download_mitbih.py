#!/usr/bin/env python3
"""Download and verify the MIT-BIH Arrhythmia Database (PLAN.md step 0.3).

Idempotent: records already present and passing their integrity check are
skipped, so re-running costs nothing and cannot corrupt existing files.

    python scripts/download_mitbih.py            # download into $SMART_HEALTH_DATA
    python scripts/download_mitbih.py --verify   # check what is on disk, no network

MIT-BIH is openly available on PhysioNet -- no account, no data use agreement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg import acquire, config  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"target directory (default: ${config.DATA_ROOT_ENV_VAR}/{config.MITDB_DIRNAME})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only verify what is already on disk; makes no network request",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download every record even if it already verifies",
    )
    args = parser.parse_args(argv)

    directory = (args.dir or config.mitdb_dir()).expanduser()
    print(f"MIT-BIH directory: {directory}")

    if args.verify:
        checks = acquire.verify(directory)
        bad = [c for c in checks if not c.ok]
        for check in bad:
            print(f"  FAIL {check}")
        print(f"{len(checks) - len(bad)}/{len(checks)} records verify.")
        return 1 if bad else 0

    outstanding = acquire.missing_records(directory)
    if not outstanding:
        print(f"All {len(config.ALL_RECORDS)} records already present and verified.")
        return 0

    print(f"{len(outstanding)} record(s) to fetch: {', '.join(outstanding)}")
    acquire.download(directory)

    size_mb = acquire.directory_size_bytes(directory) / 1e6
    print(f"Done. {len(config.ALL_RECORDS)} records verified, {size_mb:.1f} MB on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
