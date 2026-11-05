#!/usr/bin/env python3
"""Download and check the St Petersburg INCART 12-lead Arrhythmia Database.

    python scripts/incart/download_incart.py            # download into $SMART_HEALTH_DATA/incartdb
    python scripts/incart/download_incart.py --verify   # check what is on disk, no network

Source: PhysioNet ``incartdb`` v1.0.0. Open access under the Open Data Commons
Attribution License v1.0; no account and no data use agreement. 75 records,
about 800 MB.

Idempotent: records already present and passing their check are skipped. Each
record must open, be sampled at 257 Hz, carry lead II in mV, and have its
``.atr`` beat annotations. The patient documentation file is fetched too, if
PhysioNet has it; without it, intervals are computed by record.

Prints counts only, never record contents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from incart import config, io  # noqa: E402
from mit_bih import acquire  # noqa: E402


def _patient_line(directory: Path, fetch: bool) -> str:
    present = io.fetch_patient_file(directory) if fetch else (directory / config.PATIENT_FILE).is_file()
    if present:
        return "Patient file: present"
    return "Patient file: not available (results will be grouped by record)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=None,
                        help=f"target directory (default: $SMART_HEALTH_DATA/{config.DIRNAME})")
    parser.add_argument("--verify", action="store_true",
                        help="only check what is already on disk; makes no network request")
    parser.add_argument("--force", action="store_true",
                        help="re-download every record even if it already passes its check")
    args = parser.parse_args(argv)

    directory = (args.dir or config.raw_dir()).expanduser()
    print(f"INCART directory: {directory}")

    if args.verify:
        checks = io.verify(directory)
        bad = [c for c in checks if not c.ok]
        for check in bad:
            print(f"  FAIL {check}")
        print(f"{len(checks) - len(bad)}/{len(checks)} records verify.")
        print(_patient_line(directory, fetch=False))
        return 1 if bad else 0

    outstanding = config.RECORDS if args.force else io.missing_records(directory)
    if outstanding:
        print(f"{len(outstanding)} record(s) to fetch.")
        io.download(directory, force=args.force)
    else:
        print(f"All {len(config.RECORDS)} records already present and verified.")

    size_mb = acquire.directory_size_bytes(directory) / 1e6
    print(f"{len(config.RECORDS)} records verified, {size_mb:.1f} MB on disk.")
    print(_patient_line(directory, fetch=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
