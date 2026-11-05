#!/usr/bin/env python3
"""Check what this machine has: environment, datasets, models, blockers.

    python scripts/validate_setup.py
    python scripts/validate_setup.py --json status.json

Every check is reported as one of:

    OK       present and valid
    MISSING  not produced yet; the detail names the command that produces it
    BLOCKED  needs something outside this repository (credentialed access, data
             that does not exist, hardware)
    FAIL     present but wrong (bad hash, bad schema, wrong version) -- fix first

Exit status is 1 if any check FAILs, else 0. Reads only; never downloads,
writes data or trains.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import paths  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, status: str, detail: str) -> None:
    RESULTS.append({"check": name, "status": status, "detail": detail})


def check_environment() -> None:
    v = sys.version_info
    record("python", "OK" if (v.major, v.minor) == (3, 13) else "FAIL",
           f"{platform.python_version()} (project pins 3.13)")
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if m:
            pins[m.group(1)] = m.group(2)
    import_names = {"scikit-learn": "sklearn"}
    for pkg, want in pins.items():
        try:
            mod = importlib.import_module(import_names.get(pkg, pkg))
        except ImportError:
            record(f"package {pkg}", "FAIL", f"not installed; run ./venv/bin/pip install -r requirements.txt")
            continue
        have = getattr(mod, "__version__", "?").split("+")[0]
        record(f"package {pkg}", "OK" if have == want else "FAIL", f"installed {have}, pinned {want}")


def check_compute() -> None:
    try:
        import torch
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + \
                  (["mps"] if torch.backends.mps.is_available() else [])
    except ImportError:
        devices = ["torch missing"]
    record("compute", "OK", f"{os.cpu_count()} CPUs; torch devices {devices}. "
           f"Neural training ran on Colab GPUs; CPU training takes hours per phase.")


def check_mitbih() -> None:
    from mit_bih import acquire, config

    raw = config.mitdb_dir()
    if not raw.is_dir():
        record("MIT-BIH raw records", "MISSING", f"{raw} absent; run python scripts/mit_bih/download_mitbih.py")
    else:
        bad = acquire.missing_records(raw)
        record("MIT-BIH raw records", "OK" if not bad else "FAIL",
               f"48/48 verify at {raw}" if not bad else f"{len(bad)} records fail: {bad[:5]}")

    proc = config.data_root() / config.PROCESSED_DIRNAME / "mitdb-ds12"
    split = proc / "splits.json"
    if not split.exists():
        record("MIT-BIH DS1/DS2 beats", "MISSING",
               "run python scripts/mit_bih/preprocess_mitbih.py --strategy ds1_ds2 "
               "--out-dir ~/physionet-data/processed/mitdb-ds12")
    else:
        strategy = json.loads(split.read_text()).get("strategy")
        n = len(list((proc / "beats").glob("*.npz")))
        record("MIT-BIH DS1/DS2 beats", "OK" if strategy == "ds1_ds2" else "FAIL",
               f"{n} records, strategy {strategy}, at {proc}")

    try:
        from mit_bih import inference
        b = inference.load_bundle(verify=True)
        record("MIT-BIH inference bundle", "OK",
               f"{b.ensemble}, {len(b.members)} checkpoints, hashes verified; scores uncalibrated")
    except FileNotFoundError as exc:
        record("MIT-BIH inference bundle", "MISSING", str(exc))
    except (ValueError, ImportError) as exc:
        record("MIT-BIH inference bundle", "FAIL", str(exc))
    record("MIT-BIH calibration (Phase 1f Stage B)", "MISSING",
           "not started; scores are rankings, not probabilities")


def check_mimic() -> None:
    from mimic_iii import config, ingestion, train
    from mimic_iii.errors import MimicError

    try:
        config.load_task_spec()
        train.ModelSpec.from_dict(config.load_model_spec())
        record("MIMIC-III config", "OK", f"{config.DEFAULT_CONFIG_PATH.relative_to(ROOT)}")
    except (ValueError, OSError) as exc:
        record("MIMIC-III config", "FAIL", str(exc))

    for source in ("demo", "full"):
        root = config.source_dir(source)
        name = f"MIMIC-III {source} tables"
        if not root.is_dir():
            if source == "full":
                record(name, "BLOCKED", f"{root} absent. Requires PhysioNet credentialing "
                       f"(account, CITI training, signed DUA) - only the data owner can do this.")
            else:
                record(name, "MISSING", f"{root} absent; see COMMANDS.md section 8")
            continue
        try:
            ingestion.validate_source(root)
            ingestion.check_itemids(root)
            record(name, "OK", f"all {len(ingestion.REQUIRED_TABLES)} tables and every ITEMID present at {root}")
        except MimicError as exc:
            record(name, "FAIL", str(exc))

        out = config.output_dir(source)
        missing = [f for f in ("features.csv.gz", "splits.json", "model_meta.json", "report.json")
                   if not (out / f).exists()]
        record(f"MIMIC-III {source} artifacts", "OK" if not missing else "MISSING",
               f"at {out}" if not missing else
               f"missing {missing}; run scripts/mimic_iii/build_features.py, make_splits.py, "
               f"train.py --source {source}")


def check_multimodal() -> None:
    record("paired ECG + clinical dataset", "BLOCKED",
           "MIT-BIH and MIMIC-III contain different patients. A paired source (MIMIC-III "
           "Waveform Matched Subset) needs credentialed MIMIC-III and terabyte-scale storage.")
    record("fusion model training", "BLOCKED", "depends on the paired dataset above")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None, help="also write the results as JSON")
    args = parser.parse_args(argv)

    print(f"Data root: {paths.data_root()} (${paths.DATA_ROOT_ENV_VAR})\n")
    for check in (check_environment, check_compute, check_mitbih, check_mimic, check_multimodal):
        try:
            check()
        except Exception as exc:  # a crashing check is itself a failure to report
            record(check.__name__, "FAIL", f"{type(exc).__name__}: {exc}")

    width = max(len(r["check"]) for r in RESULTS)
    for r in RESULTS:
        print(f"{r['status']:8s} {r['check']:<{width}}  {r['detail']}")
    counts = {s: sum(r["status"] == s for r in RESULTS) for s in ("OK", "MISSING", "BLOCKED", "FAIL")}
    print(f"\n{counts}")
    if args.json:
        args.json.write_text(json.dumps(RESULTS, indent=2) + "\n")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
