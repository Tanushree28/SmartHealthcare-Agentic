#!/usr/bin/env python3
"""Run the whole workflow in order and write a status report.

    python scripts/run_end_to_end.py                      # MIMIC Demo, quick ECG check (~1 min)
    python scripts/run_end_to_end.py --full-ecg-eval      # + all of DS2 for the ECG model (~8 min)
    python scripts/run_end_to_end.py --source full        # credentialed MIMIC-III, once available

Steps, each run as its own script so the command-line entry points are what
gets exercised:

    1. validate datasets, environment and configuration     scripts/validate_setup.py
    2. prepare MIMIC-III features                            scripts/mimic_iii/build_features.py
    3. patient-level MIMIC-III split                         scripts/mimic_iii/make_splits.py
    4. train + evaluate the clinical model                   scripts/mimic_iii/train.py
    5. prepare MIT-BIH beats, record-level DS1/DS2 split     scripts/mit_bih/preprocess_mitbih.py
    6. evaluate the ECG model (trained on Colab; not retrained) scripts/mit_bih/evaluate_bundle.py
    7. smoke-test the shared inference interface             scripts/multimodal/infer.py
    8. fusion training -- expected to report BLOCKED          scripts/multimodal/train_fusion.py

Step 5 writes to a separate folder under the work directory, so the processed
data behind the published MIT-BIH numbers is never overwritten. Nothing here
trains a neural network: the ECG checkpoints were trained on Colab GPUs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import paths, provenance  # noqa: E402
from mimic_iii import config as mimic_config  # noqa: E402

QUICK_ECG_RECORDS = ("100", "208", "232")


def run(name: str, args: list[str], expect: int = 0, blocked_code: int | None = None) -> dict:
    cmd = [sys.executable, *args]
    shown = " ".join(["python", *args])
    print(f"\n=== {name}\n$ {shown}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    seconds = round(time.time() - t0, 1)
    out = (proc.stdout + proc.stderr).strip()
    print(out[-3000:])
    if proc.returncode == expect:
        status = "OK"
    elif blocked_code is not None and proc.returncode == blocked_code:
        status = "BLOCKED"
    else:
        status = "FAIL"
    print(f"--- {status} (exit {proc.returncode}, {seconds}s)", flush=True)
    return {"step": name, "command": shown, "exit": proc.returncode, "status": status,
            "seconds": seconds, "output_tail": out[-1500:]}


def first_test_stay(out_dir: Path) -> int | None:
    import pandas as pd

    try:
        split = json.loads((out_dir / "splits.json").read_text())
        table = pd.read_csv(out_dir / "features.csv.gz", usecols=["subject_id", "icustay_id"])
    except (OSError, ValueError):
        return None
    rows = table[table["subject_id"].isin(split["test"])]
    return int(rows["icustay_id"].iloc[0]) if len(rows) else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(mimic_config.SOURCES), default="demo")
    parser.add_argument("--work-dir", type=Path, default=paths.processed_root() / "end_to_end")
    parser.add_argument("--full-ecg-eval", action="store_true",
                        help="score all 22 DS2 records, not three (about 8 minutes on CPU)")
    args = parser.parse_args(argv)

    work = args.work_dir.expanduser()
    work.mkdir(parents=True, exist_ok=True)
    mimic_out = mimic_config.output_dir(args.source)
    ecg_dir = work / "mitdb-ds12"
    steps = []

    steps.append(run("1. validate setup", ["scripts/validate_setup.py", "--json", str(work / "validate.json")]))
    steps.append(run("2. MIMIC-III features", ["scripts/mimic_iii/build_features.py", "--source", args.source],
                     blocked_code=3))
    mimic_ok = steps[-1]["status"] == "OK"
    if mimic_ok:
        steps.append(run("3. MIMIC-III patient split", ["scripts/mimic_iii/make_splits.py", "--source", args.source]))
        steps.append(run("4. MIMIC-III train + evaluate", ["scripts/mimic_iii/train.py", "--source", args.source]))
    else:
        for n in ("3. MIMIC-III patient split", "4. MIMIC-III train + evaluate"):
            steps.append({"step": n, "command": "-", "exit": None, "status": steps[-1]["status"],
                          "seconds": 0, "output_tail": "skipped: no feature table"})

    steps.append(run("5. MIT-BIH beats + DS1/DS2 split", [
        "scripts/mit_bih/preprocess_mitbih.py", "--strategy", "ds1_ds2", "--out-dir", str(ecg_dir)]))
    ecg_eval = ["scripts/mit_bih/evaluate_bundle.py", "--processed-dir", str(ecg_dir)]
    if not args.full_ecg_eval:
        ecg_eval += ["--records", *QUICK_ECG_RECORDS]
    steps.append(run("6. MIT-BIH ECG model evaluation", ecg_eval))

    infer = ["scripts/multimodal/infer.py", "--ecg-record", "100", "--out", str(work / "inference.json")]
    stay = first_test_stay(mimic_out) if mimic_ok else None
    if stay is not None:
        infer += ["--clinical-stay", str(stay), "--source", args.source]
    steps.append(run("7. shared inference smoke test", infer))
    steps.append(run("8. fusion training", ["scripts/multimodal/train_fusion.py"], expect=-1, blocked_code=3))

    report = render(steps, args, work, mimic_out)
    (work / "report.md").write_text(report)
    provenance.write_json(work / "steps.json", {"steps": steps, "provenance": provenance.provenance()})
    print(f"\nReport: {work / 'report.md'}")
    print("\n".join(f"  {s['status']:8s} {s['step']}" for s in steps))
    return 1 if any(s["status"] == "FAIL" for s in steps) else 0


def render(steps: list[dict], args, work: Path, mimic_out: Path) -> str:
    def rel(p: Path) -> str:
        return str(p).replace(str(paths.data_root()), "$SMART_HEALTH_DATA")

    lines = ["# End-to-end run", "",
             f"Generated {provenance.utc_now()} - git {provenance.git_state()['commit']}", "",
             "| Step | Status | Exit | Seconds |", "|---|---|---:|---:|"]
    for s in steps:
        lines.append(f"| {s['step']} | **{s['status']}** | {s['exit']} | {s['seconds']} |")
    lines += ["", "## Artifacts", "",
              f"- MIMIC-III ({args.source}): `{rel(mimic_out)}/` - features, splits, model, report.md",
              f"- MIT-BIH beats for this run: `{rel(work / 'mitdb-ds12')}/`",
              f"- Inference output: `{rel(work / 'inference.json')}`",
              f"- Setup check: `{rel(work / 'validate.json')}`", "",
              "## What this run does and does not show", "",
              "- The MIMIC-III numbers on the Demo verify the pipeline. They are not clinical performance.",
              "- The ECG model was trained on Colab GPUs (Phases 1d-1g) and is evaluated here, not retrained.",
              "- ECG scores are uncalibrated (Phase 1f Stage B not started).",
              "- Software integration is complete; a multimodal model trained and evaluated on real",
              "  paired data does not exist. Fusion training is BLOCKED for lack of paired data.", "",
              "## Output of each step (tail)", ""]
    for s in steps:
        tail = s["output_tail"].replace(str(paths.data_root()), "$SMART_HEALTH_DATA")
        lines += [f"### {s['step']}", "", "```", tail, "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
