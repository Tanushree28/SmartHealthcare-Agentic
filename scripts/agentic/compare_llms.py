#!/usr/bin/env python3
"""Compare local LLMs as report writers on the same simulated cases.

    python scripts/agentic/compare_llms.py                    # 20 cases, llama3.1:8b vs qwen2.5
    python scripts/agentic/compare_llms.py --cases 4          # quick check

About 20-40 minutes for 20 cases x 2 models on an Apple M4. Writes:
  PlanDoc/agentic_results/llm_bakeoff.md     aggregate scores only (no patient data)
  <data root>/processed/agentic/bakeoff/     every draft, for inspection

Cases pair held-out MIT-BIH windows (60-80 s) with MIMIC-III Demo ICU stays
chosen by a fixed seed. They are different people: simulated cases.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic import bakeoff, orchestrator  # noqa: E402
from agentic.llm import COMPARISON_MODEL, DEFAULT_MODEL, OllamaClient  # noqa: E402
from common import paths  # noqa: E402

#: Held-out (DS2) records; 202 is left out because it is the same person as 201.
RECORDS = ("100", "103", "105", "111", "113", "117", "121", "123", "200", "210")
START_S, SECONDS = 60.0, 20.0
SEED = 20261015


def build_cases(deps, n: int) -> list[tuple[str, object]]:
    stays = sorted(int(s) for s in deps.chart.cohort["icustay_id"])
    chosen = np.random.default_rng(SEED).choice(stays, size=min(n, len(stays)), replace=False)
    cases = []
    for i, stay in enumerate(chosen):
        req = orchestrator.CaseRequest(record=RECORDS[(i // 2) % len(RECORDS)], start_s=START_S,
                                       seconds=SECONDS, stay=int(stay))
        case = list(orchestrator.run_case(req, deps))[-1].payload["case"]
        cases.append((f"case {i + 1}", case))
    return cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL, COMPARISON_MODEL])
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--out", type=Path,
                        default=paths.PROJECT_ROOT / "PlanDoc" / "agentic_results" / "llm_bakeoff.md")
    args = parser.parse_args(argv)

    clients = []
    for name in args.models:
        client = OllamaClient(name)
        status = client.status()
        print(f"{name}: {status.message}")
        if status.ok:
            clients.append(client)
    if not clients:
        print("No model is ready; nothing to compare.", file=sys.stderr)
        return 1

    deps = orchestrator.default_deps(need_ecg=True, need_chart=True)
    if deps.unavailable:
        print(f"Cannot build cases: {deps.unavailable}", file=sys.stderr)
        return 1
    cases = build_cases(deps, args.cases)
    print(f"{len(cases)} cases built; writing reports...")

    trials = bakeoff.run_trials(cases, clients)
    rows = bakeoff.summarize(trials)
    winner = bakeoff.pick_winner(rows)
    today = dt.date.today().isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(bakeoff.render_markdown(rows, winner, len(cases), SEED, today))
    raw_dir = paths.data_root() / "processed" / "agentic" / "bakeoff"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"{today}.json"
    raw.write_text(json.dumps({"rows": rows, "trials": [asdict(t) for t in trials],
                               "cases": [{"label": label, "record": c.record, "stay": c.stay}
                                         for label, c in cases]}, indent=2) + "\n")
    print(f"Winner: {winner}\nWrote {args.out}\nWrote {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
