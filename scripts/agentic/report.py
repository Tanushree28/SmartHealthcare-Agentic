#!/usr/bin/env python3
"""Run the agents on one case and print every stage and the report.

    python scripts/agentic/report.py --list
    python scripts/agentic/report.py --ecg-record 208 --seconds 20 --stay 200282
    python scripts/agentic/report.py --ecg-record 100                  # ECG only
    python scripts/agentic/report.py --stay 200282                     # chart only

The ECG comes from an MIT-BIH record (held-out records give the honest view);
the chart from one MIMIC-III ICU stay. With both, they are different people
and the report says SIMULATED CASE. A research demo, not medical advice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic import chat_agent, ecg_window, glossary, orchestrator, report_agent  # noqa: E402
from agentic.llm import DEFAULT_MODEL, OllamaClient  # noqa: E402
from common import paths  # noqa: E402
from mit_bih.errors import EcgError  # noqa: E402
from mimic_iii import config as mimic_config  # noqa: E402

CIRCLED = "①②③④⑤⑥⑦⑧⑨"


def describe(event: orchestrator.StageEvent) -> str:
    n, p = event.stage, event.payload
    head = f"{CIRCLED[n - 1]} {event.name.replace('_', ' ')}"
    if event.skipped:
        return f"{head}: skipped ({p['skipped']})"
    if n == 1:
        r = p["record"]
        note = f"; {r.note}" if r.note else ""
        return (f"{head}: record {r.name} ({r.tag}{note}), lead {p['lead']}, "
                f"{p['signal'].size / p['fs']:.1f} s from {p['start_s']:g} s")
    if n == 2:
        return f"{head}: band-pass filtered ({p['signal'].size} samples)"
    if n == 3:
        return f"{head}: {len(p['r_peaks'])} beat positions ({p['source']})"
    if n == 4:
        return f"{head}: {p['n_classified']} beats cut out, {p['n_dropped']} dropped at the edges"
    if n == 5:
        d = p["display_only"]
        return (f"{head}: model and cardiologist agree on {d['agree']} of {d['of']} beats "
                f"(display only - never given to the AI)")
    if n in (6, 7):
        return f"{head}: {len(p['findings'])} findings"
    if n == 8:
        return f"{head}: {', '.join(m.id for m in p['matches']) or 'no rules'} matched"
    r = p["report"]
    kind = "AI-written, fact-checked" if r.kind == "llm" else "template"
    return f"{head}: {kind}" + (f" ({r.note})" if r.note else "")


def list_options(deps: orchestrator.Deps) -> None:
    print("ECG records (MIT-BIH; paced records 102, 104, 107, 217 are not offered):")
    for r in ecg_window.offered_records():
        print(f"  {r.name}  {r.tag:10s} {r.note}")
    if deps.chart is None:
        print(f"\nICU stays: {deps.unavailable.get('chart', 'not loaded')}")
        return
    print(f"\nICU stays ({deps.source}):")
    for row in deps.chart.cohort.sort_values("icustay_id").itertuples(index=False):
        age = ">89" if row.age_over_89 == 1 else f"{row.age:.0f}"
        reason = deps.chart.admission_reason.get(int(row.hadm_id), "")
        print(f"  {row.icustay_id}  age {age:>3s} {row.gender}  {reason}")


def save(case, report, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"record{case.record or 'none'}_stay{case.stay or 'none'}"
    (out_dir / f"{stem}.txt").write_text(report.text + "\n")
    payload = {"banner": case.banner, "findings": [f.to_dict() for f in case.all_findings()],
               "report": {"kind": report.kind, "model": report.model, "note": report.note,
                          "text": report.text}}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return out_dir / f"{stem}.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="list the ECG records and ICU stays, then exit")
    parser.add_argument("--ecg-record", default=None)
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the record")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--detect-peaks", action="store_true",
                        help="find beats with the QRS detector instead of the annotations")
    parser.add_argument("--stay", type=int, default=None, help="MIMIC-III ICU stay id")
    parser.add_argument("--source", choices=sorted(mimic_config.SOURCES), default="demo")
    parser.add_argument("--mimic-dir", type=Path, default=None,
                        help="folder with features.csv.gz, cohort.csv.gz and the model")
    parser.add_argument("--mimic-raw-dir", type=Path, default=None,
                        help="folder with the raw MIMIC-III CSV tables")
    parser.add_argument("--mitdb-dir", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None, help="ECG inference bundle")
    parser.add_argument("--save", action="store_true",
                        help="also write the report under <data root>/processed/agentic/reports/")
    parser.add_argument("--llm", default=DEFAULT_MODEL,
                        help=f"local Ollama model for the report and chat; 'on' = "
                             f"{DEFAULT_MODEL} (the default), 'off' = no LLM")
    parser.add_argument("--ask", action="append", default=[], metavar="QUESTION",
                        help="ask a question about the case after the report (repeatable)")
    parser.add_argument("--chat", action="store_true",
                        help="after the report, keep asking questions until an empty line")
    return parser


def run(args, deps, write_report=None):
    """Print every stage; return (case, report), or None on a bad request."""
    if write_report is not None:
        deps.write_report = write_report
    req = orchestrator.CaseRequest(record=args.ecg_record, start_s=args.start,
                                   seconds=args.seconds, stay=args.stay,
                                   detect_peaks=args.detect_peaks)
    try:
        last = None
        for event in orchestrator.run_case(req, deps):
            print(describe(event))
            last = event
    except (ecg_window.WindowError, EcgError, KeyError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return None
    return last.payload["case"], last.payload["report"]


def llm_name(choice: str) -> str | None:
    """--llm value -> model name; 'on' is the default model, 'off' is none."""
    choice = choice.strip()
    if choice.lower() == "off":
        return None
    return DEFAULT_MODEL if choice.lower() == "on" else choice


def make_llm(choice: str):
    """An Ollama client if the model is ready, else None (the template is used)."""
    name = llm_name(choice)
    if name is None:
        return None
    client = OllamaClient(name)
    status = client.status()
    print(f"LLM: {client.model} ready" if status.ok
          else f"LLM: {status.message} -> template report, chat off")
    return client if status.ok else None


def questions(args):
    yield from args.ask
    while args.chat:
        try:
            q = input("\nask> ")
        except EOFError:
            return
        if not q.strip() or q.strip().lower() in ("quit", "exit"):
            return
        yield q


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list and args.ecg_record is None and args.stay is None:
        parser.error("give --ecg-record, --stay, or both (or --list)")
    deps = orchestrator.default_deps(
        need_ecg=args.ecg_record is not None, need_chart=args.list or args.stay is not None,
        source=args.source, mimic_dir=args.mimic_dir, mimic_raw_dir=args.mimic_raw_dir,
        mitdb_dir=args.mitdb_dir, bundle_path=args.bundle)
    if args.list:
        list_options(deps)
        return 0
    client = make_llm(args.llm)
    result = run(args, deps, report_agent.make_writer(client))
    if result is None:
        return 1
    case, report = result
    print("\n" + report.text)
    if report.kind == "template" and report.problems:
        print("\nWhy the AI draft was rejected:")
        print("\n".join(f"  - {p.detail}" for p in report.problems))
    if args.save:
        print(f"\nSaved {save(case, report, paths.data_root() / 'processed' / 'agentic' / 'reports')}")
    if args.ask or args.chat:
        agent = chat_agent.ChatAgent(case, glossary.load_glossary(), client)
        history = []
        for q in questions(args):
            a = agent.ask(q, history)
            cites = f" [{', '.join(a.cites)}]" if a.cites else ""
            checked = " (fact-checked)" if a.verified and not a.refused else ""
            print(f"\nQ: {q}\nA: {a.text}{cites}{checked}")
            history.append((q, a.text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
