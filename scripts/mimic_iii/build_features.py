#!/usr/bin/env python3
"""Build the MIMIC-III cohort and its feature table.

    python scripts/mimic_iii/build_features.py                  # the open Demo
    python scripts/mimic_iii/build_features.py --source full    # credentialed MIMIC-III

Reads PATIENTS, ADMISSIONS, ICUSTAYS, DIAGNOSES_ICD, D_ITEMS, D_LABITEMS,
CHARTEVENTS and LABEVENTS from $SMART_HEALTH_DATA/<mimiciii-demo|mimiciii>/ and
writes, outside the repository, to $SMART_HEALTH_DATA/processed/<same name>/:

    cohort.csv.gz        one row per included ICU stay (ids, times, label)
    features.csv.gz      the model-ready feature table
    build_summary.json   cohort flow, row accounting, ITEMID labels, task, provenance

The event tables are read in chunks (--chunksize rows at a time), so full
CHARTEVENTS (~330 M rows) is processed without loading it whole. Expect tens of
minutes on the full database; the Demo takes about a second.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import provenance  # noqa: E402
from mimic_iii import cohort, config, features, ingestion  # noqa: E402
from mimic_iii.errors import DataSourceMissingError, MimicError  # noqa: E402

EXIT_BLOCKED = 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(config.SOURCES), default="demo")
    parser.add_argument("--root", type=Path, default=None, help="raw CSV directory (overrides --source location)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG_PATH)
    parser.add_argument("--chunksize", type=int, default=ingestion.DEFAULT_CHUNKSIZE)
    args = parser.parse_args(argv)

    spec = config.load_task_spec(args.config)
    root = (args.root or config.source_dir(args.source)).expanduser()
    out_dir = (args.out_dir or config.output_dir(args.source)).expanduser()

    print(f"Source   : {args.source} ({root})")
    print(f"Output   : {out_dir}")
    print(f"Task     : {spec.name}")
    try:
        ingestion.require_source(root, args.source)
    except DataSourceMissingError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED

    t0 = time.time()
    try:
        tables = ingestion.validate_source(root)
        itemid_labels = ingestion.check_itemids(root)
        patients = ingestion.read_table(root, "PATIENTS")
        admissions = ingestion.read_table(root, "ADMISSIONS")
        icustays = ingestion.read_table(root, "ICUSTAYS")
        diagnoses = ingestion.read_table(root, "DIAGNOSES_ICD")
        result = cohort.build_cohort(patients, admissions, icustays, diagnoses, spec)

        print("\nCohort flow")
        for row in result.flow:
            print(f"  {row['step']:42s} {row['stays']:>7,} stays  {row['subjects']:>7,} patients")
        for note in result.notes:
            print(f"  note: {note}")

        def progress(stats: ingestion.IngestStats) -> None:
            if stats.chunks % 20 == 0:
                print(f"  {stats.table}: {stats.rows_read:,} rows read, "
                      f"{stats.rows_wanted_itemid:,} wanted", flush=True)

        print("\nReading events (chunked)")
        events, event_stats = features.extract_events(
            root, result.cohort, spec, chunksize=args.chunksize, progress=progress)
        table = features.build_feature_table(result.cohort, events)
    except MimicError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = out_dir / "cohort.csv.gz"
    features_path = out_dir / "features.csv.gz"
    result.cohort.to_csv(cohort_path, index=False)
    table.to_csv(features_path, index=False)

    summary = {
        "source": args.source,
        "source_description": config.SOURCE_DESCRIPTIONS[args.source],
        "raw_dir": str(root),
        "tables": tables,
        "task": spec.describe(),
        "cohort": result.to_dict(),
        "events": event_stats.to_dict(),
        "itemid_labels": itemid_labels,
        "feature_table": {
            "path": str(features_path),
            "sha256": provenance.sha256_file(features_path),
            "rows": int(len(table)),
            "columns": int(table.shape[1]),
            "model_input_columns": features.model_input_columns(),
            "missing_rate": {c.removesuffix("_missing"): float(table[c].mean())
                             for c in features.missing_flag_names()},
        },
        "seconds": round(time.time() - t0, 2),
        "provenance": provenance.provenance(),
    }
    summary_path = provenance.write_json(out_dir / "build_summary.json", summary)

    c = result.to_dict()
    print(f"\nKept events: {sum(event_stats.kept.values()):,} "
          f"(chart {event_stats.chart['rows_kept']:,} rows read in, "
          f"lab {event_stats.lab['rows_kept']:,})")
    dropped = {k: v for k, v in event_stats.dropped.items() if v}
    print(f"Dropped    : {dropped}")
    print(f"\nCohort     : {c['n_stays']} stays, {c['n_positive']} positive "
          f"({c['n_positive'] / max(c['n_stays'], 1):.0%})")
    print(f"Features   : {table.shape[0]} rows x {len(features.model_input_columns())} model inputs")
    print(f"Wrote      : {features_path}\n             {cohort_path}\n             {summary_path}")
    if args.source == "demo":
        print("\nDemo data: verifies the pipeline only. See the README before reading anything into it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
