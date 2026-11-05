#!/usr/bin/env python3
"""Resampling check: does 360 -> 257 -> 360 Hz change the DS2 score?

    python scripts/incart/resampling_check.py                # all 22 DS2 records
    python scripts/incart/resampling_check.py --records 100 103   # a subset (not comparable)

INCART is recorded at 257 Hz and resampled to 360 Hz before scoring. To tell
the effect of the sampling rate apart from the effect of new patients and
equipment, the 22 DS2 records (paced records excluded) are scored twice from
their raw WFDB files, through the same preparation as
``evaluate_bundle.py --from-raw``:

- original:   360 Hz as recorded. Must reproduce the published macro-F1 0.5554.
- round trip: 360 -> 257 -> 360 Hz, annotation positions rounded onto the
              257 Hz grid and back, mirroring what INCART goes through.

"Resampling matters" if the two headline macro-F1 values differ by more than
0.01 (fixed in the design before the run). 95% intervals resample records
(22 records = 22 patients), 1,000 draws.

Writes resampling_check.json into $SMART_HEALTH_DATA/processed/incartdb/, which
evaluate_on_incart.py requires. Exits 1 if the original does not reproduce
0.5554. Takes about 19 minutes on a laptop CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import provenance  # noqa: E402
from incart import config  # noqa: E402
from incart import evaluate as ev  # noqa: E402
from incart import io  # noqa: E402
from mit_bih import config as mit_config  # noqa: E402
from mit_bih import evaluate as mit_evaluate  # noqa: E402
from mit_bih import inference, splits  # noqa: E402

VARIANTS = {"original": None, "round_trip": config.FS}


def summarise(name: str, matrices: dict, ys: list, preds: list, counts: Counter) -> dict:
    y, pred = np.concatenate(ys), np.concatenate(preds)
    result = mit_evaluate.evaluate(y, pred, label=f"DS2, {name}")
    return {
        "headline_macro_f1": result.macro_f1_of(mit_config.HEADLINE_CLASSES),
        "ci95": list(ev.grouped_bootstrap_ci(matrices)),
        "metrics": result.to_dict(),
        "counts": dict(counts),
        "report": mit_evaluate.format_report(result),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=inference.DEFAULT_BUNDLE_PATH)
    parser.add_argument("--mitdb-dir", type=Path, default=mit_config.mitdb_dir())
    parser.add_argument("--out-dir", type=Path, default=config.processed_dir())
    parser.add_argument("--records", nargs="*", default=None,
                        help="DS2 records to score (default: DS2 without paced records)")
    args = parser.parse_args(argv)

    t0 = time.time()
    clf = inference.EcgClassifier.from_bundle(args.bundle)
    bundle = clf.bundle
    full_ds2 = [r for r in splits.DS2_RECORDS if r not in mit_config.PACED_RECORDS]
    records = args.records or full_ds2
    is_full = sorted(records) == sorted(full_ds2)
    print(f"Ensemble : {bundle.ensemble} ({len(clf.models)} checkpoints)")
    print(f"Records  : {len(records)} DS2 records, raw WFDB, scored as recorded and after "
          f"360 -> {config.FS} -> 360 Hz")

    acc = {v: {"matrices": {}, "ys": [], "preds": [], "counts": Counter()} for v in VARIANTS}
    for k, name in enumerate(records, start=1):
        rec = io.load_labelled_record(args.mitdb_dir.expanduser(), name, fs=mit_config.FS,
                                      preference=mit_config.LEAD_PREFERENCE)
        for variant, via in VARIANTS.items():
            beats = ev.record_beats(rec, bundle, round_trip_via=via)
            if not beats.reconciles():
                raise SystemExit(f"beat counts do not reconcile for record {name} ({variant}): "
                                 f"{beats.counts()}")
            pred = clf.score_beats(beats.X, beats.rr).argmax(axis=1)
            a = acc[variant]
            a["matrices"][name] = ev.confusion(beats.y, pred)
            a["ys"].append(beats.y)
            a["preds"].append(pred)
            a["counts"].update(beats.counts())
        print(f"  [{k:>2}/{len(records)}] {name}  {time.time() - t0:6.0f} s", flush=True)

    out = {v: summarise(v, a["matrices"], a["ys"], a["preds"], a["counts"]) for v, a in acc.items()}
    original = out["original"]["headline_macro_f1"]
    round_trip = out["round_trip"]["headline_macro_f1"]
    published = inference.PUBLISHED_DS2_MACRO_F1.get(bundle.ensemble) if is_full else None
    reproduces = (round(original, 4) == published) if published is not None else None
    matters = ev.resampling_matters(original, round_trip)

    for variant in VARIANTS:
        print()
        print(out[variant].pop("report"))
    print()
    for variant in VARIANTS:
        lo, hi = out[variant]["ci95"]
        print(f"{variant:<11} headline macro-F1 {out[variant]['headline_macro_f1']:.4f}   "
              f"95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"difference  {round_trip - original:+.4f}   (threshold {config.RESAMPLING_THRESHOLD})")
    print(f"Resampling matters: {'YES' if matters else 'no'}")
    if published is not None:
        print(f"Reproduces published {published:.4f}: {'yes' if reproduces else 'NO'}")
    else:
        print(f"Subset of {len(records)} records: not comparable to the published all-DS2 number.")
    print(f"\n{bundle.calibration}\n{config.NOT_CLINICAL}")

    filename = "resampling_check.json" if is_full else "resampling_check_subset.json"
    path = provenance.write_json(args.out_dir.expanduser() / filename, {
        "question": "Does resampling 360 -> 257 -> 360 Hz change the DS2 headline macro-F1?",
        "ensemble": bundle.ensemble,
        "n_models": len(clf.models),
        "records": records,
        "is_full_ds2": is_full,
        "via_fs": config.FS,
        "n_boot": config.N_BOOT,
        "seed": config.SEED,
        "interval_note": "95% percentile bootstrap resampling records (DS2: one record per patient).",
        "published_headline_macro_f1": published,
        "reproduces_published": reproduces,
        "original": out["original"],
        "round_trip": out["round_trip"],
        "difference": round_trip - original,
        "threshold": config.RESAMPLING_THRESHOLD,
        "resampling_matters": matters,
        "calibration": bundle.calibration,
        "not_clinical": config.NOT_CLINICAL,
        "seconds": round(time.time() - t0, 1),
        "provenance": provenance.provenance({"bundle": args.bundle}),
    })
    print(f"Wrote {path}")
    return 1 if reproduces is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
