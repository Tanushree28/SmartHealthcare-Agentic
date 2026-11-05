#!/usr/bin/env python3
"""Score the inference bundle on DS2 and compare with the published number.

    python scripts/mit_bih/evaluate_bundle.py                 # from the saved beats
    python scripts/mit_bih/evaluate_bundle.py --from-raw      # from the raw WFDB records

Default: loads the DS2 beats the Phase 1 pipeline saved and scores them with
the bundle's ensemble -- a CPU re-run of the published Phase 1g evaluation.

--from-raw: instead reads each DS2 record's raw signal and beat annotations
and prepares the beats through mit_bih.inference (filter, segment, normalize,
RR features) -- the path a new ECG takes. Matching numbers show the saved
inference preprocessing is the preprocessing the models were trained with.

Writes bundle_eval_<ensemble>[_raw].json into the processed folder. Takes about
7 minutes on a laptop CPU for the 20-model ensemble
(measured 2026-10-14: 7 min 24 s for 49,646 beats).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import provenance  # noqa: E402
from mit_bih import config, dataset, evaluate, inference, io, labels, splits  # noqa: E402


def raw_beats(record_dir: Path, name: str, bundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Beats of one record prepared by the inference path, with their labels."""
    rec = io.load_record(record_dir, name)
    ann = io.load_annotations(record_dir, name)
    mappings = labels.map_symbols(ann.symbols)
    is_beat = np.array([m.outcome == "beat" for m in mappings])
    peaks = np.asarray(ann.samples, dtype=np.int64)[is_beat]
    truth = np.array([m.class_index for m, b in zip(mappings, is_beat) if b], dtype=np.int64)

    prepared = inference.prepare_beats(rec.signal, rec.fs, peaks, bundle)
    label_at = dict(zip(peaks.tolist(), truth.tolist()))
    y = np.array([label_at[int(p)] for p in prepared.r_peaks], dtype=np.int64)
    return prepared.X, prepared.rr, y


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=inference.DEFAULT_BUNDLE_PATH)
    parser.add_argument("--processed-dir", type=Path,
                        default=config.data_root() / config.PROCESSED_DIRNAME / "mitdb-ds12")
    parser.add_argument("--from-raw", action="store_true",
                        help="prepare beats from the raw WFDB records through mit_bih.inference")
    parser.add_argument("--records", nargs="*", default=None,
                        help="records to score (default: DS2 without paced records)")
    args = parser.parse_args(argv)

    t0 = time.time()
    clf = inference.EcgClassifier.from_bundle(args.bundle)
    bundle = clf.bundle
    full_ds2 = [r for r in splits.DS2_RECORDS if r not in config.PACED_RECORDS]
    records = args.records or full_ds2
    is_full_ds2 = sorted(records) == sorted(full_ds2)
    print(f"Ensemble : {bundle.ensemble} ({len(clf.models)} checkpoints)")
    print(f"Records  : {len(records)} ({'raw WFDB via mit_bih.inference' if args.from_raw else 'saved beats'})")

    if args.from_raw:
        Xs, RRs, ys = [], [], []
        for name in records:
            X, rr, y = raw_beats(config.mitdb_dir(), name, bundle)
            Xs.append(X), RRs.append(rr), ys.append(y)
        X, rr, y = np.concatenate(Xs), np.concatenate(RRs), np.concatenate(ys)
    else:
        data = dataset.load_records(args.processed_dir.expanduser() / "beats", records)
        X, rr, y = data.X, data.rr, data.y

    scores = clf.score_beats(X, rr)
    result = evaluate.evaluate(y, scores.argmax(axis=1), label=f"{bundle.ensemble} on DS2")
    headline = result.macro_f1_of(config.HEADLINE_CLASSES)
    # The published number is for all of DS2; a subset score is not comparable.
    published = inference.PUBLISHED_DS2_MACRO_F1.get(bundle.ensemble) if is_full_ds2 else None

    print()
    print(evaluate.format_report(result))
    print(f"\nHeadline macro-F1 (N/S/V/F): {headline:.4f}")
    if published is not None:
        print(f"Published (Colab GPU)      : {published:.4f}   difference {headline - published:+.4f}")
    elif not is_full_ds2:
        print(f"Subset of {len(records)} records: not comparable to the published all-DS2 number.")
    print(f"\n{bundle.calibration}")

    suffix = "_raw" if args.from_raw else ""
    out = args.processed_dir.expanduser() / f"bundle_eval_{bundle.ensemble}{suffix}.json"
    provenance.write_json(out, {
        "ensemble": bundle.ensemble,
        "n_models": len(clf.models),
        "beats_from": "raw WFDB via mit_bih.inference" if args.from_raw else "saved Phase 1 beats",
        "records": records,
        "is_full_ds2": is_full_ds2,
        "n_beats": int(y.size),
        "headline_macro_f1": headline,
        "published_headline_macro_f1": published,
        "metrics": result.to_dict(),
        "calibration": bundle.calibration,
        "seconds": round(time.time() - t0, 1),
        "provenance": provenance.provenance({"bundle": args.bundle}),
    })
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
