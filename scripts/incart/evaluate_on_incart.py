#!/usr/bin/env python3
"""External test: score the unchanged MIT-BIH ensemble on INCART, once.

    python scripts/incart/evaluate_on_incart.py             # the run (about 33 min)
    python scripts/incart/evaluate_on_incart.py --smoke 2   # pipeline check: counts only

Requires resampling_check.py to have run first and reproduced the published
DS2 macro-F1 0.5554; its DS2 numbers are the reference INCART is compared with.

1. Beat-position check (no model): offset from each annotated beat to the
   band-passed peak within +/-50 ms, INCART (after 257 -> 360 Hz) vs DS2.
   If the medians differ by more than 3 samples (8.3 ms), INCART is also
   scored with every mark moved to that peak. The as-marked score stays the
   headline either way.
2. Score lead II of every INCART record without paced beats, with the
   D_all_four ensemble; predicted class = argmax.
3. Headline macro-F1 (N/S/V/F) with a 95% interval that resamples patients
   (records, if no valid patient mapping exists), 1,000 draws.
4. "The model holds up" if DS2's headline falls inside INCART's interval.

Nothing is tuned on INCART. The run refuses to replace a finished one unless
--overwrite is given. --smoke prints beat counts and timing only -- no
metrics, nothing written -- so testing the pipeline does not reveal results.

Writes external_test.json and per-beat predictions (predictions/<variant>/
<record>.npz) into $SMART_HEALTH_DATA/processed/incartdb/. Prints aggregates
only.
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

DATASET = "St Petersburg INCART 12-lead Arrhythmia Database v1.0.0 (PhysioNet incartdb)"


def score_records(clf, beats_iter, variant: str, pred_dir: Path, patient_of: dict, t0: float):
    """Score each record's beats; return the variant's aggregate block and report."""
    (pred_dir / variant).mkdir(parents=True, exist_ok=True)
    matrices, rows, ys, preds, counts = {}, [], [], [], Counter()
    for k, beats in enumerate(beats_iter, start=1):
        if not beats.reconciles():
            raise SystemExit(f"beat counts do not reconcile for record {beats.name} ({variant}): "
                             f"{beats.counts()}")
        scores = clf.score_beats(beats.X, beats.rr)
        pred = scores.argmax(axis=1) if scores.size else np.zeros(0, dtype=np.int64)
        np.savez_compressed(pred_dir / variant / f"{beats.name}.npz", y=beats.y, pred=pred,
                            scores=scores.astype(np.float32), r_peaks=beats.r_peaks)
        matrices[beats.name] = ev.confusion(beats.y, pred)
        rows.append(ev.record_row(beats.name, beats.y, pred))
        ys.append(beats.y)
        preds.append(pred)
        counts.update(beats.counts())
        print(f"  [{variant} {k:>2}] {beats.name}  {time.time() - t0:6.0f} s", flush=True)

    y, pred = np.concatenate(ys), np.concatenate(preds)
    result = mit_evaluate.evaluate(y, pred, label=f"INCART lead II, {variant}")
    block = {
        "headline_macro_f1": result.macro_f1_of(mit_config.HEADLINE_CLASSES),
        "ci95": list(ev.grouped_bootstrap_ci(ev.sum_by_group(matrices, patient_of))),
        "metrics": result.to_dict(),
        "beat_counts": {**counts, "reconciles": True},
        "per_record": rows,
    }
    return block, mit_evaluate.format_report(result)


def smoke(clf, args) -> int:
    """Counts and timing for the first N records. No metrics, nothing written."""
    ok = True
    for name in config.RECORDS[: args.smoke]:
        t = time.time()
        rec = io.load_labelled_record(args.dir, name)
        beats = ev.record_beats(rec, clf.bundle)
        clf.score_beats(beats.X, beats.rr)  # timed, not shown
        ok &= beats.reconciles()
        print(f"  {name}: {beats.counts()}  reconciles: {beats.reconciles()}  "
              f"paced: {rec.has_paced}  {time.time() - t:.0f} s")
    print("Smoke run: counts and timing only; no metrics, nothing written.")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=inference.DEFAULT_BUNDLE_PATH)
    parser.add_argument("--dir", type=Path, default=config.raw_dir(), help="raw INCART records")
    parser.add_argument("--mitdb-dir", type=Path, default=mit_config.mitdb_dir())
    parser.add_argument("--out-dir", type=Path, default=config.processed_dir())
    parser.add_argument("--smoke", type=int, default=None, metavar="N",
                        help="pipeline check on the first N records: counts only")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a finished run (record why in the write-up)")
    args = parser.parse_args(argv)
    args.dir = args.dir.expanduser()
    out_dir = args.out_dir.expanduser()
    result_path = out_dir / "external_test.json"
    verdict_path = out_dir / "resampling_check.json"

    if args.smoke is None:
        try:
            ev.refuse_to_overwrite(result_path, args.overwrite)
            ensemble = inference.load_bundle(args.bundle, verify=False).ensemble
            verdict = ev.read_resampling_verdict(verdict_path, ensemble=ensemble)
        except ev.PreconditionError as exc:
            print(f"Refusing to run: {exc}")
            return 1

    t0 = time.time()
    clf = inference.EcgClassifier.from_bundle(args.bundle)
    bundle = clf.bundle
    print(f"Ensemble : {bundle.ensemble} ({len(clf.models)} checkpoints)")
    if args.smoke is not None:
        return smoke(clf, args)

    # -- 1. First pass, no model: labels, paced records, beat positions -------
    patient_map = io.load_patient_map(args.dir)
    print(f"Grouping : {patient_map.note}")
    scored, paced, symbols = [], [], Counter()
    inc_off, inc_amp = [], []
    for name in config.RECORDS:
        rec = io.load_labelled_record(args.dir, name)
        symbols.update(rec.symbol_counts)
        if rec.has_paced:
            paced.append(name)
            continue
        scored.append(name)
        off, amp = ev.record_positions(rec, bundle)
        inc_off.append(off), inc_amp.append(amp)
    if not scored:
        raise SystemExit("every INCART record has paced beats; nothing to score")

    ds2_records = [r for r in splits.DS2_RECORDS if r not in mit_config.PACED_RECORDS]
    ds2_off, ds2_amp = [], []
    for name in ds2_records:
        rec = io.load_labelled_record(args.mitdb_dir.expanduser(), name, fs=mit_config.FS,
                                      preference=mit_config.LEAD_PREFERENCE)
        off, amp = ev.record_positions(rec, bundle)
        ds2_off.append(off), ds2_amp.append(amp)

    pos_incart = ev.summarise_positions(np.concatenate(inc_off), np.concatenate(inc_amp), bundle.fs)
    pos_ds2 = ev.summarise_positions(np.concatenate(ds2_off), np.concatenate(ds2_amp), bundle.fs)
    fires = ev.position_rule_fires(pos_incart["median_samples"], pos_ds2["median_samples"])
    print(f"Records  : {len(scored)} scored, {len(paced)} left out for paced beats")
    for label, p in (("INCART", pos_incart), ("DS2", pos_ds2)):
        print(f"Position : {label:<6} median offset {p['median_ms']:+.1f} ms "
              f"(IQR {p['q1_ms']:+.1f} to {p['q3_ms']:+.1f}), median peak {p['median_peak_mv']:.2f} mV")
    print(f"Position rule fired: {'YES -- the moved-to-peak variant is scored too' if fires else 'no'}")

    # -- 2. Score -------------------------------------------------------------
    pred_dir = out_dir / "predictions"

    def beats_of(refine: bool):
        for name in scored:
            yield ev.record_beats(io.load_labelled_record(args.dir, name), bundle, refine=refine)

    as_marked, report = score_records(clf, beats_of(False), "as_marked", pred_dir,
                                      patient_map.patient_of, t0)
    refined = None
    if fires:
        refined, _ = score_records(clf, beats_of(True), "refined", pred_dir,
                                   patient_map.patient_of, t0)

    # -- 3. Decisions -----------------------------------------------------------
    ds2 = verdict["original"]
    holds = ev.holds_up(ds2["headline_macro_f1"], tuple(as_marked["ci95"]))
    difference = as_marked["headline_macro_f1"] - ds2["headline_macro_f1"]

    print()
    print(report)
    lo, hi = as_marked["ci95"]
    dlo, dhi = ds2["ci95"]
    print()
    print(f"INCART headline macro-F1 {as_marked['headline_macro_f1']:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"DS2    headline macro-F1 {ds2['headline_macro_f1']:.4f}   95% CI [{dlo:.4f}, {dhi:.4f}]")
    print(f"Difference (INCART - DS2) {difference:+.4f}")
    print(f"The model holds up on INCART: {'yes' if holds else 'NO'}")
    print(f"Resampling matters (DS2 round trip): {'YES' if verdict['resampling_matters'] else 'no'} "
          f"({verdict['difference']:+.4f})")
    if refined is not None:
        print(f"Moved-to-peak variant headline macro-F1 {refined['headline_macro_f1']:.4f} (not the headline)")
    print(f"\n{bundle.calibration}\n{config.NOT_CLINICAL}")

    path = provenance.write_json(result_path, {
        "question": "Does the MIT-BIH beat model work on new patients and equipment?",
        "dataset": DATASET,
        "lead": config.LEAD,
        "fs_recorded": config.FS,
        "fs_model": bundle.fs,
        "ensemble": bundle.ensemble,
        "n_models": len(clf.models),
        "records_scored": scored,
        "records_excluded_paced": paced,
        "patient_grouping": {"by_patient": patient_map.by_patient,
                             "n_groups": len(set(patient_map.patient_of[r] for r in scored)),
                             "note": patient_map.note, "groups": patient_map.groups},
        "n_boot": config.N_BOOT,
        "seed": config.SEED,
        "symbol_counts": dict(symbols),
        "position_check": {"incart": pos_incart, "ds2": pos_ds2,
                           "difference_samples": pos_incart["median_samples"] - pos_ds2["median_samples"],
                           "threshold_samples": config.POSITION_THRESHOLD_SAMPLES,
                           "rule_fired": fires,
                           "amplitude_note": "Median of the largest |band-passed| value within "
                                             "+/-50 ms of each mark, in mV. Diagnostic only; "
                                             "nothing is rescaled."},
        "as_marked": as_marked,
        "refined": refined,
        "ds2_reference": {"headline_macro_f1": ds2["headline_macro_f1"], "ci95": ds2["ci95"],
                          "source": "resampling_check.json, original (360 Hz as recorded)"},
        "difference_from_ds2": difference,
        "holds_up": holds,
        "holds_up_rule": "The DS2 headline macro-F1 falls inside INCART's 95% interval.",
        "resampling_check": {"difference": verdict["difference"],
                             "resampling_matters": verdict["resampling_matters"],
                             "threshold": verdict["threshold"]},
        "calibration": bundle.calibration,
        "not_clinical": config.NOT_CLINICAL,
        "seconds": round(time.time() - t0, 1),
        "provenance": provenance.provenance({"bundle": args.bundle, "resampling_check": verdict_path}),
    })
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
