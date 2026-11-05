#!/usr/bin/env python3
"""Run the shared inference interface on an ECG, a clinical record, or both.

    python scripts/multimodal/infer.py --ecg-record 100                       # ECG only
    python scripts/multimodal/infer.py --clinical-stay 200282                 # clinical only (Demo)
    python scripts/multimodal/infer.py --ecg-record 100 --clinical-stay 200282 --out result.json

--ecg-record reads a raw MIT-BIH record (the preferred lead, its first
--seconds seconds) and its beat annotations as R peaks; --detect-peaks uses the
QRS detector instead. --clinical-stay reads one ICU stay's row from the MIMIC-III
feature table built by scripts/mimic_iii/build_features.py, and the model
trained by scripts/mimic_iii/train.py. --clinical-json reads features from a
JSON file instead.

The output is JSON. An MIT-BIH record and a MIMIC-III stay are different
people: the two sections are reported side by side, never combined, and fusion
is reported as BLOCKED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mimic_iii import config as mimic_config  # noqa: E402
from mimic_iii import dataset as mimic_dataset  # noqa: E402
from mimic_iii import features as mimic_features  # noqa: E402
from mit_bih import config as ecg_config  # noqa: E402
from mit_bih import io as ecg_io  # noqa: E402
from mit_bih import labels as ecg_labels  # noqa: E402
from multimodal import inference  # noqa: E402
from multimodal.clinical_encoder import ClinicalInput  # noqa: E402
from multimodal.ecg_encoder import EcgInput  # noqa: E402


def load_ecg(record: str, record_dir: Path, seconds: float | None, detect: bool) -> EcgInput:
    rec = ecg_io.load_record(record_dir, record)
    sig = rec.signal
    end = sig.size if seconds is None else min(sig.size, int(seconds * rec.fs))
    peaks = None
    if not detect:
        ann = ecg_io.load_annotations(record_dir, record)
        beat = [m.outcome == "beat" for m in ecg_labels.map_symbols(ann.symbols)]
        peaks = np.asarray(ann.samples, dtype=np.int64)[beat]
        peaks = peaks[peaks < end]
    return EcgInput(sig[:end], rec.fs, peaks, record_id=f"mitdb/{record} ({rec.lead})")


def load_clinical(stay: int, out_dir: Path) -> tuple[ClinicalInput, str]:
    table = mimic_features.read_feature_table(out_dir / "features.csv.gz")
    rows = table[table["icustay_id"] == stay]
    if rows.empty:
        raise SystemExit(f"ICU stay {stay} is not in the cohort feature table at {out_dir}")
    row = rows.iloc[0]
    note = "split unknown"
    split_path = out_dir / "splits.json"
    if split_path.exists():
        split = mimic_dataset.read_split(split_path)
        for name in mimic_dataset.SPLIT_NAMES:
            if int(row["subject_id"]) in split.subjects(name):
                note = f"this stay's patient is in the model's {name} split"
    return ClinicalInput.from_feature_table_row(row), note


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ecg-record", default=None)
    parser.add_argument("--mitdb-dir", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="use the first N seconds of the record (default 60; 0 = whole record)")
    parser.add_argument("--detect-peaks", action="store_true")
    parser.add_argument("--clinical-stay", type=int, default=None)
    parser.add_argument("--clinical-json", type=Path, default=None)
    parser.add_argument("--source", choices=sorted(mimic_config.SOURCES), default="demo")
    parser.add_argument("--mimic-dir", type=Path, default=None,
                        help="folder with features.csv.gz and the trained model")
    parser.add_argument("--include-beats", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.ecg_record is None and args.clinical_stay is None and args.clinical_json is None:
        parser.error("give --ecg-record, --clinical-stay/--clinical-json, or both")

    mimic_dir = (args.mimic_dir or mimic_config.output_dir(args.source)).expanduser()
    ecg_enc, clin_enc, notes = inference.load_default_encoders(
        mimic_dir if (args.clinical_stay is not None or args.clinical_json) else None)

    ecg = clinical = None
    if args.ecg_record is not None:
        ecg = load_ecg(args.ecg_record, (args.mitdb_dir or ecg_config.mitdb_dir()).expanduser(),
                       None if args.seconds == 0 else args.seconds, args.detect_peaks)
    if args.clinical_stay is not None:
        clinical, split_note = load_clinical(args.clinical_stay, mimic_dir)
        notes.append(f"clinical stay {args.clinical_stay}: {split_note}")
    elif args.clinical_json is not None:
        clinical = ClinicalInput(json.loads(args.clinical_json.read_text()))

    result = inference.run_inference(
        inference.InferenceRequest(ecg=ecg, clinical=clinical, include_beats=args.include_beats),
        ecg_encoder=ecg_enc, clinical_encoder=clin_enc)
    if notes:
        result["notes"] = notes

    text = inference.to_json(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"Wrote {args.out}")
    print(text)
    ok = all(result[m]["status"] in ("ok", "not_provided") for m in ("ecg", "clinical"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
