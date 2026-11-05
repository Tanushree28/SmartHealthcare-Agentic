#!/usr/bin/env python3
"""Watch the fusion network ("team brain") learn -- on MADE-UP patients only.

    python scripts/multimodal/demo_fusion_synthetic.py

No real patient is used. The script invents patients, each with a fake ECG
summary (13 numbers, the size the real ECG encoder produces) and a fake
clinical vector (119 numbers, the size the real clinical encoder produces).
The made-up rule is that risk rises with the first ECG number AND with the
first clinical number, so a model needs both to do well.

It then trains the fusion network and scores new made-up patients three ways:
with both inputs, with the ECG input only, and with the clinical input only.
Finally it shows that the pairing check refuses to treat an MIT-BIH heartbeat
and a MIMIC-III patient as one person.

What this shows: the fusion code works and can use both inputs.
What it does not show: anything about real patients.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from multimodal import fusion  # noqa: E402
from multimodal.errors import PairingError  # noqa: E402

ECG_DIM, CLINICAL_DIM = 13, 119


def make_patients(n: int, seed: int, first_id: int) -> fusion.PairedArrays:
    rng = np.random.default_rng(seed)
    ecg = rng.normal(size=(n, ECG_DIM))
    clinical = rng.normal(size=(n, CLINICAL_DIM))
    risk = 1 / (1 + np.exp(-(2.5 * ecg[:, 0] + 2.5 * clinical[:, 0])))
    y = (rng.random(n) < risk).astype(np.int64)
    ones = np.ones(n, dtype=bool)
    return fusion.PairedArrays(ecg, clinical, ones, ones.copy(), y, np.arange(first_id, first_id + n))


def only(data: fusion.PairedArrays, keep: str) -> fusion.PairedArrays:
    """The same patients with one input hidden (marked absent, zeroed)."""
    e_on = keep == "ecg"
    return fusion.PairedArrays(
        data.ecg if e_on else np.zeros_like(data.ecg),
        data.clinical if not e_on else np.zeros_like(data.clinical),
        np.full(data.y.shape, e_on), np.full(data.y.shape, not e_on),
        data.y, data.subject_id)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    print("\nSYNTHETIC DEMO - every patient below is invented. Nothing here is medical.\n")
    print("Step 1. Invent patients. Each has a fake ECG summary (13 numbers) and a fake")
    print("        chart (119 numbers). Made-up rule: risk goes up with ECG number #1")
    print("        AND chart number #1. All other numbers are noise.")
    train = make_patients(600, args.seed, 0)
    val = make_patients(300, args.seed + 1, 10_000)
    test = make_patients(1000, args.seed + 2, 20_000)
    print(f"        {len(train.y)} to learn from, {len(val.y)} to decide when to stop, "
          f"{len(test.y)} new ones to test on.\n")

    print("Step 2. Train the team brain (ConcatFusion) on the made-up patients...")
    model, info = fusion.train_fusion(train, val, fusion.FusionTrainConfig(epochs=args.epochs,
                                                                           seed=args.seed))
    print(f"        stopped after {info['epochs_run']} epochs.\n")

    both = fusion.predict(model, test)
    ecg_only = fusion.predict(model, only(test, "ecg"))
    clin_only = fusion.predict(model, only(test, "clinical"))
    print("Step 3. Score 1,000 NEW made-up patients (AUROC: 0.5 = guessing, 1.0 = perfect)")
    print(f"        both inputs (team)  : {roc_auc_score(test.y, both):.3f}")
    print(f"        ECG input only      : {roc_auc_score(test.y, ecg_only):.3f}")
    print(f"        chart input only    : {roc_auc_score(test.y, clin_only):.3f}")
    print("        The team wins because the made-up rule needs both clues.\n")

    print("Step 4. Five made-up patients up close")
    print(f"        {'patient':>8s}  {'ECG clue':>8s}  {'chart clue':>10s}  {'really':>8s}  "
          f"{'team':>5s}  {'ECG only':>8s}  {'chart only':>10s}")
    for i in range(5):
        print(f"        {'fake-' + str(i + 1):>8s}  {test.ecg[i, 0]:8.2f}  {test.clinical[i, 0]:10.2f}  "
              f"{'event' if test.y[i] else 'no event':>8s}  {both[i]:5.2f}  "
              f"{ecg_only[i]:8.2f}  {clin_only[i]:10.2f}")
    print()

    print("Step 5. Try to train on a REAL MIT-BIH heartbeat paired with a REAL MIMIC-III patient")
    attempt = pd.DataFrame([{
        "subject_id": 1, "ecg_subject_id": 1, "clinical_subject_id": 1,
        "ecg_source": "mitbih", "clinical_source": "mimiciii",
        "ecg_time": "2150-01-01 10:00", "index_time": "2150-01-02 08:00",
        "label": 1, "split": "train"}])
    try:
        fusion.validate_pairs(attempt)
        print("        accepted (this should never happen)")
        return 1
    except PairingError as exc:
        print(f"        REFUSED: {exc}\n")

    print("Summary: the fusion code works and can combine two inputs, but it has only ever")
    print("seen invented patients. Real training needs ECG + chart from the SAME people.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
