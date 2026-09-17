#!/usr/bin/env python
"""Bring a Colab run's results and weights back into the repository.

Copies the metrics JSON into ``results/`` and the checkpoints into ``models/``,
then prints the comparison table so the numbers can be read immediately rather
than retyped from a notebook output cell.

Refuses a bundle whose arms disagree with ``train_nn.ARMS``: a stale archive
from an earlier code version would otherwise be written into ``results/`` and
read later as if it matched the code that is checked in.

Usage:
    ./venv/bin/python scripts/ingest_colab_results.py ~/Downloads/phase1d-results.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Metrics filenames, newest phase first.
METRICS_NAMES = ("ensemble_mix_metrics.json", "nn_v2_metrics.json", "nn_metrics.json")

#: Which phase each metrics filename belongs to, and the stem it lands under.
PHASE_BY_METRICS = {
    "ensemble_mix_metrics.json": ("mix", "ensemble-mix"),
    "nn_v2_metrics.json": ("nn_v2", "nn-v2"),
    "nn_metrics.json": ("nn", "nn"),
}
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ecg import config  # noqa: E402


def _extract(bundle: Path, into: Path) -> Path:
    if bundle.is_dir():
        return bundle
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(into)
    return into


def _table(payload: dict) -> str:
    """The comparison table, ranked by the Q-excluded headline metric."""
    rows = []
    for result in payload.get("results", []):
        per_class = {m["class"]: m for m in result["per_class"]}
        rows.append(
            {
                "arm": result.get("label", "?"),
                "headline": result.get("macro_f1_headline", float("nan")),
                "macro_f1": result.get("macro_f1", float("nan")),
                "macro_recall": result.get("macro_recall", float("nan")),
                "s_recall": per_class.get("S", {}).get("recall", float("nan")),
                "s_prec": per_class.get("S", {}).get("precision", float("nan")),
                "v_recall": per_class.get("V", {}).get("recall", float("nan")),
                "n": result.get("n_test", 0),
            }
        )
    rows.sort(key=lambda r: r["headline"], reverse=True)

    lines = [
        f"{'arm':<16}{'macroF1*':>10}{'macroF1':>9}{'macroRec':>10}"
        f"{'S rec':>8}{'S prec':>8}{'V rec':>8}{'n':>9}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"{r['arm']:<16}{r['headline']:>10.3f}{r['macro_f1']:>9.3f}"
            f"{r['macro_recall']:>10.3f}{r['s_recall']:>8.3f}"
            f"{r['s_prec']:>8.3f}{r['v_recall']:>8.3f}{r['n']:>9,}"
        )
    lines.append("")
    lines.append("* macroF1* is the ranking metric: N/S/V/F only, Q excluded.")
    lines.append("  Phase 1b floor -- random_forest rr+morphology: macroF1 0.405, S recall 0.271")
    return "\n".join(lines)


def _table_mix(payload: dict) -> str:
    """Phase 1g: candidate ensembles, chosen out-of-fold and then scored on DS2.

    The chosen row and the best-on-DS2 row are both marked. When they differ,
    selection cost something real -- and the honest number to quote is the
    chosen one, not the best one.
    """
    oof = payload.get("oof", {}).get("scores", {})
    ds2 = payload.get("ds2", {}).get("scores", {})
    selection = payload.get("selection", {})
    selected = selection.get("selected")
    reference = selection.get("reference")

    if not ds2:
        return "no compositions scored"

    best_on_ds2 = max(ds2, key=lambda k: ds2[k]["macro_average_precision"])

    lines = [
        f"{'composition':<28}{'OOF macroAP':>12}{'DS2 macroAP':>13}"
        f"{'DS2 S AP':>10}{'DS2 F1':>9}  ",
        "-" * 76,
    ]
    for name in ds2:
        marks = []
        if name == selected:
            marks.append("chosen")
        if name == best_on_ds2:
            marks.append("best on DS2")
        if name == reference:
            marks.append("reference")
        lines.append(
            f"{name:<28}"
            f"{oof.get(name, {}).get('macro_average_precision', float('nan')):>12.4f}"
            f"{ds2[name]['macro_average_precision']:>13.4f}"
            f"{ds2[name]['s_average_precision']:>10.4f}"
            f"{ds2[name]['headline_macro_f1']:>9.4f}"
            f"  {', '.join(marks)}"
        )

    lines.append("")
    lines.append(f"selection metric: {selection.get('metric', '?')}")
    if selected and best_on_ds2 and selected != best_on_ds2:
        lines.append(
            f"NOTE: out-of-fold selection chose {selected}, but {best_on_ds2} "
            f"scores higher on DS2."
        )
        lines.append(
            "      Quote the chosen one. Switching now would be selecting on "
            "the test set."
        )
    elif selected:
        lines.append(f"out-of-fold selection picked the DS2 winner: {selected}")

    if selected and reference and selected != reference:
        win, ref = ds2[selected], ds2[reference]
        lines.append("")
        lines.append(
            f"{selected} vs {reference}:  macro AP "
            f"{win['macro_average_precision'] - ref['macro_average_precision']:+.4f}   "
            f"S AP {win['s_average_precision'] - ref['s_average_precision']:+.4f}   "
            f"F1 {win['headline_macro_f1'] - ref['headline_macro_f1']:+.4f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="phase1d-results.zip from Colab (or a directory)")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument(
        "--allow-arm-mismatch",
        action="store_true",
        help="ingest even if the bundle's arms differ from train_nn.ARMS",
    )
    args = parser.parse_args(argv)

    bundle = Path(args.bundle).expanduser()
    if not bundle.exists():
        print(f"no such bundle: {bundle}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        source = _extract(bundle, Path(tmp))

        # Phase 1d writes nn_metrics.json; Phase 1e writes nn_v2_metrics.json.
        # An earlier version knew only the first name and rejected a perfectly
        # good 1e bundle.
        metrics = next(
            (source / n for n in METRICS_NAMES if (source / n).exists()), None
        )
        if metrics is None:
            print(
                f"no metrics file in the bundle (looked for "
                f"{', '.join(METRICS_NAMES)}). The Colab run either did not "
                f"finish or was stopped before scoring DS2; the *_cv.json file "
                f"alone holds the cross-validation results.",
                file=sys.stderr,
            )
            return 3
        phase, stem_label = PHASE_BY_METRICS[metrics.name]

        payload = json.loads(metrics.read_text())

        if phase == "mix":
            # Phase 1g names compositions, not arms, and the guard that matters
            # is a different one: the bundle must carry the composition list the
            # checked-in code fixed in advance. A bundle whose candidates differ
            # was selected under a different rule, and its choice cannot be read
            # as pre-registered.
            import train_ensemble_mix as orchestrator  # noqa: PLC0415

            expected = set(orchestrator.COMPOSITIONS)
            found = set(payload.get("ds2", {}).get("scores", {}))
        else:
            if phase == "nn_v2":
                import train_nn_v2 as orchestrator  # noqa: PLC0415
            else:
                import train_nn as orchestrator  # noqa: PLC0415

            expected = {a.name for a in orchestrator.ARMS}
            # Phase 1e also scores an "+ensemble" row per arm.
            expected |= {f"{n}+ensemble" for n in expected}
            found = {r.get("label") for r in payload.get("results", [])}

        if found != expected and not args.allow_arm_mismatch:
            print(
                f"arm mismatch: bundle has {sorted(found)}, code defines "
                f"{sorted(expected)}.\nThis bundle was produced by a different "
                f"version of the code. Re-run, or pass --allow-arm-mismatch "
                f"and say so in the write-up.",
                file=sys.stderr,
            )
            return 4

        results_dir = ROOT / "results"
        models_dir = ROOT / "models"
        results_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)

        stem = f"{args.date}-{stem_label}-ds1_ds2"
        shutil.copy(metrics, results_dir / f"{stem}.json")
        print(f"results/{stem}.json")

        for name in ("nn_cv.json", "nn_v2_cv.json", "nn_folds.json", "run.log"):
            # (Phase 1g adds no *_cv.json; its selection lives in the metrics file.)
            src = source / name
            if src.exists():
                suffix = "".join(Path(name).suffixes)
                shutil.copy(src, results_dir / f"{stem}-{Path(name).stem}{suffix}")
                print(f"results/{stem}-{Path(name).stem}{suffix}")

        for directory in ("checkpoints", "checkpoints_v2", "checkpoints_mix"):
            for ckpt in sorted((source / directory).glob("*.pt")):
                shutil.copy(ckpt, models_dir / ckpt.name)
                print(f"models/{ckpt.name}")

    env = payload.get("environment", {})
    print()
    print(f"device: {payload.get('device') or env.get('device')}   "
          f"torch: {env.get('torch')}")
    if phase != "mix":
        print(f"weight cap selected: {payload.get('weight_cap_selected')}")
    print()
    print(_table_mix(payload) if phase == "mix" else _table(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
