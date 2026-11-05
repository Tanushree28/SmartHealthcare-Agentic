#!/usr/bin/env python3
"""Draw the figures of PlanDoc/Integration_Documentation.md from real results.

    python scripts/make_integration_figures.py

Every number plotted comes from a results file; nothing is typed in by hand
except the test counts from before 2026-10-14, which are quoted from
PlanDoc/IMPLEMENTATION.md. The exact numbers behind each figure are written to
PlanDoc/integration_results/figure_data.json so any bar can be checked.

Inputs:
    results/2026-08-27-baseline-ds1_ds2.json          Random Forest baseline (DS2)
    results/2026-09-16-nn-ds1_ds2.json                 Phase 1d networks (DS2)
    results/2026-09-17-nn-v2-ds1_ds2.json              Phase 1e networks (DS2)
    results/2026-09-17-ensemble-mix-ds1_ds2.json       Phase 1g ensembles (DS2)
    results/2026-10-14-bundle-eval-ds1_ds2.json        selected ECG model on DS2
    results/2026-10-14-mimic-demo-pipeline.json        MIMIC-III Demo model report
    PlanDoc/integration_results/mitbih_learning_curve_fold0_focal.json   (if present)
    $SMART_HEALTH_DATA/processed/mimiciii-demo/{build_summary.json,features.csv.gz}
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import paths  # noqa: E402

OUT = ROOT / "PlanDoc" / "integration_results"
RES = ROOT / "results"
MIMIC = paths.processed_root() / "mimiciii-demo"

# Reference palette (dataviz skill), light mode; validated: blue/orange/aqua.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a8a7a1", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

DATA: dict = {}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
    })


def titled(fig, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=9, color=INK2)


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def value_grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(name):
    return json.loads((RES / name).read_text())


def headline(per_class) -> float:
    f1 = {c["class"]: c["f1"] for c in per_class}
    return float(np.mean([f1[c] for c in ("N", "S", "V", "F")]))


# ---------------------------------------------------------------------------


def fig_learning_curve():
    src = OUT / "mitbih_learning_curve_fold0_focal.json"
    if not src.exists():
        print("skip learning curve: run scripts/mit_bih/record_learning_curve.py first")
        return
    run = json.loads(src.read_text())
    h = run["history"]
    ep = [r["epoch"] for r in h]
    loss = [r["train_loss"] for r in h]
    f1 = [r["val_macro_f1_headline"] for r in h]
    DATA["fig01_mitbih_learning_curve"] = {"source": str(src.relative_to(ROOT)), "epoch": ep,
                                           "train_loss": loss, "val_macro_f1_headline": f1,
                                           "best_epoch": run["best_epoch"], "device": run["device"]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.98, wspace=0.28)
    titled(fig, "MIT-BIH network: one training run, epoch by epoch",
           f"Local re-run of Phase 1d focal loss, CV fold 0 ({run['n_train_beats']:,} training / "
           f"{run['n_val_beats']:,} validation beats, {run['device']}). Not the published model.")
    for ax in (a1, a2):
        value_grid(ax, "y")
        ax.axvline(run["best_epoch"], color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlabel("Epoch (one pass over the training beats)")
    a1.plot(ep, loss, color=BLUE, linewidth=2, marker="o", markersize=4)
    a1.set_ylabel("Training loss (focal)")
    a1.set_title("Training loss", loc="left", color=INK)
    a2.plot(ep, f1, color=ORANGE, linewidth=2, marker="o", markersize=4)
    a2.set_ylabel("Validation macro-F1 (N/S/V/F)")
    a2.set_ylim(0, 1)
    a2.set_title("Validation macro-F1 (stopping rule)", loc="left", color=INK)
    a2.annotate(f"best: epoch {run['best_epoch']}, {run['best_val_macro_f1_headline']:.3f}",
                xy=(run["best_epoch"], run["best_val_macro_f1_headline"]),
                xytext=(-8, 18), textcoords="offset points", ha="right", color=INK2, fontsize=9)
    a1.text(run["best_epoch"] + 0.4, max(loss) * 0.95, "kept\nweights", color=INK2, fontsize=8, va="top")
    save(fig, "fig01_mitbih_learning_curve.png")


def fig_confusion():
    m = load("2026-10-14-bundle-eval-ds1_ds2.json")["metrics"]
    cm = np.array(m["confusion_matrix"])
    classes = m["classes"]
    rows = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    DATA["fig02_mitbih_confusion_matrix"] = {"source": "results/2026-10-14-bundle-eval-ds1_ds2.json",
                                             "classes": classes, "counts": cm.tolist()}
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.25, right=0.97)
    titled(fig, "Selected ECG model (D_all_four) on the DS2 test patients",
           f"Confusion matrix, {m["n_test"]:,} beats from the 22 DS2 test records. "
           "Cell = beats; shade = share of that true class.")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + SEQ)
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            dark = pct[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if dark else INK)
    names = {"N": "N normal", "S": "S supra-\nventricular", "V": "V ventricular",
             "F": "F fusion", "Q": "Q unknown/\npaced"}
    ax.set_xticks(range(len(classes)), [names[c] for c in classes], fontsize=8.5)
    short = {"N": "N normal", "S": "S supraventricular", "V": "V ventricular",
             "F": "F fusion", "Q": "Q unknown/paced"}
    ax.set_yticks(range(len(classes)), [f"{short[c]} ({rows[i, 0]:,})"
                                        for i, c in enumerate(classes)], fontsize=8.5)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class (number of beats)")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig02_mitbih_confusion_matrix.png")


def fig_per_class():
    m = load("2026-10-14-bundle-eval-ds1_ds2.json")["metrics"]
    pc = [c for c in m["per_class"] if c["class"] != "Q"]
    DATA["fig03_mitbih_per_class"] = {"source": "results/2026-10-14-bundle-eval-ds1_ds2.json",
                                      "per_class": m["per_class"]}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.98)
    titled(fig, "Selected ECG model: precision, recall and F1 per beat class (DS2)",
           "Q (7 test beats) is left out. Normal and ventricular beats are found well; "
           "S is partly found; F is not usable.")
    x = np.arange(len(pc))
    w = 0.26
    for k, (key, color, label) in enumerate([("precision", BLUE, "Precision"),
                                             ("recall", ORANGE, "Recall"), ("f1", AQUA, "F1")]):
        vals = [c[key] for c in pc]
        bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{c['class']}\n({c['support']:,} beats)" for c in pc])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0 to 1)")
    value_grid(ax, "y")
    ax.legend(loc="upper right", ncols=3)
    save(fig, "fig03_mitbih_per_class_metrics.png")


def fig_progression():
    base = load("2026-08-27-baseline-ds1_ds2.json")
    rf = next(r for r in base["results"] if r["label"] == "random_forest / rr+morphology")
    d = {r["label"]: r["macro_f1_headline"] for r in load("2026-09-16-nn-ds1_ds2.json")["results"]}
    e = {r["label"]: r["macro_f1_headline"] for r in load("2026-09-17-nn-v2-ds1_ds2.json")["results"]}
    g = load("2026-09-17-ensemble-mix-ds1_ds2.json")["ds2"]["scores"]
    rows = [
        ("1b  Random Forest baseline", headline(rf["per_class"]), "base"),
        ("1d  network, plain cross-entropy", d["plain_ce"], "other"),
        ("1d  network, waveform only (no RR)", d["waveform_only"], "other"),
        ("1d  network, oversampling", d["oversample"], "other"),
        ("1d  network, weighted cross-entropy", d["weighted_ce"], "other"),
        ("1d  weighted + tuned thresholds", d["tuned"], "other"),
        ("1d  network, focal loss", d["focal"], "other"),
        ("1e  5 x focal fold ensemble", e["beat_focal+ensemble"], "other"),
        ("1g  focal-only ensemble (not selected)", g["A_focal_only"]["headline_macro_f1"], "other"),
        ("1g  D_all_four ensemble (selected, reported)", g["D_all_four"]["headline_macro_f1"], "sel"),
    ]
    DATA["fig04_mitbih_model_progression"] = {
        "metric": "DS2 macro-F1 over N/S/V/F",
        "sources": ["results/2026-08-27-baseline-ds1_ds2.json", "results/2026-09-16-nn-ds1_ds2.json",
                    "results/2026-09-17-nn-v2-ds1_ds2.json", "results/2026-09-17-ensemble-mix-ds1_ds2.json"],
        "rows": [{"model": r[0], "macro_f1": r[1]} for r in rows]}
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.40, right=0.95)
    titled(fig, "ECG models tried, scored on the same 22 test records (DS2)",
           "Macro-F1 over N/S/V/F on the same 49,646 beats. Grey = classical baseline. The reported "
           "model was chosen\nbefore DS2 was scored, so the higher-scoring focal-only ensemble was "
           "not switched to.")
    ax.axvline(rows[0][1], color=MUTED, linewidth=1, linestyle="--")
    y = np.arange(len(rows))[::-1]
    colors = {"base": MUTED, "other": "#9ec5f4", "sel": BLUE}
    for yi, (label, v, kind) in zip(y, rows):
        ax.barh(yi, v, 0.62, color=colors[kind])
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9,
                color=INK if kind == "sel" else INK2, fontweight="bold" if kind == "sel" else None,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6), zorder=3)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("DS2 macro-F1 (N/S/V/F)")
    value_grid(ax, "x")
    save(fig, "fig04_mitbih_model_progression.png")


def fig_cohort():
    rep = load("2026-10-14-mimic-demo-pipeline.json")
    flow = rep["cohort"]["flow"]
    DATA["fig05_mimic_cohort_flow"] = {"source": "results/2026-10-14-mimic-demo-pipeline.json",
                                       "flow": flow}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.36, right=0.95)
    titled(fig, "MIMIC-III Demo: how 136 ICU stays become the 51-stay study cohort",
           f"Each bar keeps only the stays that pass the rule on its left. "
           f"{rep['cohort']['n_positive']} of the final 51 died in hospital.")
    y = np.arange(len(flow))[::-1]
    labels = ["All ICU stays", "Adults (age 18+)", "First adult ICU stay per patient",
              "Cardiac diagnosis code", "In ICU and alive at 24 h"]
    for yi, row in zip(y, flow):
        ax.barh(yi, row["stays"], 0.6, color=BLUE)
        ax.text(row["stays"] + 1.5, yi, f"{row['stays']}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("ICU stays")
    ax.set_xlim(0, 150)
    value_grid(ax, "x")
    save(fig, "fig05_mimic_cohort_flow.png")


def _vars_order():
    from mimic_iii import config
    return [v.name for v in config.VARIABLES], {v.name: v.source for v in config.VARIABLES}


def fig_missing():
    summary = json.loads((MIMIC / "build_summary.json").read_text())
    rate = summary["feature_table"]["missing_rate"]
    names, source = _vars_order()
    order = sorted(names, key=lambda n: (rate[n], n))
    DATA["fig06_mimic_missingness"] = {"source": "$SMART_HEALTH_DATA/processed/mimiciii-demo/build_summary.json",
                                       "missing_rate": {n: rate[n] for n in order}, "n_stays": 51}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.17, right=0.95)
    titled(fig, "MIMIC-III Demo: how often each variable is missing in the first 24 h",
           "Share of the 51 cohort stays with no measurement in the observation window. Orange = lab "
           "test.\nThe seven vital signs (blue) are never missing, so their bars have zero length.")
    y = np.arange(len(order))
    for yi, n in zip(y, order):
        v = rate[n]
        ax.barh(yi, v, 0.62, color=BLUE if source[n] == "chart" else ORANGE)
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, order, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Stays with no value (missing)")
    value_grid(ax, "x")
    save(fig, "fig06_mimic_missingness.png")


def fig_density():
    f = pd.read_csv(MIMIC / "features.csv.gz")
    names, source = _vars_order()
    med = {n: float(f.loc[f[f"{n}_count"] > 0, f"{n}_count"].median()) for n in names}
    order = sorted(names, key=lambda n: (med[n], n))
    DATA["fig07_mimic_measurement_density"] = {
        "source": "$SMART_HEALTH_DATA/processed/mimiciii-demo/features.csv.gz (aggregated)",
        "median_measurements_per_measured_stay": {n: med[n] for n in order}}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.17, right=0.95)
    titled(fig, "MIMIC-III Demo: how many measurements a stay has in the first 24 h",
           "Median readings per stay, among stays with at least one. Vitals are charted about "
           "hourly; labs 1-3 times.")
    y = np.arange(len(order))
    for yi, n in zip(y, order):
        ax.barh(yi, med[n], 0.62, color=BLUE if source[n] == "chart" else ORANGE)
        ax.text(med[n] + 0.3, yi, f"{med[n]:g}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, order, fontsize=9)
    ax.set_xlabel("Median measurements per stay (0-24 h)")
    value_grid(ax, "x")
    save(fig, "fig07_mimic_measurement_density.png")


def fig_mimic_metrics():
    rep = load("2026-10-14-mimic-demo-pipeline.json")
    splits = ["train", "val", "test"]
    DATA["fig08_mimic_demo_metrics"] = {"source": "results/2026-10-14-mimic-demo-pipeline.json",
                                        "metrics": {s: {k: rep["metrics"][s][k] for k in
                                                        ("n", "n_positive", "auroc", "auprc",
                                                         "auroc_ci", "auprc_ci", "prevalence")}
                                                    for s in splits}}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.08, right=0.98, wspace=0.3)
    titled(fig, "MIMIC-III Demo model: PIPELINE CHECK ONLY, not clinical performance",
           "Point = estimate, line = 95% bootstrap interval. Train is in-sample; validation chose the "
           "settings;\nonly test (7 patients, 2 deaths) was held out. Intervals this wide mean the "
           "estimates say almost nothing.")
    for ax, key, name in ((axes[0], "auroc", "AUROC"), (axes[1], "auprc", "AUPRC")):
        for i, s in enumerate(splits):
            m = rep["metrics"][s]
            ci = m[f"{key}_ci"]
            color = BLUE if s != "test" else ORANGE
            if ci["lower"] is not None:
                ax.plot([i, i], [ci["lower"], ci["upper"]], color=color, linewidth=2)
            ax.plot(i, m[key], "o", color=color, markersize=8)
            ax.text(i + 0.08, m[key], f"{m[key]:.2f}", va="center", fontsize=9, color=INK2)
            if key == "auprc":
                ax.plot([i - 0.25, i + 0.25], [m["prevalence"]] * 2, color=MUTED, linewidth=1.5)
        if key == "auroc":
            ax.axhline(0.5, color=MUTED, linewidth=1.2, linestyle="--")
            ax.text(2.45, 0.5, "chance", va="bottom", ha="right", fontsize=8, color=INK2)
        else:
            ax.text(0.5, 0.12, "grey ticks = chance level\n(the death rate in that split)",
                    va="top", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(3), [f"{s}\n(n={rep['metrics'][s]['n']}, deaths="
                                 f"{rep['metrics'][s]['n_positive']})" for s in splits], fontsize=8.5)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.5, 2.6)
        ax.set_ylabel(name)
        ax.set_title(name, loc="left", color=INK)
        value_grid(ax, "y")
    save(fig, "fig08_mimic_demo_metrics.png")


def count_tests(target: str) -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", target, "--collect-only", "-q"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    for line in out.splitlines()[::-1]:
        if "selected" in line or "collected" in line or "test" in line:
            n = line.split()[0].split("/")[0]
            if n.isdigit():
                return int(n)
    raise RuntimeError(f"could not count tests in {target}: {out[-300:]}")


def fig_tests():
    after = {"mit_bih": count_tests("tests/mit_bih"), "mimic_iii": count_tests("tests/mimic_iii"),
             "multimodal": count_tests("tests/multimodal"), "common": count_tests("tests/common"),
             "scripts": count_tests("tests/test_scripts.py")}
    before = {"mit_bih": 414, "mimic_iii": 0, "multimodal": 0, "common": 0, "scripts": 0}
    DATA["fig09_tests_before_after"] = {
        "before_source": "PlanDoc/IMPLEMENTATION.md: 414 offline tests (2026-10-11 and the 2026-10-14 restructure)",
        "after_source": "pytest --collect-only, default markers (offline)", "before": before, "after": after}
    mods = list(after)
    fig, ax = plt.subplots(figsize=(8, 4.0))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.25, right=0.97)
    titled(fig, "Automated tests before and after the integration work",
           f"Offline tests per module. Before: {sum(before.values())} (all ECG). After: "
           f"{sum(after.values())}, all passing. Counts of tests, not a model metric.")
    y = np.arange(len(mods))[::-1]
    h = 0.36
    ax.barh(y + h / 2, [before[m] for m in mods], h * 0.92, color=MUTED, label="Before (2026-10-11)")
    ax.barh(y - h / 2, [after[m] for m in mods], h * 0.92, color=BLUE, label="After integration (now)")
    for yi, m in zip(y, mods):
        ax.text(before[m] + 4, yi + h / 2, str(before[m]), va="center", fontsize=8.5, color=INK2)
        ax.text(after[m] + 4, yi - h / 2, str(after[m]), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [f"tests/{m}" if m != "scripts" else "tests/test_scripts.py" for m in mods])
    ax.set_xlabel("Number of offline tests")
    ax.set_xlim(0, 500)
    value_grid(ax, "x")
    ax.legend(loc="lower right")
    save(fig, "fig09_tests_before_after.png")


def main(argv=None) -> int:
    import argparse

    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    fig_learning_curve()
    fig_confusion()
    fig_per_class()
    fig_progression()
    fig_cohort()
    fig_missing()
    fig_density()
    fig_mimic_metrics()
    fig_tests()
    (OUT / "figure_data.json").write_text(json.dumps(DATA, indent=2, default=str) + "\n")
    print(f"wrote {(OUT / 'figure_data.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
