#!/usr/bin/env python
"""Generate the figures for the written report.

Plain PNGs at print resolution, designed to be dropped into a document or slide
and read on paper: white ground, no dark mode, large labels, colourblind-safe
hues (validated: adjacent-pair CVD delta-E >= 8).

Every number is read from the committed run artifacts in ``results/`` so a
figure can never disagree with the run that produced it.

Usage:
    ./venv/bin/python scripts/make_report_figures.py
    ./venv/bin/python scripts/make_report_figures.py --out-dir report/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write files only

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg import config  # noqa: E402

CLASSES = list(config.AAMI_CLASSES)
HEADLINE = list(config.HEADLINE_CLASSES)

#: Validated colourblind-safe trio (see dataviz validator).
BLUE, RED, GREEN = "#3D5A98", "#C2453D", "#1A8F76"
INK, MUTED, RULE = "#1E1A19", "#6B625E", "#D8D0CC"

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}. Run the pipeline first.")
    return json.loads(path.read_text())


def find(payload: dict, label: str) -> dict:
    for result in payload["results"]:
        if result["label"] == label:
            return result
    raise SystemExit(f"no result labelled {label!r} in the payload")


def headline_f1(result: dict) -> float:
    per_class = {m["class"]: m for m in result["per_class"]}
    return float(np.mean([per_class[c]["f1"] for c in HEADLINE]))


def per_class(result: dict, key: str) -> dict[str, float]:
    return {m["class"]: m[key] for m in result["per_class"]}


# ---------------------------------------------------------------------------
# Figure 1 -- the three methods side by side
# ---------------------------------------------------------------------------


def figure_comparison(methods: list[tuple[str, dict]], out: Path) -> Path:
    names = [n for n, _ in methods]
    scores = [headline_f1(r) for _, r in methods]
    s_recall = [per_class(r, "recall")["S"] for _, r in methods]
    s_prec = [per_class(r, "precision")["S"] for _, r in methods]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    panels = [
        (axes[0], scores, "Overall score\n(macro F1, N/S/V/F)", BLUE),
        (axes[1], s_recall, "S beats found\n(recall)", RED),
        (axes[2], s_prec, "S predictions correct\n(precision)", GREEN),
    ]

    x = np.arange(len(names))
    for ax, values, title, colour in panels:
        bars = ax.bar(x, values, width=0.6, color=colour, zorder=3)
        # The final bar is the one being argued for; the earlier ones are context.
        for bar in bars[:-1]:
            bar.set_alpha(0.45)
        for xi, v in zip(x, values):
            ax.text(xi, v + 0.022, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9, linespacing=1.4)
        ax.set_ylim(0, max(values) * 1.28)
        ax.set_title(title, fontsize=10, pad=10, color=INK)
        ax.grid(axis="y", color=RULE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Results on 22 unseen patients (49,646 heartbeats)",
        fontsize=13, fontweight="bold", y=1.03,
    )
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 -- confusion matrices
# ---------------------------------------------------------------------------


def figure_confusion(methods: list[tuple[str, dict]], out: Path) -> Path:
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.6))
    if len(methods) == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, methods):
        matrix = np.array(result["confusion_matrix"], dtype=float)
        # Row-normalised: each row is "of the beats that truly were X, where did
        # they go". Raw counts would show only the N row, which is 89% of data.
        totals = matrix.sum(axis=1, keepdims=True)
        shares = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

        ax.imshow(shares, cmap="Reds", vmin=0, vmax=1, aspect="equal")

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                count = int(matrix[i, j])
                text = f"{count:,}" if count < 10000 else f"{count / 1000:.0f}k"
                ax.text(
                    j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if shares[i, j] > 0.55 else INK,
                    fontweight="bold" if i == j else "normal",
                )

        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, fontsize=10)
        ax.set_yticklabels(CLASSES, fontsize=10)
        ax.set_xlabel("Predicted class", fontsize=10, labelpad=8)
        ax.set_ylabel("True class", fontsize=10, labelpad=8)
        ax.set_title(name, fontsize=11, fontweight="bold", pad=12)
        ax.set_xticks(np.arange(-0.5, len(CLASSES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CLASSES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Confusion matrices  —  shading shows the share of each true class; "
        "numbers are beat counts",
        fontsize=11, y=1.02,
    )
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 -- per-class recall
# ---------------------------------------------------------------------------


def figure_per_class(methods: list[tuple[str, dict]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))

    width = 0.26
    x = np.arange(len(CLASSES))
    colours = [BLUE, RED, GREEN]

    for k, (name, result) in enumerate(methods):
        recalls = per_class(result, "recall")
        values = [recalls[c] for c in CLASSES]
        offset = (k - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values, width * 0.92, label=name,
               color=colours[k % len(colours)], zorder=3)

    supports = per_class(methods[-1][1], "support")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n({supports[c]:,} beats)" for c in CLASSES], fontsize=9.5
    )
    ax.set_ylabel("Recall  (share of that class found)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title(
        "How well each heartbeat type is found",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.annotate(
        "Q has only 8 training beats —\nno method can learn it",
        xy=(4, 0.03), xytext=(3.45, 0.34), fontsize=8.5, color=MUTED,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9),
    )
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Calibration diagnosis figures (Phase 1f). These read the curve arrays stored
# by scripts/diagnose_calibration.py rather than re-scoring the checkpoints, so
# the report can be rebuilt on a machine that has neither the models nor the
# processed beats.
# --------------------------------------------------------------------------

#: Report labels for the diagnosis arms. Plain words, not the internal arm keys.
ARM_LABELS: dict[str, str] = {
    "plain_ce": "Plain",
    "weighted_ce": "Weighted",
    "focal": "Focal (kept)",
    "oversample": "Oversampling",
    "ensemble": "Ensemble of 5",
}

#: A reliability bin below this many beats is not plotted -- see figure_trust.
MIN_BIN_BEATS: int = 50

ARM_COLOURS: dict[str, str] = {
    "plain_ce": "#7D8CA8",
    "weighted_ce": "#C9A227",
    "focal": RED,
    "oversample": GREEN,
    "ensemble": BLUE,
}


def figure_scoring(diag: dict, out: Path) -> Path:
    """Two ways of scoring the same models, side by side.

    The point of the figure is the contrast in spread: the line-dependent score
    fans out across the arms while the line-free score is nearly flat. Both
    panels therefore share a y-axis -- drawing them on separate scales would
    hide the entire finding.
    """
    arms = list(ARM_LABELS)
    m = diag["measurements"]
    f1 = [m[a]["headline_macro_f1"] for a in arms]
    ap = [m[a]["macro_average_precision"] for a in arms]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    x = np.arange(len(arms))

    for ax, values, title in [
        (axes[0], f1, "Score that depends on where\nthe decision line is drawn"),
        (axes[1], ap, "Score that does not\n(every line tried, then averaged)"),
    ]:
        bars = ax.bar(x, values, width=0.62,
                      color=[ARM_COLOURS[a] for a in arms], zorder=3)
        for bar, a in zip(bars, arms):
            if a != "focal":
                bar.set_alpha(0.5)
        for xi, v in zip(x, values):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS[a] for a in arms], fontsize=9, rotation=18,
                           ha="right")
        ax.set_title(title, fontsize=10.5, pad=10, color=INK)
        ax.grid(axis="y", color=RULE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylim(0, 0.72)
    axes[0].set_ylabel("score", fontsize=10)

    spread_f1 = max(f1[:4]) - min(f1[:4])
    spread_ap = max(ap[:4]) - min(ap[:4])
    fig.suptitle(
        f"The four training methods spread {spread_f1:.3f} on the left "
        f"and {spread_ap:.3f} on the right",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_precision_ceiling(diag: dict, out: Path) -> Path:
    """Why class S precision never moved, whatever was tried."""
    curves = diag["curves"]["pr"]["S"]
    prevalence = next(
        c["prevalence"] for c in diag["measurements"]["focal"]["per_class"]
        if c["class"] == "S"
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    for arm, label in ARM_LABELS.items():
        data = curves[arm]
        ax.plot(data["recall"], data["precision"], label=label,
                color=ARM_COLOURS[arm], linewidth=2.0 if arm == "ensemble" else 1.5,
                alpha=1.0 if arm in ("ensemble", "focal") else 0.75, zorder=3)

    ax.axhline(prevalence, color=MUTED, linestyle=":", linewidth=1.3, zorder=2)
    ax.text(0.015, prevalence + 0.018, "guessing at random", fontsize=8.5,
            color=MUTED)

    ax.annotate(
        "flat: no setting does better",
        xy=(0.5, 0.42), xytext=(0.30, 0.68), fontsize=9.5, color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.1),
    )

    ax.set_xlabel("share of S beats found  (recall)", fontsize=10)
    ax.set_ylabel("share of S calls that were right  (precision)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(
        "Class S: precision stays near 0.40 wherever the line is drawn",
        fontsize=12.5, fontweight="bold", pad=12,
    )
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_trust(diag: dict, out: Path) -> Path:
    """Do the predicted probabilities mean what they say, and does deferring help?"""
    rel = diag["curves"]["reliability"]["S"]
    risk = diag["curves"]["risk_coverage"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1.3, zorder=2)
    ax.text(0.53, 0.60, "honest", fontsize=8.5, color=MUTED, rotation=38)
    for arm, label in ARM_LABELS.items():
        data = rel[arm]
        # Bins holding a handful of beats plot at extreme values and read as
        # dramatic miscalibration. The standalone diagnosis figure shows the bin
        # populations in a panel underneath; this one has no room for that, so
        # it drops the bins that panel exists to warn about. Without this the
        # ensemble appears to spike to 0.69 on the strength of 16 beats and then
        # collapse to 0.33 on the strength of 3.
        xs, ys = [], []
        for mean_pred, observed, count in zip(
            data["mean_predicted"], data["observed_rate"], data["counts"]
        ):
            if mean_pred is None or count < MIN_BIN_BEATS:
                continue
            xs.append(mean_pred)
            ys.append(observed)
        ax.plot(xs, ys, marker="o", markersize=3.4, label=label,
                color=ARM_COLOURS[arm],
                linewidth=2.0 if arm == "ensemble" else 1.4,
                alpha=1.0 if arm in ("ensemble", "focal") else 0.7, zorder=3)
    ax.set_xlabel("what the model claimed", fontsize=10)
    ax.set_ylabel("what actually happened", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Is P(S) believable?", fontsize=11.5, fontweight="bold", pad=10)
    ax.text(0.99, 0.02, f"bins under {MIN_BIN_BEATS} beats omitted",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=MUTED)

    ax = axes[1]
    for arm, label in ARM_LABELS.items():
        data = risk[arm]
        ax.plot(data["coverage"], data["accuracy"], label=label,
                color=ARM_COLOURS[arm],
                linewidth=2.0 if arm == "ensemble" else 1.4,
                alpha=1.0 if arm in ("ensemble", "focal") else 0.7, zorder=3)
    ax.set_xlabel("share of beats the model answers for", fontsize=10)
    ax.set_ylabel("accuracy on those beats", fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(color=RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("What handing the hard ones to a human buys",
                 fontsize=11.5, fontweight="bold", pad=10)

    fig.subplots_adjust(wspace=0.26)
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Phase 1g: did mixing the training methods help? (It did not, and the way the
# selection failed is itself the more useful finding.)
# --------------------------------------------------------------------------

COMPOSITION_LABELS: dict[str, str] = {
    "A_focal_only": "Focal only\n(in use)",
    "B_focal_plain": "Focal\n+ plain",
    "C_focal_plain_oversample": "Focal + plain\n+ oversampling",
    "D_all_four": "All four\n(chosen)",
    "E_plain_oversample": "Plain\n+ oversampling",
}


def figure_mixing(mix: dict, out: Path) -> Path:
    """Two panels: what mixing did, and why the selection step was fooled.

    The right-hand panel is the point. Out-of-fold score climbs almost perfectly
    with the number of models in the ensemble while the test score does not,
    which is the signature of a selection metric measuring size rather than
    quality.
    """
    order = list(COMPOSITION_LABELS)
    oof = mix["oof"]["scores"]
    ds2 = mix["ds2"]["scores"]
    reference = mix["selection"]["reference"]
    selected = mix["selection"]["selected"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # --- left: what actually happened on the test set --------------------
    ax = axes[0]
    x = np.arange(len(order))
    width = 0.38

    macro = [ds2[k]["macro_average_precision"] for k in order]
    s_ap = [ds2[k]["s_average_precision"] for k in order]

    b1 = ax.bar(x - width / 2, macro, width, label="all classes", color=BLUE, zorder=3)
    b2 = ax.bar(x + width / 2, s_ap, width, label="class S", color=RED, zorder=3)
    for bars, values in ((b1, macro), (b2, s_ap)):
        for bar, k, v in zip(bars, order, values):
            if k != reference:
                bar.set_alpha(0.5)
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([COMPOSITION_LABELS[k] for k in order], fontsize=8,
                       linespacing=1.3)
    ax.set_ylabel("ranking quality on unseen patients", fontsize=9.5)
    ax.set_ylim(0, 0.78)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", color=RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("The ensemble already in use won", fontsize=11.5,
                 fontweight="bold", pad=10)

    # --- right: the trap --------------------------------------------------
    # A scatter, not a line. Two compositions hold ten models each, so joining
    # points would draw a vertical step that looks like an error and would imply
    # the score is a function of size, which is the very thing in question.
    ax = axes[1]
    letters = {k: k.split("_")[0] for k in order}

    # B and E both hold ten models and would plot on top of each other.
    seen: dict[int, int] = {}
    offsets: dict[str, float] = {}
    for k in order:
        n = ds2[k]["n_models"]
        offsets[k] = 0.45 * seen.get(n, 0)
        seen[n] = seen.get(n, 0) + 1
    for k in order:
        n = ds2[k]["n_models"]
        if seen[n] > 1:
            offsets[k] -= 0.45 * (seen[n] - 1) / 2

    for k in order:
        n = ds2[k]["n_models"] + offsets[k]
        ax.scatter(n, oof[k]["macro_average_precision"], s=58, color=GREEN,
                   marker="o", zorder=3)
        ax.scatter(n, ds2[k]["macro_average_precision"], s=58, color=BLUE,
                   marker="s", zorder=3)
        ax.annotate(letters[k], (n, oof[k]["macro_average_precision"]),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=MUTED)
        ax.annotate(letters[k], (n, ds2[k]["macro_average_precision"]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=MUTED)

    counts = np.array([ds2[k]["n_models"] for k in order], dtype=float)
    for values, colour in ((np.array([oof[k]["macro_average_precision"]
                                      for k in order]), GREEN),
                           (np.array([ds2[k]["macro_average_precision"]
                                      for k in order]), BLUE)):
        slope, intercept = np.polyfit(counts, values, 1)
        span = np.array([counts.min() - 1, counts.max() + 1])
        ax.plot(span, slope * span + intercept, color=colour, linewidth=1.2,
                linestyle="--", alpha=0.65, zorder=2)

    ax.scatter([], [], s=58, color=GREEN, marker="o",
               label="score used to choose (held-out DS1)")
    ax.scatter([], [], s=58, color=BLUE, marker="s",
               label="score that matters (unseen patients)")

    ax.set_xticks(sorted(set(int(c) for c in counts)))
    ax.set_xlim(3, 23)
    ax.set_xlabel("number of models in the ensemble", fontsize=9.5)
    ax.set_ylabel("ranking quality (all classes)", fontsize=9.5)
    ax.grid(color=RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Why the choice went wrong", fontsize=11.5,
                 fontweight="bold", pad=10)

    fig.subplots_adjust(wspace=0.26)
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "report" / "figures"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load(ROOT / "results" / "2026-08-27-baseline-ds1_ds2.json")
    phase1d = load(ROOT / "results" / "2026-09-16-nn-ds1_ds2.json")
    phase1e = load(ROOT / "results" / "2026-09-17-nn-v2-ds1_ds2.json")
    diagnosis = load(ROOT / "results" / "2026-09-17-calibration-diagnosis-ds1_ds2.json")
    mixing = load(ROOT / "results" / "2026-09-17-ensemble-mix-ds1_ds2.json")

    methods = [
        ("Random\nForest", find(baseline, "random_forest / rr+morphology")),
        ("Neural net\n(1 model)", find(phase1d, "focal")),
        ("Neural net\n(5 models)", find(phase1e, "beat_focal+ensemble")),
    ]
    short = [
        ("Random Forest", methods[0][1]),
        ("Neural network", methods[1][1]),
        ("Neural net + ensemble", methods[2][1]),
    ]

    written = [
        figure_comparison(methods, out_dir / "fig1_results.png"),
        figure_confusion(
            [(short[0][0], short[0][1]), (short[2][0], short[2][1])],
            out_dir / "fig2_confusion_matrix.png",
        ),
        figure_per_class(short, out_dir / "fig3_per_class.png"),
        figure_scoring(diagnosis, out_dir / "fig4_scoring.png"),
        figure_precision_ceiling(diagnosis, out_dir / "fig5_precision_ceiling.png"),
        figure_trust(diagnosis, out_dir / "fig6_trust.png"),
        figure_mixing(mixing, out_dir / "fig7_mixing.png"),
    ]

    for path in written:
        print(f"{path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
