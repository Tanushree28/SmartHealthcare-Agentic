"""Figures for the calibration diagnosis.

Every function here takes already-computed arrays and returns a
``matplotlib.figure.Figure``. None of them computes a metric, and none of them
reads a file. That boundary matters: the numbers behind each figure are written
to the results JSON by the driver, so a figure can be redrawn without re-scoring
49,646 beats, and a figure can never disagree with the number it is supposed to
be showing.

``Agg`` is selected on import -- these are written to disk from a script, never
shown in a window, and the default backend on macOS tries to reach a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  -- must follow use("Agg")
import numpy as np  # noqa: E402

#: One colour per model arm, held constant across every figure so a line can be
#: followed from the PR curves to the reliability diagram without a re-read.
ARM_COLORS: dict[str, str] = {
    "plain_ce": "#4C72B0",
    "weighted_ce": "#DD8452",
    "focal": "#C44E52",
    "oversample": "#55A868",
    "waveform_only": "#8172B3",
    "ensemble": "#000000",
}

FIGURE_DPI: int = 150


def _color(arm: str) -> str:
    return ARM_COLORS.get(arm, "#7F7F7F")


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, path, dpi: int = FIGURE_DPI) -> Path:
    """Write a figure and close it, so a long run cannot exhaust the figure pool."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def pr_curves(curves_by_arm: dict, class_name: str, prevalence: float):
    """Precision-recall for one class, every arm overlaid.

    ``curves_by_arm`` maps arm name to ``(precision, recall)``. The chance line
    is drawn at the class prevalence, which is where an uninformative ranker
    sits -- without it, a curve hovering at 0.3 precision is unreadable.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    for arm, (precision, recall) in curves_by_arm.items():
        ax.plot(recall, precision, label=arm, color=_color(arm), linewidth=1.8)

    ax.axhline(
        prevalence,
        color="#999999",
        linestyle="--",
        linewidth=1.0,
        label=f"chance ({prevalence:.3f})",
    )

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision-recall, class {class_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return fig


def reliability(curves_by_arm: dict, class_name: str):
    """Reliability diagram with the bin populations underneath.

    ``curves_by_arm`` maps arm name to a ``calibration.ReliabilityCurve``. The
    lower panel is not decoration: a bin holding four beats plots at 0.0 or 1.0
    and reads as catastrophic miscalibration, and only the counts reveal it.
    """
    fig, (ax, ax_n) = plt.subplots(
        2,
        1,
        figsize=(5.6, 5.8),
        height_ratios=[3, 1],
        sharex=True,
    )

    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.0,
            label="perfectly calibrated")

    for arm, curve in curves_by_arm.items():
        filled = curve.counts > 0
        ax.plot(
            curve.mean_predicted[filled],
            curve.observed_rate[filled],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=_color(arm),
            label=arm,
        )
        centers = (curve.edges[:-1] + curve.edges[1:]) / 2
        ax_n.step(centers, curve.counts, where="mid", color=_color(arm), linewidth=1.2)

    ax.set_ylabel(f"observed rate of {class_name}")
    ax.set_title(f"Reliability, class {class_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)

    ax_n.set_yscale("log")
    ax_n.set_xlabel(f"predicted P({class_name} | beat)")
    ax_n.set_ylabel("beats/bin")
    _style(ax_n)

    fig.tight_layout()
    return fig


def confidence_histogram(proba_by_arm: dict, class_index: int, class_name: str):
    """Where each arm's probability mass for one class actually sits."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bins = np.linspace(0, 1, 41)

    for arm, proba in proba_by_arm.items():
        ax.hist(
            proba[:, class_index],
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=_color(arm),
            label=arm,
            log=True,
        )

    ax.set_xlabel(f"predicted P({class_name} | beat)")
    ax.set_ylabel("beats (log)")
    ax.set_title(f"Confidence distribution, class {class_name}")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return fig


def metric_bars(values_by_arm: dict, metric_name: str, lower_is_better: bool = False):
    """One metric across arms, sorted best-first."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    items = sorted(
        values_by_arm.items(),
        key=lambda kv: (np.inf if np.isnan(kv[1]) else kv[1]),
        reverse=not lower_is_better,
    )
    names = [k for k, _ in items]
    heights = [v for _, v in items]

    ax.bar(names, heights, color=[_color(n) for n in names], width=0.6)
    for i, v in enumerate(heights):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} ({direction})")
    ax.tick_params(axis="x", rotation=20)
    _style(ax)
    fig.tight_layout()
    return fig


def risk_coverage(curves_by_arm: dict):
    """Accuracy against fraction of beats kept, most-confident first.

    The trade a consumer that can abstain actually makes. A flat line means
    confidence carries no information and deferring the low-confidence tail
    would buy nothing.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    for arm, (coverage, accuracy) in curves_by_arm.items():
        ax.plot(coverage, accuracy, color=_color(arm), linewidth=1.8, label=arm)

    ax.set_xlabel("coverage (fraction of beats kept)")
    ax.set_ylabel("accuracy on kept beats")
    ax.set_title("Risk-coverage: what abstention would buy")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return fig


def prior_drift_bars(drift_by_arm: dict, classes: tuple[str, ...]):
    """Mean predicted probability over observed rate, per class, per arm.

    1.0 is unbiased. This is the figure that shows loss weighting distorting the
    output distribution, if it does.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    arms = list(drift_by_arm)
    width = 0.8 / max(len(arms), 1)
    x = np.arange(len(classes))

    for i, arm in enumerate(arms):
        ratios = [
            d.ratio for d in drift_by_arm[arm] if d.name in classes
        ]
        ax.bar(x + i * width - 0.4 + width / 2, ratios, width=width,
               color=_color(arm), label=arm)

    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2,
               label="unbiased (1.0)")
    ax.set_yscale("log")
    ax.set_xticks(x, classes)
    ax.set_xlabel("class")
    ax.set_ylabel("mean predicted / observed rate")
    ax.set_title("Prior drift: does the model overstate each class?")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig
