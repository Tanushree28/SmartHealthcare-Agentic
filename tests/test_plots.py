"""Smoke tests: the figures render and land on disk.

Deliberately shallow. There is no assertion here about what a figure *looks*
like -- pixel comparison is brittle and would fail on a matplotlib point
release for no reason anyone cares about. What these catch is the class of bug
that actually happens: an array shape the plotting code cannot take, a dict key
that is never looked up, a figure that is built but never closed.

The numbers behind every figure are asserted in ``test_calibration.py`` and
``test_ranking.py``. ``plots.py`` computes nothing, so there is nothing else
here to be wrong.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from ecg import calibration, config, plots, ranking


@pytest.fixture
def scored():
    """A small two-arm scoring of five AAMI classes."""
    rng = np.random.default_rng(0)
    n = 2_000
    y = rng.choice(len(config.AAMI_CLASSES), size=n, p=[0.89, 0.04, 0.06, 0.009, 0.001])

    def noisy(strength: float) -> np.ndarray:
        logits = rng.normal(size=(n, len(config.AAMI_CLASSES)))
        logits[np.arange(n), y] += strength
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    return y, {"focal": noisy(2.0), "plain_ce": noisy(1.0)}


def test_pr_curves_render(scored, tmp_path):
    y, arms = scored
    index = config.AAMI_CLASSES.index("S")
    curves = {}
    for arm, proba in arms.items():
        precision, recall, _ = ranking.pr_curve(y, proba, index)
        curves[arm] = (precision, recall)

    out = plots.save(plots.pr_curves(curves, "S", float((y == index).mean())),
                     tmp_path / "pr.png")

    assert out.exists() and out.stat().st_size > 0


def test_reliability_renders_with_empty_bins(scored, tmp_path):
    """Empty bins are the normal case for a rare class; they must not raise."""
    y, arms = scored
    index = config.AAMI_CLASSES.index("F")
    curves = {
        arm: calibration.reliability_curve(y, proba, index, 15, "F")
        for arm, proba in arms.items()
    }
    assert any(np.isnan(c.observed_rate).any() for c in curves.values())

    out = plots.save(plots.reliability(curves, "F"), tmp_path / "rel.png")
    assert out.exists() and out.stat().st_size > 0


def test_confidence_histogram_renders(scored, tmp_path):
    y, arms = scored
    out = plots.save(
        plots.confidence_histogram(arms, config.AAMI_CLASSES.index("S"), "S"),
        tmp_path / "conf.png",
    )
    assert out.exists() and out.stat().st_size > 0


def test_metric_bars_tolerate_nan(tmp_path):
    """Q's average precision is nan by design; the bar chart must survive it."""
    out = plots.save(
        plots.metric_bars({"focal": 0.42, "plain_ce": float("nan")}, "AP"),
        tmp_path / "bars.png",
    )
    assert out.exists() and out.stat().st_size > 0


def test_risk_coverage_renders(scored, tmp_path):
    y, arms = scored
    curves = {
        arm: calibration.risk_coverage_curve(y, proba) for arm, proba in arms.items()
    }
    out = plots.save(plots.risk_coverage(curves), tmp_path / "rc.png")
    assert out.exists() and out.stat().st_size > 0


def test_prior_drift_bars_render(scored, tmp_path):
    y, arms = scored
    drift = {arm: calibration.prior_drift(y, proba) for arm, proba in arms.items()}
    out = plots.save(
        plots.prior_drift_bars(drift, config.HEADLINE_CLASSES), tmp_path / "drift.png"
    )
    assert out.exists() and out.stat().st_size > 0


def test_figures_are_closed_not_leaked(scored, tmp_path):
    """plots.save closes every figure; a long run would otherwise warn and grow."""
    import matplotlib.pyplot as plt

    y, arms = scored
    before = len(plt.get_fignums())
    for i in range(12):
        plots.save(
            plots.confidence_histogram(arms, 1, "S"), tmp_path / f"f{i}.png"
        )
    assert len(plt.get_fignums()) == before


def test_backend_is_headless():
    """macOS default backend reaches for a display; a script must not."""
    assert matplotlib.get_backend().lower() == "agg"
