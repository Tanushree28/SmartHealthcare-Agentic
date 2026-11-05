"""Binary-classification metrics with explicit handling of undefined values.

Used by the MIMIC-III mortality model and by the fusion module. Small test sets
are the norm here (the MIMIC-III Demo yields about eight test patients), so the
functions never invent a number: a metric that is undefined on the data it was
given -- AUROC with one class present, PPV with no positive predictions -- is
returned as ``None`` together with the reason.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def _check(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got shapes {y.shape} and {s.shape}")
    if y.size != s.size:
        raise ValueError(f"length mismatch: {y.size} labels vs {s.size} scores")
    if y.size == 0:
        raise ValueError("cannot score an empty set")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError(f"labels must be 0/1, got {sorted(set(y.tolist()))}")
    if not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > 1):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return y.astype(np.int64), s


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def binary_report(y_true, score, threshold: float = 0.5) -> dict:
    """Threshold-free and thresholded metrics, with reasons for any undefined.

    ``score`` is the predicted probability of the positive class. A case is
    predicted positive when ``score >= threshold``.
    """
    y, s = _check(y_true, score)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    n = int(y.size)
    n_pos = int(y.sum())
    undefined: dict[str, str] = {}

    both = 0 < n_pos < n
    if both:
        auroc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))
    else:
        auroc = auprc = None
        reason = f"only one class present ({n_pos} positive of {n})"
        undefined["auroc"] = undefined["auprc"] = reason

    pred = (s >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    ppv = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    for name, value, why in (
        ("sensitivity", sensitivity, "no positive cases"),
        ("specificity", specificity, "no negative cases"),
        ("ppv", ppv, "no positive predictions"),
        ("npv", npv, "no negative predictions"),
    ):
        if value is None:
            undefined[name] = why

    if sensitivity is None or ppv is None or (sensitivity + ppv) == 0:
        f1 = None
        undefined.setdefault("f1", "precision or recall undefined or both zero")
    else:
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity)

    return {
        "n": n,
        "n_positive": n_pos,
        "prevalence": n_pos / n,
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(brier_score_loss(y, s)),
        "log_loss": float(log_loss(y, np.clip(s, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "mean_predicted": float(s.mean()),
        "threshold": float(threshold),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "undefined": undefined,
    }


def bootstrap_ci(
    y_true,
    score,
    metric: str = "auroc",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap interval for AUROC or AUPRC.

    Resamples that contain only one class cannot be scored and are skipped;
    their count is reported so an interval resting on few valid resamples is
    visible. Returns ``None`` bounds when fewer than 100 resamples are valid.
    """
    if metric not in ("auroc", "auprc"):
        raise ValueError(f"metric must be 'auroc' or 'auprc', got {metric!r}")
    y, s = _check(y_true, score)
    fn = roc_auc_score if metric == "auroc" else average_precision_score

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        yb = y[idx]
        if 0 < yb.sum() < yb.size:
            values.append(float(fn(yb, s[idx])))

    result = {"metric": metric, "n_boot": n_boot, "n_valid": len(values),
              "alpha": alpha, "lower": None, "upper": None}
    if len(values) >= 100:
        result["lower"] = float(np.quantile(values, alpha / 2))
        result["upper"] = float(np.quantile(values, 1 - alpha / 2))
    else:
        result["reason"] = (
            f"only {len(values)} of {n_boot} resamples contained both classes"
        )
    return result


def reliability_bins(y_true, score, n_bins: int = 5) -> list[dict]:
    """Observed event rate against mean prediction, in equal-width bins."""
    y, s = _check(y_true, score)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed last bin so a score of exactly 1.0 is counted.
    which = np.clip(np.digitize(s, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = which == b
        count = int(mask.sum())
        out.append({
            "lower": float(edges[b]),
            "upper": float(edges[b + 1]),
            "n": count,
            "mean_predicted": float(s[mask].mean()) if count else None,
            "observed_rate": float(y[mask].mean()) if count else None,
        })
    return out


def choose_threshold(y_true, score, rule: str = "youden") -> tuple[float, str]:
    """Pick a decision threshold on validation data.

    ``youden`` maximises sensitivity + specificity - 1; ``max_f1`` maximises F1.
    Returns ``(threshold, note)``. With one class present no threshold can be
    chosen from the data, so 0.5 is returned and the note says why.
    """
    y, s = _check(y_true, score)
    if not 0 < y.sum() < y.size:
        return 0.5, "default 0.5: validation set has one class only"

    if rule == "youden":
        fpr, tpr, thr = roc_curve(y, s)
        best = int(np.argmax(tpr - fpr))
        value = float(thr[best])
        if not math.isfinite(value):  # roc_curve's leading +inf sentinel
            value = 1.0
        return min(value, 1.0), "youden on validation"
    if rule == "max_f1":
        best_t, best_f1 = 0.5, -1.0
        for t in np.unique(s):
            pred = s >= t
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        return best_t, "max F1 on validation"
    raise ValueError(f"unknown threshold rule {rule!r}")
