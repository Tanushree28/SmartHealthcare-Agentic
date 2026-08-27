"""Metrics for beat classification.

**Accuracy is not the headline here and never should be.** Normal beats are
about 86% of the database, so a model that predicts "normal" for everything
scores 86% while being clinically useless -- it would miss every arrhythmia,
which is the only thing anyone cares about. Per-class recall is what matters:
of the ventricular beats that exist, how many did we find?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from . import config


@dataclass
class ClassMetrics:
    """Per-class performance."""

    name: str
    support: int
    recall: float  # sensitivity: of the beats of this class, how many found
    precision: float  # of the beats called this class, how many were right
    f1: float


@dataclass
class Evaluation:
    """Full evaluation of one model on one test set."""

    classes: tuple[str, ...]
    matrix: np.ndarray  # (n_classes, n_classes) rows = true, cols = predicted
    per_class: tuple[ClassMetrics, ...]
    accuracy: float
    macro_recall: float
    n_test: int
    label: str = ""

    def reconciles(self) -> bool:
        """The matrix must account for every test beat exactly once."""
        return int(self.matrix.sum()) == self.n_test

    def recall_of(self, name: str) -> float:
        for metrics in self.per_class:
            if metrics.name == name:
                return metrics.recall
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n_test": self.n_test,
            "classes": list(self.classes),
            "confusion_matrix": self.matrix.tolist(),
            "confusion_matrix_note": "rows = true class, columns = predicted class",
            "per_class": [
                {
                    "class": m.name,
                    "support": m.support,
                    "recall": round(m.recall, 4),
                    "precision": round(m.precision, 4),
                    "f1": round(m.f1, 4),
                }
                for m in self.per_class
            ],
            "macro_recall": round(self.macro_recall, 4),
            "accuracy": round(self.accuracy, 4),
            "accuracy_note": (
                "Uninformative on this dataset: normal beats are ~86% of the "
                "data, so predicting 'normal' for everything scores ~0.86 while "
                "finding no arrhythmias at all. Read per-class recall instead."
            ),
        }


def evaluate(
    y_true,
    y_pred,
    classes: tuple[str, ...] = config.AAMI_CLASSES,
    label: str = "",
) -> Evaluation:
    """Compute per-class metrics and the confusion matrix."""
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)

    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot evaluate an empty test set")

    indices = list(range(len(classes)))
    matrix = confusion_matrix(true, pred, labels=indices)

    per_class: list[ClassMetrics] = []
    for i, name in enumerate(classes):
        support = int(matrix[i].sum())
        tp = int(matrix[i, i])
        predicted_as = int(matrix[:, i].sum())

        # A class absent from the test set is legitimate -- F and Q are very
        # rare. Report zero rather than dividing by zero or crashing.
        recall = tp / support if support else 0.0
        precision = tp / predicted_as if predicted_as else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class.append(ClassMetrics(name, support, recall, precision, f1))

    present = [m for m in per_class if m.support > 0]

    return Evaluation(
        classes=tuple(classes),
        matrix=matrix,
        per_class=tuple(per_class),
        accuracy=float((true == pred).mean()),
        macro_recall=float(np.mean([m.recall for m in present])) if present else 0.0,
        n_test=int(true.size),
        label=label,
    )


def format_report(result: Evaluation) -> str:
    """Human-readable report, per-class recall first."""
    lines: list[str] = []
    if result.label:
        lines.append(result.label)
        lines.append("=" * len(result.label))

    lines.append(f"{'class':<8}{'support':>10}{'recall':>10}{'precision':>12}{'F1':>8}")
    for m in result.per_class:
        note = "   (absent from test set)" if m.support == 0 else ""
        lines.append(
            f"{m.name:<8}{m.support:>10,}{m.recall:>10.3f}{m.precision:>12.3f}"
            f"{m.f1:>8.3f}{note}"
        )

    lines.append("")
    lines.append(f"macro-average recall : {result.macro_recall:.3f}   <- the number that matters")
    lines.append(f"accuracy             : {result.accuracy:.3f}   <- uninformative, ~86% of beats are normal")

    lines.append("")
    lines.append("confusion matrix (rows = true, columns = predicted)")
    header = "".join(f"{c:>9}" for c in result.classes)
    lines.append(f"{'':<6}{header}")
    for i, name in enumerate(result.classes):
        row = "".join(f"{int(v):>9,}" for v in result.matrix[i])
        lines.append(f"{name:<6}{row}")

    return "\n".join(lines)


def write_json(results, path: Path, extra: dict | None = None) -> Path:
    """Persist one or more evaluations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [r.to_dict() for r in results],
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
