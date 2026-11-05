"""Fit the clinical model: choose on validation, never on test.

Protocol: for each regularisation strength ``C`` in the grid, fit the whole
preprocessing + logistic-regression pipeline on the training split and score
the validation split by AUPRC (the positive class is the minority). The best
``C`` wins; ties go to the smaller ``C`` (stronger regularisation). The decision
threshold is then chosen on the validation split. The test split is scored
once, afterwards, by ``evaluate``.

When the validation split has only one class -- possible on the 100-patient
MIMIC-III Demo -- nothing can be chosen from it. The defaults (``C = 1``,
threshold 0.5) are used and the reason is recorded in the model metadata.

Saved models are a pickle file plus a JSON sidecar holding the
task definition, input columns, chosen settings, data source and the file's
SHA-256. ``load_model`` refuses a file whose hash does not match, because
unpickling an unverified file can run arbitrary code.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from common import metrics, provenance

from . import dataset, features, preprocessing

MODEL_FILENAME = "model.pkl"
META_FILENAME = "model_meta.json"


@dataclass(frozen=True)
class ModelSpec:
    C_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    class_weight: str | None = "balanced"
    max_iter: int = 5000
    threshold_rule: str = "youden"
    n_bootstrap: int = 1000
    default_C: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown [model] keys {sorted(unknown)}")
        kwargs = dict(raw)
        if "C_grid" in kwargs:
            kwargs["C_grid"] = tuple(float(c) for c in kwargs["C_grid"])
        return cls(**kwargs)


def select_and_fit(train_df, val_df, spec: ModelSpec = ModelSpec(), seed: int = 0):
    """Return ``(fitted_pipeline, selection_record)``."""
    X_tr, y_tr = dataset.xy(train_df)
    X_va, y_va = dataset.xy(val_df)
    if np.unique(y_tr).size < 2:
        raise ValueError(
            f"the training split has one class only ({int(y_tr.sum())} positive "
            f"of {y_tr.size}); no classifier can be fitted"
        )
    val_scorable = np.unique(y_va).size == 2

    candidates = []
    fitted = {}
    for C in sorted(spec.C_grid):
        model = preprocessing.build_model(C=C, class_weight=spec.class_weight,
                                          max_iter=spec.max_iter, seed=seed)
        model.fit(X_tr, y_tr)
        fitted[C] = model
        entry = {"C": C}
        if val_scorable:
            entry["val_auprc"] = float(average_precision_score(
                y_va, model.predict_proba(X_va)[:, 1]))
        candidates.append(entry)

    if val_scorable:
        best = candidates[0]
        for entry in candidates[1:]:
            if entry["val_auprc"] > best["val_auprc"]:
                best = entry
        chosen_C = best["C"]
        note = "C chosen by validation AUPRC; ties to the smaller C"
    else:
        chosen_C = spec.default_C if spec.default_C in fitted else sorted(fitted)[len(fitted) // 2]
        note = (f"validation split has one class ({int(y_va.sum())} positive of "
                f"{y_va.size}); C not selectable, default {chosen_C} used")

    model = fitted[chosen_C]
    threshold, threshold_note = metrics.choose_threshold(
        y_va, model.predict_proba(X_va)[:, 1], rule=spec.threshold_rule)

    return model, {
        "candidates": candidates,
        "selected_C": chosen_C,
        "selection_note": note,
        "threshold": threshold,
        "threshold_rule": spec.threshold_rule,
        "threshold_note": threshold_note,
        "n_train": int(y_tr.size),
        "n_train_positive": int(y_tr.sum()),
        "n_val": int(y_va.size),
        "n_val_positive": int(y_va.sum()),
        "model_spec": asdict(spec),
    }


def save_model(out_dir: Path | str, model, meta: dict) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILENAME
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        **meta,
        "input_columns": features.model_input_columns(),
        "output_features": preprocessing.output_feature_names(model),
        "model_file": MODEL_FILENAME,
        "model_sha256": provenance.sha256_file(model_path),
    }
    meta_path = provenance.write_json(out_dir / META_FILENAME, meta)
    return {"model": model_path, "meta": meta_path}


def load_model(out_dir: Path | str):
    """Load a saved model after checking its hash and its input columns."""
    import json

    out_dir = Path(out_dir)
    meta_path = out_dir / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no clinical model at {out_dir}; run scripts/mimic_iii/train.py first")
    meta = json.loads(meta_path.read_text())
    model_path = out_dir / meta["model_file"]
    actual = provenance.sha256_file(model_path)
    if actual != meta["model_sha256"]:
        raise ValueError(
            f"{model_path} does not match the hash in {meta_path.name}; refusing "
            f"to unpickle a file that changed after it was written")
    if meta["input_columns"] != features.model_input_columns():
        raise ValueError(
            "the saved model expects different input columns than this code "
            "version produces; retrain it with scripts/mimic_iii/train.py")
    with open(model_path, "rb") as handle:
        return pickle.load(handle), meta
