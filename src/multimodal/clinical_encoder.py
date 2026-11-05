"""Clinical input -> preprocessed vector and the clinical model's score.

Wraps the model ``scripts/mimic_iii/train.py`` saves. The input is one ICU
stay's features as the MIMIC-III pipeline defines them (the columns of
``mimic_iii.features.model_input_columns``), as a dict or a one-row table.

Validation is strict about what would make the score wrong and lenient about
what the model was built to handle:

- A post-index field (outcome, discharge, length of stay, diagnosis codes) or
  an identifier in the input is **rejected** -- it is leakage, or a sign the
  caller passed a raw database row.
- A missing feature is allowed and imputed, as in training, and is listed in
  the output so the caller can see what was filled in.
- An unknown field is ignored and listed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_iii import config as mimic_config
from mimic_iii import features as mimic_features
from mimic_iii import train as mimic_train

from .errors import InputValidationError


@dataclass
class ClinicalInput:
    features: dict
    stay_id: str | None = None

    @classmethod
    def from_feature_table_row(cls, row: pd.Series | dict) -> "ClinicalInput":
        """A row of the MIMIC-III feature table, minus identifiers and label.

        The ICU stay id is kept as a reference for the output only; it never
        reaches the model.
        """
        data = dict(row)
        stay = data.get("icustay_id")
        drop = set(mimic_config.ID_COLUMNS) | {mimic_config.LABEL_COLUMN}
        return cls(features={k: v for k, v in data.items() if k not in drop},
                   stay_id=None if stay is None else str(int(stay)))


@dataclass
class ClinicalEncoding:
    vector: np.ndarray
    names: tuple[str, ...]
    risk_score: float
    threshold: float
    missing_fields: list[str] = field(default_factory=list)
    ignored_fields: list[str] = field(default_factory=list)


class ClinicalEncoder:
    def __init__(self, model, meta: dict) -> None:
        self.model = model
        self.meta = meta
        self.columns = list(meta["input_columns"])
        self.names = tuple(meta["output_features"])

    @classmethod
    def from_dir(cls, model_dir: Path | str) -> "ClinicalEncoder":
        model, meta = mimic_train.load_model(model_dir)
        return cls(model, meta)

    @property
    def output_dim(self) -> int:
        return len(self.names)

    def _frame(self, clinical: ClinicalInput) -> tuple[pd.DataFrame, list[str], list[str]]:
        if not isinstance(clinical.features, dict) or not clinical.features:
            raise InputValidationError("clinical input must be a non-empty mapping of feature -> value")
        given = {str(k): v for k, v in clinical.features.items()}
        forbidden = sorted((set(given) & mimic_config.POST_INDEX_COLUMNS)
                           | (set(given) & set(mimic_config.ID_COLUMNS)))
        if forbidden:
            raise InputValidationError(
                f"clinical input contains fields not allowed at prediction time: {forbidden}. "
                f"Pass only features known at the index time.")
        missing = [c for c in self.columns if c not in given or _is_missing(given[c])]
        ignored = sorted(set(given) - set(self.columns))

        categorical = set(mimic_features.feature_columns()["categorical"])
        row = {}
        for c in self.columns:
            v = given.get(c)
            if c in categorical:
                row[c] = np.nan if _is_missing(v) else str(v)
            elif c in mimic_features.missing_flag_names():
                # A flag left out is derived from its count, as the pipeline does.
                count = given.get(c.removesuffix("_missing") + "_count")
                row[c] = int(v) if not _is_missing(v) else int(_is_missing(count) or count == 0)
            else:
                try:
                    row[c] = np.nan if _is_missing(v) else float(v)
                except (TypeError, ValueError) as exc:
                    raise InputValidationError(f"clinical field {c!r} must be numeric, got {v!r}") from exc
        frame = pd.DataFrame([row], columns=self.columns)
        for c in categorical:
            frame[c] = frame[c].astype(object)
        return frame, missing, ignored

    def encode(self, clinical: ClinicalInput) -> ClinicalEncoding:
        frame, missing, ignored = self._frame(clinical)
        vector = self.model.named_steps["pre"].transform(frame)[0]
        score = float(self.model.predict_proba(frame)[0, 1])
        return ClinicalEncoding(vector=np.asarray(vector, dtype=np.float64), names=self.names,
                                risk_score=score, threshold=float(self.meta["threshold"]),
                                missing_fields=missing, ignored_fields=ignored)


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return isinstance(v, float) and math.isnan(v) or v is pd.NA
    except TypeError:
        return False
