"""Patient-level train/validation/test splits and model matrices.

The unit of splitting is the patient (``subject_id``), never the row. A patient
with two ICU stays in the training set and a third in the test set would let
the model recognise the patient rather than learn the outcome -- the same
leakage the MIT-BIH pipeline prevents by splitting on records.

Splits are stratified on the patient-level label (1 if any of the patient's
stays is positive) so a small cohort does not end up with a test set of one
class. Assignment is deterministic for a given seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features
from .errors import LeakageError

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class PatientSplit:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    seed: int
    fractions: tuple[float, float, float]
    #: Hash of the feature table the split was made from. ``train.py`` refuses
    #: to run when it no longer matches -- the MIMIC version of the MIT-BIH
    #: normalization tripwire.
    feature_table_sha256: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.assert_disjoint()

    def assert_disjoint(self) -> None:
        tr, va, te = set(self.train), set(self.val), set(self.test)
        overlap = (tr & va) | (tr & te) | (va & te)
        if overlap:
            raise LeakageError(f"subjects in more than one split: {sorted(overlap)[:10]}")

    def subjects(self, name: str) -> tuple[int, ...]:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
        return getattr(self, name)

    def to_dict(self) -> dict:
        return {
            "unit": "subject_id",
            "seed": self.seed,
            "fractions": list(self.fractions),
            "feature_table_sha256": self.feature_table_sha256,
            "counts": {n: len(self.subjects(n)) for n in SPLIT_NAMES},
            **{n: sorted(int(s) for s in self.subjects(n)) for n in SPLIT_NAMES},
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PatientSplit":
        if payload.get("unit") != "subject_id":
            raise LeakageError(f"split unit must be subject_id, got {payload.get('unit')!r}")
        return cls(
            train=tuple(int(s) for s in payload["train"]),
            val=tuple(int(s) for s in payload["val"]),
            test=tuple(int(s) for s in payload["test"]),
            seed=int(payload["seed"]),
            fractions=tuple(payload["fractions"]),
            feature_table_sha256=payload.get("feature_table_sha256"),
            extra=payload.get("extra", {}),
        )


def _allocate(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split ``n`` items into three counts; train gets the remainder."""
    n_val = int(round(n * fractions[1]))
    n_test = int(round(n * fractions[2]))
    while n_val + n_test >= n and n > 0 and (n_val or n_test):
        # Keep at least one training subject in every stratum.
        if n_val >= n_test and n_val:
            n_val -= 1
        else:
            n_test -= 1
    return n - n_val - n_test, n_val, n_test


def patient_split(
    table: pd.DataFrame,
    fractions: tuple[float, float, float] = config.TaskSpec().split_fractions,
    seed: int = config.TaskSpec().split_seed,
    feature_table_sha256: str | None = None,
) -> PatientSplit:
    """Assign every subject in ``table`` to exactly one of train/val/test."""
    for col in ("subject_id", config.LABEL_COLUMN):
        if col not in table:
            raise ValueError(f"table has no {col!r} column")
    if table["subject_id"].isna().any():
        raise ValueError("subject_id has missing values")

    patient_label = table.groupby("subject_id")[config.LABEL_COLUMN].max()
    rng = np.random.default_rng(seed)
    parts: dict[str, list[int]] = {n: [] for n in SPLIT_NAMES}
    for label in sorted(patient_label.unique()):
        subjects = np.sort(patient_label.index[patient_label == label].to_numpy())
        subjects = subjects[rng.permutation(subjects.size)]
        n_tr, n_va, _ = _allocate(subjects.size, fractions)
        parts["train"] += subjects[:n_tr].tolist()
        parts["val"] += subjects[n_tr:n_tr + n_va].tolist()
        parts["test"] += subjects[n_tr + n_va:].tolist()

    empty = [n for n in SPLIT_NAMES if not parts[n]]
    if empty:
        raise ValueError(
            f"{patient_label.size} subjects are too few for a three-way split; "
            f"empty: {empty}"
        )
    return PatientSplit(
        train=tuple(sorted(parts["train"])),
        val=tuple(sorted(parts["val"])),
        test=tuple(sorted(parts["test"])),
        seed=seed,
        fractions=tuple(fractions),
        feature_table_sha256=feature_table_sha256,
    )


def split_frame(table: pd.DataFrame, split: PatientSplit) -> dict[str, pd.DataFrame]:
    """Rows of ``table`` per split. Every subject must be assigned exactly once."""
    assignment: dict[int, str] = {}
    for name in SPLIT_NAMES:
        for s in split.subjects(name):
            assignment[int(s)] = name
    present = set(int(s) for s in table["subject_id"].unique())
    unassigned = present - set(assignment)
    if unassigned:
        raise LeakageError(
            f"{len(unassigned)} subjects in the table have no split assignment "
            f"(the table changed after the split was made), e.g. {sorted(unassigned)[:5]}"
        )
    which = table["subject_id"].map(lambda s: assignment[int(s)])
    return {name: table[which == name].reset_index(drop=True) for name in SPLIT_NAMES}


def xy(table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Model inputs and labels. Identifiers and the label never reach X."""
    columns = features.model_input_columns()
    features.assert_no_leakage(columns)
    missing = [c for c in columns if c not in table]
    if missing:
        raise ValueError(f"table is missing model input columns {missing[:5]}")
    return table[columns].copy(), table[config.LABEL_COLUMN].to_numpy(dtype=np.int64)


def write_split(path: Path | str, split: PatientSplit, table: pd.DataFrame | None = None) -> Path:
    payload = split.to_dict()
    if table is not None:
        parts = split_frame(table, split)
        payload["label_counts"] = {
            n: {"stays": int(len(df)), "positive": int(df[config.LABEL_COLUMN].sum())}
            for n, df in parts.items()
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_split(path: Path | str) -> PatientSplit:
    return PatientSplit.from_dict(json.loads(Path(path).read_text()))
