"""Cohort selection and labels. The task definition is in ``config``.

One row per ICU stay that enters the study, with its index time and label.
Every exclusion step is counted in ``CohortResult.flow`` so the cohort can be
reported the way clinical papers report it (a flow diagram), and so a change
in the numbers is visible.

**Joins are validated, not assumed.** Each ICU stay must belong to exactly one
admission and one patient, and the ``subject_id`` on the stay must equal the
``subject_id`` on its admission. A mismatch raises
:class:`~mimic_iii.errors.IntegrityError` instead of silently pairing one
patient's stay with another's outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .errors import IntegrityError, SchemaError

COHORT_COLUMNS: tuple[str, ...] = (
    "subject_id", "hadm_id", "icustay_id", "intime", "index_time",
    "age", "age_over_89", "gender", "admission_type", "first_careunit",
    "label",
)


@dataclass
class CohortResult:
    cohort: pd.DataFrame
    flow: list[dict] = field(default_factory=list)
    #: Diagnosis groups of the final cohort, for description only. Assigned at
    #: discharge, so never a feature.
    cardiac_groups: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        y = self.cohort["label"]
        return {
            "flow": self.flow,
            "n_stays": int(len(self.cohort)),
            "n_subjects": int(self.cohort["subject_id"].nunique()),
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "cardiac_groups_description_only": self.cardiac_groups,
            "notes": self.notes,
        }


def age_in_years(dob: pd.Series, at: pd.Series) -> pd.Series:
    """Whole years between ``dob`` and ``at``, computed from calendar fields.

    MIMIC-III shifts the birth date of patients over 89 by about 300 years, and
    subtracting those timestamps overflows the nanosecond range pandas uses.
    Year/month/day arithmetic cannot overflow.
    """
    years = at.dt.year - dob.dt.year
    before_birthday = (at.dt.month * 100 + at.dt.day) < (dob.dt.month * 100 + dob.dt.day)
    return (years - before_birthday.astype(int)).astype(float)


def _flow(rows: list[dict], step: str, df: pd.DataFrame) -> None:
    rows.append({"step": step, "stays": int(len(df)),
                 "subjects": int(df["subject_id"].nunique())})


def build_cohort(
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    diagnoses: pd.DataFrame,
    spec: config.TaskSpec = config.TaskSpec(),
) -> CohortResult:
    """Apply the cohort definition and attach labels and index times."""
    flow: list[dict] = []
    notes: list[str] = []

    stays = icustays.copy()
    if stays["intime"].isna().any() or stays["outtime"].isna().any():
        n_bad = int((stays["intime"].isna() | stays["outtime"].isna()).sum())
        notes.append(f"{n_bad} ICU stays without intime/outtime dropped")
        stays = stays.dropna(subset=["intime", "outtime"])
    if (stays["outtime"] < stays["intime"]).any():
        raise IntegrityError(
            f"{int((stays['outtime'] < stays['intime']).sum())} ICU stays end before they start"
        )
    _flow(flow, "All ICU stays", stays)

    # --- validated joins ------------------------------------------------------
    adm_cols = ["subject_id", "hadm_id", "admission_type", "deathtime", "dischtime",
                "hospital_expire_flag"]
    try:
        merged = stays.merge(
            admissions[adm_cols].rename(columns={"subject_id": "adm_subject_id"}),
            on="hadm_id", how="left", validate="many_to_one", indicator=True,
        )
    except pd.errors.MergeError as exc:
        raise IntegrityError(f"ADMISSIONS.hadm_id is not unique: {exc}") from exc
    orphans = merged["_merge"] != "both"
    if orphans.any():
        raise IntegrityError(
            f"{int(orphans.sum())} ICU stays reference an hadm_id missing from "
            f"ADMISSIONS, e.g. {merged.loc[orphans, 'hadm_id'].head(3).tolist()}"
        )
    mismatch = merged["subject_id"] != merged["adm_subject_id"]
    if mismatch.any():
        raise IntegrityError(
            f"{int(mismatch.sum())} ICU stays have a subject_id different from "
            f"their admission's, e.g. icustay_id "
            f"{merged.loc[mismatch, 'icustay_id'].head(3).tolist()}"
        )
    merged = merged.drop(columns=["_merge", "adm_subject_id"])

    try:
        merged = merged.merge(
            patients[["subject_id", "gender", "dob"]], on="subject_id",
            how="left", validate="many_to_one", indicator=True,
        )
    except pd.errors.MergeError as exc:
        raise IntegrityError(f"PATIENTS.subject_id is not unique: {exc}") from exc
    if (merged["_merge"] != "both").any():
        raise IntegrityError("ICU stays reference a subject_id missing from PATIENTS")
    merged = merged.drop(columns="_merge")

    flags = set(pd.unique(merged["hospital_expire_flag"].dropna()))
    if not flags <= {0, 1}:
        raise SchemaError(f"ADMISSIONS.hospital_expire_flag must be 0/1, found {sorted(flags)}")
    if merged["hospital_expire_flag"].isna().any():
        raise SchemaError("ADMISSIONS.hospital_expire_flag has missing values")

    # --- age ------------------------------------------------------------------
    merged["age"] = age_in_years(merged["dob"], merged["intime"])
    merged["age_over_89"] = (merged["age"] > 89).astype(int)
    merged.loc[merged["age_over_89"] == 1, "age"] = spec.age_over_89_value

    adults = merged[merged["age"] >= spec.min_age]
    _flow(flow, f"Adults (age >= {spec.min_age:g})", adults)

    # --- first ICU stay ---------------------------------------------------------
    if spec.first_stay_only:
        adults = (adults.sort_values(["subject_id", "intime", "icustay_id"])
                  .groupby("subject_id", as_index=False).head(1))
        _flow(flow, "First adult ICU stay per patient", adults)

    # --- cardiac diagnosis (selection only) --------------------------------------
    dx = diagnoses.dropna(subset=["icd9_code"]).copy()
    dx["prefix"] = dx["icd9_code"].astype(str).str.strip().str[:3]
    dx = dx[dx["prefix"].isin(spec.cardiac_icd9_prefixes)]
    cardiac = adults[adults["hadm_id"].isin(set(dx["hadm_id"]))]
    _flow(flow, "Cardiac ICD-9 code on the admission", cardiac)

    # --- still in the ICU and alive at the index time ------------------------------
    cohort = cardiac.copy()
    cohort["index_time"] = cohort["intime"] + pd.Timedelta(hours=spec.observation_hours)
    left_early = cohort["outtime"] < cohort["index_time"]
    died_early = cohort["deathtime"].notna() & (cohort["deathtime"] < cohort["index_time"])
    cohort = cohort[~(left_early | died_early)]

    inconsistent = (cohort["hospital_expire_flag"] == 1) & cohort["deathtime"].isna()
    if inconsistent.any():
        notes.append(
            f"{int(inconsistent.sum())} stays have hospital_expire_flag=1 but no "
            f"deathtime; the flag is used as the label"
        )
    # MIMIC-III has a few stays whose ICU outtime is later than the hospital
    # discharge. If the admission ended before the index time there is nothing
    # left to predict, so the stay is excluded and counted.
    discharged_before_index = cohort["dischtime"] < cohort["index_time"]
    if discharged_before_index.any():
        notes.append(
            f"{int(discharged_before_index.sum())} stays excluded: hospital "
            f"discharge before the index time despite a later ICU outtime"
        )
        cohort = cohort[~discharged_before_index]
    _flow(flow, f"In the ICU and alive at intime + {spec.observation_hours:g} h", cohort)

    cohort["label"] = cohort["hospital_expire_flag"].astype(np.int64)
    cohort = cohort[list(COHORT_COLUMNS)].sort_values("icustay_id").reset_index(drop=True)
    for col in ("subject_id", "hadm_id", "icustay_id"):
        cohort[col] = cohort[col].astype(np.int64)

    groups = (dx[dx["hadm_id"].isin(set(cohort["hadm_id"]))]
              .assign(group=lambda d: d["prefix"].map(config.CARDIAC_ICD9))
              .groupby("group")["hadm_id"].nunique())
    return CohortResult(
        cohort=cohort,
        flow=flow,
        cardiac_groups={str(k): int(v) for k, v in groups.items()},
        notes=notes,
    )
