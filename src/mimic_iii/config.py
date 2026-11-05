"""The MIMIC-III prediction task, variables and data locations.

**The task, stated once.** Everything below is the default ``TaskSpec``; the
same values are in ``configs/mimic_iii.toml``, and a test asserts the two agree.

- *Cohort:* adults (age >= 18 at ICU admission); each patient's first adult
  ICU stay; the hospital admission carries a cardiac ICD-9 code (410-412
  myocardial infarction, 414 coronary artery disease, 427 arrhythmia, 428 heart
  failure); the patient is still in the ICU and alive at the index time.
- *Index time:* ICU admission (``ICUSTAYS.intime``) + 24 h. The prediction is
  made at this moment, using only data recorded before it.
- *Observation window:* vitals from ``[intime, index)``; labs from
  ``[intime - 6 h, index)``, because the first labs of an ICU stay are often
  drawn in the emergency department shortly before transfer.
- *Target:* in-hospital mortality, ``ADMISSIONS.hospital_expire_flag``.
- *Prediction horizon:* from the index time to hospital discharge. Patients who
  died or left the ICU before the index time are excluded, so every positive
  label is a death *after* the prediction was made.

**A known limitation of the cohort definition.** ICD-9 codes are assigned at
discharge. Selecting the cohort with them is standard practice for
retrospective MIMIC studies, but a deployed model would not know them at the
index time. They are used to *select* patients only, never as features.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from common import paths

# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------

#: Data source name -> directory under the data root.
SOURCES: dict[str, str] = {
    "demo": "mimiciii-demo",  # MIMIC-III Clinical Database Demo v1.4, open
    "full": "mimiciii",  # MIMIC-III Clinical Database v1.4, credentialed
}

SOURCE_DESCRIPTIONS: dict[str, str] = {
    "demo": (
        "MIMIC-III Clinical Database Demo v1.4: 100 patients, all of whom had "
        "died by the time PhysioNet selected them. Verifies the pipeline; "
        "cannot estimate clinical performance."
    ),
    "full": "MIMIC-III Clinical Database v1.4 (credentialed access).",
}

DEFAULT_CONFIG_PATH = paths.PROJECT_ROOT / "configs" / "mimic_iii.toml"


def source_dir(source: str) -> Path:
    """Raw CSV directory for a data source."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(SOURCES)}")
    return paths.data_root() / SOURCES[source]


def output_dir(source: str) -> Path:
    """Where this pipeline writes features, splits, models and reports."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(SOURCES)}")
    return paths.processed_root() / SOURCES[source]


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------

#: ICD-9 three-character category -> cardiac group.
CARDIAC_ICD9: dict[str, str] = {
    "410": "MI", "411": "MI", "412": "MI", "414": "CAD",
    "427": "Arrhythmia", "428": "Heart failure",
}


@dataclass(frozen=True)
class TaskSpec:
    """Cohort, timing and split settings. See the module docstring."""

    name: str = "cardiac_icu_in_hospital_mortality_24h"
    target: str = "in_hospital_mortality"
    min_age: float = 18.0
    #: MIMIC-III shifts the date of birth of patients older than 89 so their
    #: computed age is about 300. They are recoded to this value, the median
    #: age of that group reported in the MIMIC-III documentation.
    age_over_89_value: float = 91.4
    first_stay_only: bool = True
    cardiac_icd9_prefixes: tuple[str, ...] = tuple(CARDIAC_ICD9)
    observation_hours: float = 24.0
    lab_lookback_hours: float = 6.0
    split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_seed: int = 20261014

    def __post_init__(self) -> None:
        if self.observation_hours <= 0:
            raise ValueError("observation_hours must be positive")
        if self.lab_lookback_hours < 0:
            raise ValueError("lab_lookback_hours must be >= 0")
        if len(self.split_fractions) != 3 or any(f <= 0 for f in self.split_fractions):
            raise ValueError(f"split_fractions must be three positive numbers, got {self.split_fractions}")
        if abs(sum(self.split_fractions) - 1.0) > 1e-9:
            raise ValueError(f"split_fractions must sum to 1, got {sum(self.split_fractions)}")
        unknown = set(self.cardiac_icd9_prefixes) - set(CARDIAC_ICD9)
        if unknown:
            raise ValueError(f"no cardiac group defined for ICD-9 prefixes {sorted(unknown)}")

    def describe(self) -> dict:
        """The task in words and numbers, for every report and model file."""
        return {
            **asdict(self),
            "index_time": f"ICUSTAYS.intime + {self.observation_hours:g} h",
            "vitals_window": f"[intime, intime + {self.observation_hours:g} h)",
            "labs_window": (
                f"[intime - {self.lab_lookback_hours:g} h, "
                f"intime + {self.observation_hours:g} h)"
            ),
            "label": "ADMISSIONS.hospital_expire_flag (death before hospital discharge)",
            "horizon": "index time to hospital discharge",
            "exclusions": [
                f"age < {self.min_age:g}",
                "not the patient's first adult ICU stay" if self.first_stay_only else None,
                "no cardiac ICD-9 code on the admission",
                "left the ICU before the index time",
                "died before the index time",
            ],
        }


def load_task_spec(path: Path | str = DEFAULT_CONFIG_PATH) -> TaskSpec:
    """Read the ``[task]`` table of a TOML config into a :class:`TaskSpec`.

    Unknown keys are rejected: a misspelt key that silently fell back to the
    default would change the task without anyone noticing.
    """
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    table = raw.get("task")
    if table is None:
        raise ValueError(f"{path} has no [task] table")
    allowed = {f.name for f in fields(TaskSpec)}
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown [task] keys {sorted(unknown)}")
    kwargs = dict(table)
    for key in ("cardiac_icd9_prefixes", "split_fractions"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return TaskSpec(**kwargs)


def load_model_spec(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    """The ``[model]`` table, or ``{}`` when absent."""
    with open(path, "rb") as handle:
        return dict(tomllib.load(handle).get("model", {}))


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variable:
    """One clinical measurement: where it is stored and what values are valid.

    ``itemids`` follow the MIT-LCP ``mimic-code`` concepts and cover both
    charting systems (CareVue ITEMIDs < 200000, MetaVision >= 220000). The
    labels are checked against ``D_ITEMS``/``D_LABITEMS`` at ingestion time.
    """

    name: str
    source: str  # "chart" (CHARTEVENTS) or "lab" (LABEVENTS)
    itemids: tuple[int, ...]
    unit: str  # canonical unit after conversion
    valid_range: tuple[float, float]
    #: Lower-case unit strings accepted for lab rows. A lab row whose unit is
    #: present and not in this set is dropped and counted, never rescaled by
    #: guesswork. Chart units are fixed per ITEMID, so chart rows are not checked.
    accepted_units: tuple[str, ...] = ()
    #: ITEMIDs recorded in Fahrenheit, converted to Celsius on ingestion.
    fahrenheit_itemids: tuple[int, ...] = field(default=())


VARIABLES: tuple[Variable, ...] = (
    # --- vital signs, CHARTEVENTS -------------------------------------------
    Variable("heart_rate", "chart", (211, 220045), "bpm", (0, 300)),
    Variable("sbp", "chart", (51, 442, 455, 6701, 220179, 220050), "mmHg", (0, 400)),
    Variable("dbp", "chart", (8368, 8440, 8441, 8555, 220180, 220051), "mmHg", (0, 300)),
    Variable("mbp", "chart", (456, 52, 6702, 443, 220052, 220181, 225312), "mmHg", (0, 300)),
    Variable("resp_rate", "chart", (615, 618, 220210, 224690), "breaths/min", (0, 80)),
    Variable("spo2", "chart", (646, 220277), "%", (0, 100)),
    Variable("temp_c", "chart", (223762, 676, 223761, 678), "degC", (25, 45),
             fahrenheit_itemids=(223761, 678)),
    # --- laboratory results, LABEVENTS --------------------------------------
    Variable("troponin_t", "lab", (51003,), "ng/mL", (0, 100), ("ng/ml",)),
    Variable("troponin_i", "lab", (51002,), "ng/mL", (0, 1000), ("ng/ml",)),
    Variable("ck_mb", "lab", (50911,), "ng/mL", (0, 1000), ("ng/ml",)),
    Variable("creatinine", "lab", (50912,), "mg/dL", (0, 30), ("mg/dl",)),
    Variable("potassium", "lab", (50971, 50822), "mEq/L", (1, 12), ("meq/l", "mmol/l")),
    Variable("sodium", "lab", (50983, 50824), "mEq/L", (90, 200), ("meq/l", "mmol/l")),
    Variable("bnp", "lab", (50963,), "pg/mL", (0, 100000), ("pg/ml",)),
    Variable("lactate", "lab", (50813,), "mmol/L", (0, 40), ("mmol/l",)),
    Variable("glucose", "lab", (50931, 50809), "mg/dL", (10, 2000), ("mg/dl",)),
    Variable("hemoglobin", "lab", (51222, 50811), "g/dL", (1, 25), ("g/dl",)),
    Variable("wbc", "lab", (51301, 51300), "K/uL", (0, 500), ("k/ul",)),
)

VARIABLE_NAMES: tuple[str, ...] = tuple(v.name for v in VARIABLES)

#: Per-variable summaries over the observation window, in column order.
AGGREGATES: tuple[str, ...] = ("mean", "min", "max", "last", "count")

#: Demographic and administrative fields known at the index time.
NUMERIC_STATIC: tuple[str, ...] = ("age", "age_over_89")
CATEGORICAL_STATIC: tuple[str, ...] = ("gender", "admission_type", "first_careunit")

#: Identifier and label columns carried by the feature table but never fed to
#: a model.
ID_COLUMNS: tuple[str, ...] = ("subject_id", "hadm_id", "icustay_id")
LABEL_COLUMN = "label"

#: Columns that encode what happened **after** the index time (outcome,
#: discharge, length of stay) or that are assigned at discharge (diagnosis
#: codes). Any of them in a model's input is target leakage. Checked by
#: ``features.assert_no_leakage``.
POST_INDEX_COLUMNS: frozenset[str] = frozenset({
    "hospital_expire_flag", "deathtime", "dischtime", "discharge_location",
    "dod", "dod_hosp", "dod_ssn", "expire_flag", "outtime", "los",
    "last_careunit", "last_wardid", "icd9_code", "seq_num", "cardiac_group",
    LABEL_COLUMN,
})


def itemids(source: str) -> tuple[int, ...]:
    """All ITEMIDs read from one events table."""
    return tuple(sorted({i for v in VARIABLES if v.source == source for i in v.itemids}))


def itemid_to_variable(source: str) -> dict[int, Variable]:
    mapping: dict[int, Variable] = {}
    for v in VARIABLES:
        if v.source != source:
            continue
        for i in v.itemids:
            if i in mapping:
                raise ValueError(f"ITEMID {i} assigned to both {mapping[i].name} and {v.name}")
            mapping[i] = v
    return mapping
