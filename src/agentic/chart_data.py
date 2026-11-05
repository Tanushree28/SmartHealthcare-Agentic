"""Everything the chart agent reads for the cohort, loaded once.

The outcome is never in memory here: the label column is dropped as soon as
the feature table is read, the cohort file is read without it, and ADMISSIONS
is read for two columns only (hadm_id, diagnosis).

The lab window and unit rules repeat those of mimic_iii.features
(window_lab_events, drop_implausible, aggregate's tie-break) because that code
returns events without the hospital's ``flag`` column. A test checks that the
last value found here equals the feature table's ``<variable>_last`` for every
stay, so the two cannot drift apart unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mimic_iii import config, dataset, features, ingestion, train
from mimic_iii.errors import SchemaError

COHORT_COLUMNS = ("subject_id", "hadm_id", "icustay_id", "intime", "index_time", "age",
                  "age_over_89", "gender", "admission_type", "first_careunit")
LAB_COLUMNS = frozenset({"subject_id", "hadm_id", "itemid", "charttime", "valuenum",
                         "valueuom", "flag"})


@dataclass
class ChartSources:
    table: pd.DataFrame  # feature table without the label
    cohort: pd.DataFrame  # COHORT_COLUMNS
    admission_reason: dict[int, str]  # hadm_id -> ADMISSIONS.diagnosis
    medications: pd.DataFrame  # hadm_id, drug, startdate (days before the index day only)
    last_labs: pd.DataFrame  # icustay_id, variable, value, flag (last value in the window)
    model: object
    meta: dict
    split_of: dict[int, str]  # subject_id -> "train" | "val" | "test"
    source: str = "demo"


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _read(path: Path, wanted) -> pd.DataFrame:
    wanted = set(wanted)
    return _lower(pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in wanted,
                              low_memory=False))


def read_admission_reasons(raw_dir, hadm_ids) -> dict[int, str]:
    df = _read(ingestion.find_table(raw_dir, "ADMISSIONS"), {"hadm_id", "diagnosis"})
    if "diagnosis" not in df:
        return {}
    df = df[df["hadm_id"].isin(set(hadm_ids)) & df["diagnosis"].notna()]
    return {int(h): str(d).strip() for h, d in zip(df["hadm_id"], df["diagnosis"])}


def read_medications(raw_dir, cohort: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"hadm_id": pd.Series(dtype="int64"), "drug": pd.Series(dtype="object"),
                          "startdate": pd.Series(dtype="datetime64[ns]")})
    try:
        path = ingestion.find_table(raw_dir, "PRESCRIPTIONS")
    except SchemaError:
        return empty
    df = _read(path, {"hadm_id", "startdate", "drug"})
    df["startdate"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.merge(cohort[["hadm_id", "index_time"]], on="hadm_id", how="inner")
    # startdate has no time of day: keep only drugs started on an earlier DAY
    # than the index time, so nothing started after the 24-hour mark slips in.
    keep = (df["startdate"].notna() & df["drug"].notna()
            & (df["startdate"] < df["index_time"].dt.normalize()))
    return df.loc[keep, ["hadm_id", "drug", "startdate"]].reset_index(drop=True)


def read_last_labs(raw_dir, cohort: pd.DataFrame, spec: config.TaskSpec | None = None,
                   chunksize: int = 1_000_000) -> pd.DataFrame:
    spec = spec or config.load_task_spec()
    lookup = config.itemid_to_variable("lab")
    subjects = set(cohort["subject_id"].astype(int))
    columns = ["icustay_id", "variable", "value", "flag"]

    parts = []
    reader = pd.read_csv(ingestion.find_table(raw_dir, "LABEVENTS"),
                         usecols=lambda c: str(c).strip().lower() in LAB_COLUMNS,
                         chunksize=chunksize, low_memory=False)
    for chunk in reader:
        chunk = _lower(chunk)
        chunk = chunk[chunk["itemid"].isin(list(lookup)) & chunk["subject_id"].isin(subjects)
                      & chunk["valuenum"].notna()]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=columns)
    df = pd.concat(parts, ignore_index=True)
    if "flag" not in df:
        df["flag"] = None
    df["charttime"] = pd.to_datetime(df["charttime"])

    stays = cohort[["icustay_id", "subject_id", "hadm_id", "intime", "index_time"]].rename(
        columns={"hadm_id": "stay_hadm_id"})
    df = df.merge(stays, on="subject_id", how="inner")
    start = df["intime"] - pd.Timedelta(hours=spec.lab_lookback_hours)
    df = df[(df["charttime"] >= start) & (df["charttime"] < df["index_time"])]
    other = (df["hadm_id"].notna() & (df["hadm_id"] != df["stay_hadm_id"])).astype(bool)
    df = df[~other].copy()

    df["variable"] = df["itemid"].map(lambda i: lookup[int(i)].name)
    df["value"] = df["valuenum"].astype(float)
    unit = df["valueuom"].astype("string").str.strip().str.lower()
    keep = pd.Series(True, index=df.index)
    for var in config.VARIABLES:
        if var.source != "lab":
            continue
        rows = df["variable"] == var.name
        bad_unit = (rows & unit.notna() & ~unit.isin(var.accepted_units)).fillna(False)
        lo, hi = var.valid_range
        implausible = rows & ((df["value"] < lo) | (df["value"] > hi))
        keep &= ~bad_unit.astype(bool) & ~implausible
    df = df[keep]

    df = df.sort_values(["icustay_id", "variable", "charttime", "value"], kind="mergesort")
    last = df.groupby(["icustay_id", "variable"], as_index=False).tail(1)
    return last[columns].reset_index(drop=True)


def load_chart_sources(out_dir, raw_dir, source: str = "demo") -> ChartSources:
    out_dir, raw_dir = Path(out_dir).expanduser(), Path(raw_dir).expanduser()
    model, meta = train.load_model(out_dir)
    table = features.read_feature_table(out_dir / "features.csv.gz").drop(
        columns=[config.LABEL_COLUMN])
    cohort = pd.read_csv(out_dir / "cohort.csv.gz", usecols=list(COHORT_COLUMNS),
                         parse_dates=["intime", "index_time"])
    split_of: dict[int, str] = {}
    split_path = out_dir / "splits.json"
    if split_path.exists():
        split = dataset.read_split(split_path)
        for name in dataset.SPLIT_NAMES:
            for subject in split.subjects(name):
                split_of[int(subject)] = name
    return ChartSources(
        table=table, cohort=cohort,
        admission_reason=read_admission_reasons(raw_dir, set(cohort["hadm_id"])),
        medications=read_medications(raw_dir, cohort),
        last_labs=read_last_labs(raw_dir, cohort),
        model=model, meta=meta, split_of=split_of, source=source)
