"""Time-window features: vitals and labs before the index time.

For each cohort stay and each variable in ``config.VARIABLES``: mean, min, max,
last value and number of measurements in the observation window, plus a
missingness flag (1 when nothing was measured). Demographics and admission
fields known at the index time are added. Nothing recorded at or after the
index time can reach a feature: every window is half-open, ``[start, index)``.

**Joins.** Chart events are tied to a stay by ``icustay_id`` and their
``subject_id`` must match the stay's. Lab events have no ``icustay_id``; they
are tied by ``subject_id`` and time window, and a lab whose ``hadm_id`` is
present but names a *different* admission is dropped. Every dropped row is
counted with its reason in ``EventStats``.

**Units.** Temperatures charted in Fahrenheit are converted to Celsius by
ITEMID. Lab rows whose unit is present but not one the variable accepts are
dropped, never rescaled by guesswork. Values outside a physiologic range are
dropped as charting errors.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, ingestion
from .errors import IntegrityError, LeakageError

EVENT_COLUMNS: tuple[str, ...] = ("icustay_id", "variable", "charttime", "value")


@dataclass
class EventStats:
    chart: dict = field(default_factory=dict)
    lab: dict = field(default_factory=dict)
    dropped: Counter = field(default_factory=Counter)
    converted: Counter = field(default_factory=Counter)
    accepted_without_unit: Counter = field(default_factory=Counter)
    kept: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "chart_ingest": self.chart,
            "lab_ingest": self.lab,
            "dropped": dict(sorted(self.dropped.items())),
            "converted": dict(sorted(self.converted.items())),
            "lab_rows_without_unit_accepted": dict(sorted(self.accepted_without_unit.items())),
            "kept_per_variable": dict(sorted(self.kept.items())),
        }


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame({
        "icustay_id": pd.Series(dtype="int64"),
        "variable": pd.Series(dtype="object"),
        "charttime": pd.Series(dtype="datetime64[ns]"),
        "value": pd.Series(dtype="float64"),
    })


def window_chart_events(
    chart: pd.DataFrame,
    cohort: pd.DataFrame,
    spec: config.TaskSpec,
    stats: EventStats,
) -> pd.DataFrame:
    """Chart rows inside ``[intime, index_time)`` of their own stay."""
    lookup = config.itemid_to_variable("chart")
    if chart.empty:
        return _empty_events()

    stays = cohort[["icustay_id", "subject_id", "intime", "index_time"]].rename(
        columns={"subject_id": "stay_subject_id"})
    df = chart.merge(stays, on="icustay_id", how="inner", validate="many_to_one")

    mismatch = df["subject_id"] != df["stay_subject_id"]
    stats.dropped["chart:subject_mismatch"] += int(mismatch.sum())
    df = df[~mismatch]

    inside = (df["charttime"] >= df["intime"]) & (df["charttime"] < df["index_time"])
    stats.dropped["chart:outside_window"] += int((~inside).sum())
    df = df[inside].copy()

    df["variable"] = df["itemid"].map(lambda i: lookup[int(i)].name)
    df["value"] = df["valuenum"].astype(float)

    for var in config.VARIABLES:
        if var.source != "chart" or not var.fahrenheit_itemids:
            continue
        f = df["itemid"].isin(var.fahrenheit_itemids)
        df.loc[f, "value"] = (df.loc[f, "value"] - 32.0) * 5.0 / 9.0
        stats.converted[f"{var.name}:fahrenheit_to_celsius"] += int(f.sum())

    return df[list(EVENT_COLUMNS)]


def window_lab_events(
    labs: pd.DataFrame,
    cohort: pd.DataFrame,
    spec: config.TaskSpec,
    stats: EventStats,
) -> pd.DataFrame:
    """Lab rows inside ``[intime - lookback, index_time)`` of the patient's stay."""
    lookup = config.itemid_to_variable("lab")
    if labs.empty:
        return _empty_events()

    stays = cohort[["icustay_id", "subject_id", "hadm_id", "intime", "index_time"]].rename(
        columns={"hadm_id": "stay_hadm_id"})
    df = labs.merge(stays, on="subject_id", how="inner")

    start = df["intime"] - pd.Timedelta(hours=spec.lab_lookback_hours)
    inside = (df["charttime"] >= start) & (df["charttime"] < df["index_time"])
    stats.dropped["lab:outside_window"] += int((~inside).sum())
    df = df[inside].copy()

    other_admission = df["hadm_id"].notna() & (df["hadm_id"] != df["stay_hadm_id"])
    other_admission = other_admission.fillna(False).astype(bool)
    stats.dropped["lab:other_admission"] += int(other_admission.sum())
    df = df[~other_admission].copy()

    df["variable"] = df["itemid"].map(lambda i: lookup[int(i)].name)
    unit = df["valueuom"].astype("string").str.strip().str.lower()
    keep = pd.Series(True, index=df.index)
    for var in config.VARIABLES:
        if var.source != "lab":
            continue
        rows = df["variable"] == var.name
        no_unit = rows & unit.isna()
        bad_unit = rows & unit.notna() & ~unit.isin(var.accepted_units)
        bad_unit = bad_unit.fillna(False).astype(bool)
        stats.accepted_without_unit[var.name] += int(no_unit.sum())
        stats.dropped[f"lab:unit_mismatch:{var.name}"] += int(bad_unit.sum())
        keep &= ~bad_unit
    df = df[keep].copy()
    df["value"] = df["valuenum"].astype(float)
    return df[list(EVENT_COLUMNS)]


def drop_implausible(events: pd.DataFrame, stats: EventStats) -> pd.DataFrame:
    """Remove values outside each variable's physiologic range."""
    keep = pd.Series(True, index=events.index)
    for var in config.VARIABLES:
        lo, hi = var.valid_range
        rows = events["variable"] == var.name
        bad = rows & ((events["value"] < lo) | (events["value"] > hi))
        stats.dropped[f"out_of_range:{var.name}"] += int(bad.sum())
        keep &= ~bad
    out = events[keep]
    stats.kept.update(out["variable"].value_counts().to_dict())
    return out.reset_index(drop=True)


def extract_events(
    root: Path | str,
    cohort: pd.DataFrame,
    spec: config.TaskSpec = config.TaskSpec(),
    chunksize: int = ingestion.DEFAULT_CHUNKSIZE,
    progress: Callable[[ingestion.IngestStats], None] | None = None,
) -> tuple[pd.DataFrame, EventStats]:
    """Read, window, convert and clean the cohort's vitals and labs."""
    stats = EventStats()

    chart_raw, chart_stats = ingestion.read_events(
        root, "CHARTEVENTS", config.itemids("chart"), "icustay_id",
        cohort["icustay_id"], chunksize=chunksize, progress=progress)
    stats.chart = chart_stats.to_dict()

    lab_raw, lab_stats = ingestion.read_events(
        root, "LABEVENTS", config.itemids("lab"), "subject_id",
        cohort["subject_id"], chunksize=chunksize, progress=progress)
    stats.lab = lab_stats.to_dict()

    events = pd.concat(
        [window_chart_events(chart_raw, cohort, spec, stats),
         window_lab_events(lab_raw, cohort, spec, stats)],
        ignore_index=True,
    )
    return drop_implausible(events, stats), stats


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def event_feature_names() -> list[str]:
    return [f"{v}_{a}" for v in config.VARIABLE_NAMES for a in config.AGGREGATES]


def missing_flag_names() -> list[str]:
    return [f"{v}_missing" for v in config.VARIABLE_NAMES]


def feature_columns() -> dict[str, list[str]]:
    """Model input columns by type. Identifiers and the label are excluded."""
    return {
        "numeric": list(config.NUMERIC_STATIC) + event_feature_names(),
        "flags": missing_flag_names(),
        "categorical": list(config.CATEGORICAL_STATIC),
    }


def model_input_columns() -> list[str]:
    cols = feature_columns()
    return cols["numeric"] + cols["flags"] + cols["categorical"]


def assert_no_leakage(columns) -> None:
    """Refuse any post-index or identifier column as a model input."""
    cols = set(columns)
    leaked = sorted(cols & config.POST_INDEX_COLUMNS)
    ids = sorted(cols & set(config.ID_COLUMNS))
    times = sorted(cols & {"intime", "index_time", "admittime"})
    if leaked or ids or times:
        raise LeakageError(
            f"not allowed as model inputs -- post-index: {leaked}, "
            f"identifiers: {ids}, timestamps: {times}"
        )


def aggregate(events: pd.DataFrame, icustay_ids) -> pd.DataFrame:
    """Wide per-stay summaries. Stays with no events get NaN and count 0."""
    index = pd.Index(sorted({int(i) for i in icustay_ids}), name="icustay_id")
    names = event_feature_names()

    if events.empty:
        wide = pd.DataFrame(index=index, columns=names, dtype="float64")
    else:
        # Ties at one charttime are broken by value so "last" is deterministic.
        ordered = events.sort_values(["icustay_id", "variable", "charttime", "value"],
                                     kind="mergesort")
        grouped = ordered.groupby(["icustay_id", "variable"])["value"]
        summary = grouped.agg(["mean", "min", "max", "last", "count"])
        wide = summary.unstack("variable")
        wide.columns = [f"{var}_{agg}" for agg, var in wide.columns]
        wide = wide.reindex(index=index, columns=names)

    for v in config.VARIABLE_NAMES:
        wide[f"{v}_count"] = wide[f"{v}_count"].fillna(0).astype("int64")
        wide[f"{v}_missing"] = (wide[f"{v}_count"] == 0).astype("int64")
    return wide


def build_feature_table(cohort: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per stay: identifiers, label, static fields, event features."""
    unknown = set(events["icustay_id"]) - set(cohort["icustay_id"])
    if unknown:
        raise IntegrityError(f"events for stays not in the cohort: {sorted(unknown)[:5]}")
    wide = aggregate(events, cohort["icustay_id"])
    static = cohort.set_index("icustay_id")[
        ["subject_id", "hadm_id", *config.NUMERIC_STATIC, *config.CATEGORICAL_STATIC,
         config.LABEL_COLUMN]]
    table = static.join(wide, how="left").reset_index()
    ordered = (list(config.ID_COLUMNS) + [config.LABEL_COLUMN] + model_input_columns())
    table = table[ordered].sort_values("icustay_id").reset_index(drop=True)
    assert_no_leakage(model_input_columns())
    return table


#: Column dtypes for reading a saved feature table back.
def table_dtypes() -> dict[str, str]:
    dtypes = {c: "int64" for c in config.ID_COLUMNS}
    dtypes[config.LABEL_COLUMN] = "int64"
    for c in config.CATEGORICAL_STATIC:
        dtypes[c] = "string"
    for c in feature_columns()["numeric"]:
        dtypes[c] = "float64"
    for c in missing_flag_names():
        dtypes[c] = "int64"
    return dtypes


def read_feature_table(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=table_dtypes())
    expected = list(config.ID_COLUMNS) + [config.LABEL_COLUMN] + model_input_columns()
    if list(df.columns) != expected:
        missing = sorted(set(expected) - set(df.columns))
        extra = sorted(set(df.columns) - set(expected))
        raise ingestion.SchemaError(
            f"{path}: feature table columns differ from this code version "
            f"(missing {missing}, unexpected {extra}); rebuild it with "
            f"scripts/mimic_iii/build_features.py"
        )
    # Categorical NA must stay NA (not the string "<NA>") for the imputer.
    for c in config.CATEGORICAL_STATIC:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df
