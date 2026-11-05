"""Reading MIMIC-III tables: schema checks, date parsing, chunked event reads.

**Both releases look different on disk.** The open Demo ships lower-case column
names in ``.csv`` files; the full credentialed release ships upper-case names in
``.csv.gz``. Every reader here matches columns case-insensitively and accepts
either extension, then returns lower-case column names.

**Memory.** Full ``CHARTEVENTS`` is about 330 million rows (roughly 33 GB
uncompressed). :func:`read_events` never holds more than one chunk of the raw
table: each chunk is cut down to the wanted ITEMIDs and cohort keys before the
next is read, and only those rows are kept. Peak memory is set by the chunk
size and the size of the cohort's own events, not by the table.

**Malformed input fails loudly.** A missing table or column, an unparseable
date, a non-numeric value or a duplicated primary key raises
:class:`~mimic_iii.errors.SchemaError` naming the table and column.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .errors import DataSourceMissingError, SchemaError


@dataclass(frozen=True)
class TableSpec:
    required: tuple[str, ...]
    dates: tuple[str, ...] = ()
    key: str | None = None
    strings: tuple[str, ...] = ()  # read as text: codes with leading zeros, units


TABLES: dict[str, TableSpec] = {
    "PATIENTS": TableSpec(
        required=("subject_id", "gender", "dob"),
        dates=("dob",),
        key="subject_id",
        strings=("gender",),
    ),
    "ADMISSIONS": TableSpec(
        required=("subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                  "admission_type", "hospital_expire_flag"),
        dates=("admittime", "dischtime", "deathtime"),
        key="hadm_id",
        strings=("admission_type",),
    ),
    "ICUSTAYS": TableSpec(
        required=("subject_id", "hadm_id", "icustay_id", "first_careunit",
                  "intime", "outtime"),
        dates=("intime", "outtime"),
        key="icustay_id",
        strings=("first_careunit",),
    ),
    "DIAGNOSES_ICD": TableSpec(
        required=("subject_id", "hadm_id", "icd9_code"),
        strings=("icd9_code",),
    ),
    "D_ITEMS": TableSpec(required=("itemid", "label"), key="itemid", strings=("label",)),
    "D_LABITEMS": TableSpec(required=("itemid", "label"), key="itemid", strings=("label",)),
    "CHARTEVENTS": TableSpec(
        required=("subject_id", "hadm_id", "icustay_id", "itemid", "charttime",
                  "valuenum", "valueuom", "error"),
        dates=("charttime",),
        strings=("valueuom",),
    ),
    "LABEVENTS": TableSpec(
        required=("subject_id", "hadm_id", "itemid", "charttime", "valuenum",
                  "valueuom"),
        dates=("charttime",),
        strings=("valueuom",),
    ),
}

#: Tables the pipeline needs. ``validate_source`` checks all of them.
REQUIRED_TABLES: tuple[str, ...] = tuple(TABLES)

DEFAULT_CHUNKSIZE = 1_000_000


# ---------------------------------------------------------------------------
# Locating tables
# ---------------------------------------------------------------------------


def require_source(root: Path | str, source: str | None = None) -> Path:
    """Return ``root`` if it is a directory, else raise a message that says why."""
    root = Path(root).expanduser()
    if root.is_dir():
        return root
    if source == "full":
        raise DataSourceMissingError(
            f"full MIMIC-III not found at {root}. It requires PhysioNet "
            f"credentialing (account, CITI training, signed data use "
            f"agreement); place the v1.4 CSV files there once access is granted."
        )
    raise DataSourceMissingError(
        f"MIMIC-III data not found at {root}. The open Demo is at "
        f"https://physionet.org/content/mimiciii-demo/1.4/ (see COMMANDS.md)."
    )


def find_table(root: Path | str, name: str) -> Path:
    """Path of ``NAME.csv`` or ``NAME.csv.gz``, matched case-insensitively.

    Searches ``root`` first, then its subfolders (some downloads unpack into a
    nested folder). Prefers ``.csv`` when both forms exist.
    """
    root = Path(root)
    target = name.upper()

    def matches(paths: Iterable[Path]) -> list[Path]:
        out = []
        for p in paths:
            n = p.name.upper()
            if n in (f"{target}.CSV", f"{target}.CSV.GZ") and p.is_file():
                out.append(p)
        return sorted(out)

    hits = matches(root.iterdir()) if root.is_dir() else []
    if not hits:
        hits = matches(root.rglob("*.csv*"))
    if not hits:
        raise SchemaError(f"table {target} not found under {root} (looked for {target}.csv[.gz])")

    plain = [p for p in hits if p.name.upper().endswith(".CSV")]
    gz = [p for p in hits if p.name.upper().endswith(".CSV.GZ")]
    pick = plain if plain else gz
    if len(pick) > 1:
        raise SchemaError(f"table {target} is ambiguous: {[str(p) for p in pick]}")
    return pick[0]


def _header(path: Path) -> dict[str, str]:
    """Lower-case column name -> the name as written in the file."""
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise SchemaError(f"{path.name}: cannot read header: {exc}") from exc
    lookup: dict[str, str] = {}
    for c in columns:
        low = str(c).strip().lower()
        if low in lookup:
            raise SchemaError(f"{path.name}: column {low!r} appears twice")
        lookup[low] = c
    return lookup


def _check_columns(path: Path, name: str, lookup: dict[str, str]) -> None:
    missing = [c for c in TABLES[name].required if c not in lookup]
    if missing:
        raise SchemaError(f"{name} ({path.name}) is missing required columns {missing}")


def _parse_dates(df: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    for col in columns:
        if col not in df:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="ISO8601")
        bad = parsed.isna() & df[col].notna()
        if bad.any():
            examples = df.loc[bad, col].astype(str).head(3).tolist()
            raise SchemaError(
                f"{table}.{col}: {int(bad.sum())} unparseable timestamps, e.g. {examples}"
            )
        df[col] = parsed


def _check_numeric(df: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    for col in columns:
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad = coerced.isna() & df[col].notna()
            examples = df.loc[bad, col].astype(str).head(3).tolist()
            raise SchemaError(f"{table}.{col}: non-numeric values, e.g. {examples}")


_ID_COLUMNS = ("subject_id", "hadm_id", "icustay_id", "itemid")


# ---------------------------------------------------------------------------
# Small tables
# ---------------------------------------------------------------------------


def read_table(root: Path | str, name: str, optional: Iterable[str] = ()) -> pd.DataFrame:
    """Read a dimension or cohort table whole, validated and with parsed dates.

    ``optional`` names extra columns to keep when the file has them.
    """
    name = name.upper()
    if name not in TABLES:
        raise ValueError(f"no schema for table {name}")
    spec = TABLES[name]
    path = find_table(root, name)
    lookup = _header(path)
    _check_columns(path, name, lookup)

    wanted = list(spec.required) + [c for c in optional if c in lookup and c not in spec.required]
    try:
        df = pd.read_csv(
            path,
            usecols=[lookup[c] for c in wanted],
            dtype={lookup[c]: str for c in spec.strings if c in lookup},
            low_memory=False,
        )
    except (pd.errors.ParserError, ValueError) as exc:
        raise SchemaError(f"{name} ({path.name}): {exc}") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]

    _check_numeric(df, [c for c in _ID_COLUMNS if c in df], name)
    _parse_dates(df, spec.dates, name)

    if spec.key is not None:
        if df[spec.key].isna().any():
            raise SchemaError(f"{name}.{spec.key}: {int(df[spec.key].isna().sum())} missing keys")
        dup = df[spec.key].duplicated()
        if dup.any():
            raise SchemaError(
                f"{name}.{spec.key} is not unique: {int(dup.sum())} duplicates, "
                f"e.g. {df.loc[dup, spec.key].head(3).tolist()}"
            )
    return df


# ---------------------------------------------------------------------------
# Event tables, chunked
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    """Row accounting for one events table. Every dropped row has a reason."""

    table: str
    path: str = ""
    chunks: int = 0
    rows_read: int = 0
    rows_wanted_itemid: int = 0
    rows_outside_cohort: int = 0
    rows_missing_key: int = 0
    rows_error_flag: int = 0
    rows_missing_value: int = 0
    rows_duplicate: int = 0
    rows_kept: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def reconciles(self) -> bool:
        """Every row with a wanted ITEMID is either kept or counted as dropped."""
        dropped = (self.rows_outside_cohort + self.rows_missing_key + self.rows_error_flag
                   + self.rows_missing_value + self.rows_duplicate)
        return self.rows_kept + dropped == self.rows_wanted_itemid


def read_events(
    root: Path | str,
    name: str,
    itemids: Iterable[int],
    key: str,
    key_values: Iterable[int],
    chunksize: int = DEFAULT_CHUNKSIZE,
    progress: Callable[[IngestStats], None] | None = None,
) -> tuple[pd.DataFrame, IngestStats]:
    """Rows of ``CHARTEVENTS`` or ``LABEVENTS`` for the given ITEMIDs and cohort.

    ``key`` is the column that ties a row to the cohort (``icustay_id`` for
    chart events, ``subject_id`` for labs) and ``key_values`` the cohort's
    values of it. Rows flagged ``error == 1`` and rows without a numeric value
    are dropped and counted. Exact duplicate rows are dropped and counted.
    """
    name = name.upper()
    if name not in ("CHARTEVENTS", "LABEVENTS"):
        raise ValueError(f"read_events reads CHARTEVENTS or LABEVENTS, not {name}")
    if chunksize < 1:
        raise ValueError(f"chunksize must be >= 1, got {chunksize}")

    spec = TABLES[name]
    path = find_table(root, name)
    lookup = _header(path)
    _check_columns(path, name, lookup)
    if key not in spec.required:
        raise ValueError(f"{name} has no column {key!r}")

    wanted_items = np.asarray(sorted({int(i) for i in itemids}), dtype=np.int64)
    wanted_keys = np.asarray(sorted({int(k) for k in key_values}), dtype=np.int64)
    stats = IngestStats(table=name, path=str(path))

    parts: list[pd.DataFrame] = []
    columns = list(spec.required)
    try:
        reader = pd.read_csv(
            path,
            usecols=[lookup[c] for c in columns],
            dtype={lookup[c]: str for c in spec.strings},
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            stats.chunks += 1
            stats.rows_read += len(chunk)
            _check_numeric(chunk, ["itemid", key, "valuenum"], name)

            chunk = chunk[chunk["itemid"].isin(wanted_items)]
            stats.rows_wanted_itemid += len(chunk)

            missing_key = chunk[key].isna()
            stats.rows_missing_key += int(missing_key.sum())
            chunk = chunk[~missing_key]

            in_cohort = chunk[key].isin(wanted_keys)
            stats.rows_outside_cohort += int((~in_cohort).sum())
            chunk = chunk[in_cohort]

            if "error" in chunk:
                flagged = chunk["error"] == 1
                stats.rows_error_flag += int(flagged.sum())
                chunk = chunk[~flagged]

            no_value = chunk["valuenum"].isna()
            stats.rows_missing_value += int(no_value.sum())
            chunk = chunk[~no_value]

            if len(chunk):
                parts.append(chunk.drop(columns=["error"], errors="ignore"))
            if progress is not None:
                progress(stats)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise SchemaError(f"{name} ({path.name}): {exc}") from exc

    keep_cols = [c for c in columns if c != "error"]
    if parts:
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.DataFrame({c: pd.Series(dtype="float64") for c in keep_cols})
        df["valueuom"] = df["valueuom"].astype(object)
        df["charttime"] = pd.Series(dtype="datetime64[ns]")

    _parse_dates(df, ["charttime"], name)
    for col in ("subject_id", "icustay_id", "itemid"):
        if col in df:
            df[col] = df[col].astype("int64")
    if "hadm_id" in df:
        # Lab rows from outpatient visits have no admission.
        df["hadm_id"] = df["hadm_id"].astype("Int64")

    before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    stats.rows_duplicate = before - len(df)
    stats.rows_kept = len(df)
    return df, stats


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def check_itemids(root: Path | str) -> dict[str, list[str]]:
    """Confirm every configured ITEMID exists in its dictionary.

    Returns variable name -> dictionary labels, for the record. Raises if an
    ITEMID is absent -- a typo there would otherwise produce an all-missing
    feature that looks like a data gap.
    """
    d_items = read_table(root, "D_ITEMS")
    d_lab = read_table(root, "D_LABITEMS")
    known = {"chart": dict(zip(d_items.itemid, d_items.label)),
             "lab": dict(zip(d_lab.itemid, d_lab.label))}
    labels: dict[str, list[str]] = {}
    missing: list[str] = []
    for var in config.VARIABLES:
        table = known[var.source]
        for i in var.itemids:
            if i not in table:
                missing.append(f"{var.name}:{i}")
        labels[var.name] = [str(table[i]) for i in var.itemids if i in table]
    if missing:
        raise SchemaError(f"ITEMIDs absent from the dictionaries: {missing}")
    return labels


def validate_source(root: Path | str) -> dict[str, dict]:
    """Header-level check of every required table: present, columns complete.

    Cheap -- reads only headers -- so it can run before a long ingestion.
    Returns table -> {path, columns_ok}; raises on the first problem.
    """
    root = require_source(root)
    report: dict[str, dict] = {}
    for name in REQUIRED_TABLES:
        path = find_table(root, name)
        lookup = _header(path)
        _check_columns(path, name, lookup)
        report[name] = {"path": str(path), "bytes": path.stat().st_size}
    return report
