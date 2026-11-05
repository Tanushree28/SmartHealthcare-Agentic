"""A small synthetic MIMIC-III database for the offline tests.

Not real patients and not derived from real patients: every row is invented
here. The tables have the real MIMIC-III v1.4 column names, so the readers are
exercised exactly as on the real files, in either column-name case and with or
without gzip.

The first patients are hand-placed so each one exercises one rule:

====  ===========================================================  ===========
subj  what it tests                                                in cohort?
====  ===========================================================  ===========
1     the reference stay: windows, units, error rows, a second     yes (101)
      later ICU stay, labs before/inside the lookback, a lab from
      another admission, a wrong-unit lab, values at/after index
2     died after the index time -> label 1                         yes (201)
3     under 18                                                     no
4     over 89 (shifted DOB); no events at all -> all missing       yes (401)
5     no cardiac ICD-9 code                                        no
6     died before the index time                                   no
7     left the ICU before the index time                           no
====  ===========================================================  ===========

Subjects 100+ are generated: cardiac, adult, a few vitals and labs each, one in
three positive, so a patient-level three-way split is possible.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from mimic_iii import config

T0 = datetime(2150, 1, 1, 8, 0, 0)
FMT = "%Y-%m-%d %H:%M:%S"


def ts(base: datetime, hours: float = 0.0) -> str:
    return (base + timedelta(hours=hours)).strftime(FMT)


#: The expected cohort for ``make_database(n_extra=N)``: reference stays.
EXPECTED_FIXED_STAYS = {101: 0, 201: 1, 401: 0}
EXCLUDED_STAYS = {102, 301, 501, 601, 701}


def _event(rows, subject, hadm, icustay, itemid, at, value, uom, error=0):
    rows.append({"subject_id": subject, "hadm_id": hadm, "icustay_id": icustay,
                 "itemid": itemid, "charttime": at, "value": str(value),
                 "valuenum": value, "valueuom": uom, "error": error})


def _lab(rows, subject, hadm, itemid, at, value, uom):
    rows.append({"subject_id": subject, "hadm_id": hadm, "itemid": itemid,
                 "charttime": at, "value": str(value), "valuenum": value,
                 "valueuom": uom, "flag": None})


def build_tables(n_extra: int = 24) -> dict[str, pd.DataFrame]:
    patients, admissions, icustays, dx, chart, labs = [], [], [], [], [], []

    def add_patient(subject, dob, gender="F"):
        patients.append({"subject_id": subject, "gender": gender, "dob": dob,
                         "dod": None, "dod_hosp": None, "dod_ssn": None, "expire_flag": 0})

    def add_stay(subject, hadm, icustay, intime, icu_hours, icd9, died_at=None,
                 adm_type="EMERGENCY", unit="CCU", hosp_hours=None):
        outtime = intime + timedelta(hours=icu_hours)
        disch = died_at or (intime + timedelta(hours=hosp_hours or icu_hours + 48))
        admissions.append({
            "subject_id": subject, "hadm_id": hadm,
            "admittime": ts(intime, -6), "dischtime": disch.strftime(FMT),
            "deathtime": died_at.strftime(FMT) if died_at else None,
            "admission_type": adm_type, "admission_location": "EMERGENCY ROOM ADMIT",
            "discharge_location": "DEAD/EXPIRED" if died_at else "HOME",
            "insurance": "Medicare", "hospital_expire_flag": 1 if died_at else 0,
            "has_chartevents_data": 1,
        })
        icustays.append({
            "subject_id": subject, "hadm_id": hadm, "icustay_id": icustay,
            "dbsource": "metavision", "first_careunit": unit, "last_careunit": unit,
            "intime": intime.strftime(FMT), "outtime": outtime.strftime(FMT),
            "los": icu_hours / 24,
        })
        for seq, code in enumerate(icd9, start=1):
            dx.append({"subject_id": subject, "hadm_id": hadm, "seq_num": seq, "icd9_code": code})

    # --- subject 1: the reference stay ------------------------------------------
    add_patient(1, "2100-03-01 00:00:00", "M")
    add_stay(1, 11, 101, T0, 72, ["4280", "5849"])
    hr = 220045
    _event(chart, 1, 11, 101, hr, ts(T0, -1), 200, "bpm")        # before intime: out
    _event(chart, 1, 11, 101, hr, ts(T0, 1), 80, "bpm")
    _event(chart, 1, 11, 101, hr, ts(T0, 2), 90, "bpm")
    _event(chart, 1, 11, 101, hr, ts(T0, 3), 70, "bpm", error=1)  # error row: out
    _event(chart, 1, 11, 101, hr, ts(T0, 23), 100, "bpm")        # last in window
    _event(chart, 1, 11, 101, hr, ts(T0, 24), 150, "bpm")        # at index time: out
    _event(chart, 1, 11, 101, hr, ts(T0, 30), 160, "bpm")        # after index: out
    _event(chart, 1, 11, 101, 223761, ts(T0, 2), 98.6, "?F")     # Fahrenheit -> 37.0 C
    _event(chart, 1, 11, 101, 220179, ts(T0, 4), 500, "mmHg")    # implausible: out
    _event(chart, 1, 11, 101, 220179, ts(T0, 4), 120, "mmHg")
    _event(chart, 1, 11, 101, 220277, ts(T0, 5), float("nan"), "%")  # no value: out
    _lab(labs, 1, 11, 50912, ts(T0, -7), 9.0, "mg/dL")   # before the 6 h lookback: out
    _lab(labs, 1, 11, 50912, ts(T0, -5), 1.0, "mg/dL")   # inside the lookback
    _lab(labs, 1, 11, 50912, ts(T0, 10), 2.0, "mg/dL")
    _lab(labs, 1, 11, 50912, ts(T0, 26), 7.0, "mg/dL")   # after index: out
    _lab(labs, 1, 11, 50971, ts(T0, 6), 4.0, "mg/dL")    # wrong unit: out
    _lab(labs, 1, None, 50983, ts(T0, 5), 140.0, "mEq/L")  # no hadm_id: kept
    _lab(labs, 1, 12, 50931, ts(T0, 7), 300.0, "mg/dL")  # another admission: out
    _lab(labs, 1, 11, 51003, ts(T0, 8), 0.5, "ng/ml")    # lower-case unit: kept
    # A second, later ICU stay: not the first stay, so excluded with its events.
    add_stay(1, 12, 102, T0 + timedelta(days=150), 72, ["4280"],
             died_at=T0 + timedelta(days=152))
    _event(chart, 1, 12, 102, hr, ts(T0 + timedelta(days=150), 2), 140, "bpm")

    # --- subject 2: died after the index time -------------------------------------
    add_patient(2, "2090-01-01 00:00:00", "F")
    add_stay(2, 21, 201, T0, 96, ["41071"], died_at=T0 + timedelta(hours=48),
             adm_type="URGENT", unit="MICU")
    _event(chart, 2, 21, 201, hr, ts(T0, 2), 110, "bpm")
    _event(chart, 2, 21, 201, 220210, ts(T0, 2), 28, "insp/min")
    _lab(labs, 2, 21, 50813, ts(T0, 3), 4.5, "mmol/L")

    # --- subjects 3-7: exclusions and the over-89 patient ---------------------------
    add_patient(3, "2140-01-01 00:00:00")  # 10 years old at admission
    add_stay(3, 31, 301, T0, 72, ["4271"])
    add_patient(4, "1850-01-01 00:00:00", "M")  # shifted: computed age ~300
    add_stay(4, 41, 401, T0, 72, ["4271"], adm_type="ELECTIVE", unit="CSRU")
    add_patient(5, "2100-01-01 00:00:00")
    add_stay(5, 51, 501, T0, 72, ["486"])  # pneumonia, not cardiac
    add_patient(6, "2100-01-01 00:00:00")
    add_stay(6, 61, 601, T0, 72, ["4280"], died_at=T0 + timedelta(hours=10))
    add_patient(7, "2100-01-01 00:00:00")
    add_stay(7, 71, 701, T0, 12, ["4280"])  # out of the ICU at 12 h

    # --- generated patients -----------------------------------------------------------
    for k in range(n_extra):
        subject = 100 + k
        hadm, icustay = 1000 + k, 10_000 + k
        start = T0 + timedelta(days=3 * k)
        positive = k % 3 == 0
        died = start + timedelta(hours=60) if positive else None
        add_patient(subject, "2095-06-15 00:00:00", "M" if k % 2 else "F")
        add_stay(subject, hadm, icustay, start, 80, ["4280" if k % 2 else "41401"], died_at=died,
                 unit="CCU" if k % 4 else "SICU")
        base = 115.0 if positive else 80.0
        for h in (1, 6, 12, 20):
            _event(chart, subject, hadm, icustay, hr, ts(start, h), base + h + k % 5, "bpm")
            _event(chart, subject, hadm, icustay, 220179, ts(start, h), 125 - (25 if positive else 0) + k % 7, "mmHg")
        _lab(labs, subject, hadm, 50813, ts(start, 4), 4.0 if positive else 1.2 + 0.1 * (k % 4), "mmol/L")
        _lab(labs, subject, hadm, 50912, ts(start, 4), 1.0 + 0.1 * (k % 6), "mg/dL")

    d_items = [{"itemid": i, "label": f"chart item {i}"} for i in config.itemids("chart")]
    d_lab = [{"itemid": i, "label": f"lab item {i}"} for i in config.itemids("lab")]

    def frame(rows):
        df = pd.DataFrame(rows)
        df.insert(0, "row_id", range(1, len(df) + 1))
        return df

    tables = {
        "PATIENTS": frame(patients), "ADMISSIONS": frame(admissions),
        "ICUSTAYS": frame(icustays), "DIAGNOSES_ICD": frame(dx),
        "D_ITEMS": frame(d_items), "D_LABITEMS": frame(d_lab),
        "CHARTEVENTS": frame(chart), "LABEVENTS": frame(labs),
    }
    for t in ("CHARTEVENTS", "LABEVENTS", "ADMISSIONS", "ICUSTAYS"):
        tables[t]["hadm_id"] = tables[t]["hadm_id"].astype("Int64")
    return tables


def write_database(directory: Path, tables: dict[str, pd.DataFrame] | None = None,
                   upper: bool = True, gz: bool = True, n_extra: int = 24) -> Path:
    """Write the tables like the full release (upper-case, .csv.gz) or the Demo."""
    tables = tables if tables is not None else build_tables(n_extra=n_extra)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out = df.copy()
        out.columns = [c.upper() if upper else c.lower() for c in out.columns]
        suffix = ".csv.gz" if gz else ".csv"
        out.to_csv(directory / f"{name}{suffix}", index=False)
    return directory
