#!/usr/bin/env python3
"""Exploratory run of the Phase 2 decisions on the MIMIC-III Demo v1.4.

    python scripts/mimic_iii/explore_demo.py                    # $SMART_HEALTH_DATA/mimiciii-demo
    python scripts/mimic_iii/explore_demo.py /path/to/mimiciii-demo

A spike, not the Phase 2 pipeline: it has no tests and prints aggregate numbers
only. On the Demo's 52 cardiac stays, chosen from patients who had all died,
the survivor-vs-died comparison is not evidence of anything.

Cohort: adults (>=18) at ICU admission, each patient's first ICU stay,
admission carrying a cardiac ICD-9 code (410-412 MI, 414 CAD, 427 arrhythmia,
428 heart failure), still in the ICU and alive at 24 h. Label:
ADMISSIONS.hospital_expire_flag. Features: vitals and labs in
[ICU intime, intime + 24h].
"""
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else
        Path(os.environ.get("SMART_HEALTH_DATA", "~/physionet-data"))
        / "mimiciii-demo").expanduser()
if not ROOT.is_dir():
    sys.exit(f"MIMIC-III Demo not found at {ROOT}. Download it from "
             "https://physionet.org/content/mimiciii-demo/1.4/ and unzip it there.")


def load(name, **kw):
    hits = [p for p in ROOT.rglob("*.csv") if p.stem.upper() == name]
    assert len(hits) == 1, f"{name}: {hits}"
    df = pd.read_csv(hits[0], **kw)
    df.columns = [c.lower() for c in df.columns]
    return df


CARDIAC = {"410": "MI", "411": "MI", "412": "MI", "414": "CAD",
           "427": "Arrhythmia", "428": "Heart failure"}

# ITEMIDs follow MIT-LCP mimic-code concepts; labels are printed below as a check.
VITALS = {
    "heart_rate": [211, 220045],
    "sbp": [51, 442, 455, 6701, 220179, 220050],
    "dbp": [8368, 8440, 8441, 8555, 220180, 220051],
    "mbp": [456, 52, 6702, 443, 220052, 220181, 225312],
    "resp_rate": [615, 618, 220210, 224690],
    "spo2": [646, 220277],
    "temp_c": [223762, 676],
    "temp_f": [223761, 678],
}
LABS = {
    "troponin_t": [51003], "troponin_i": [51002], "ck_mb": [50911],
    "creatinine": [50912], "potassium": [50971, 50822], "sodium": [50983, 50824],
    "bnp": [50963], "lactate": [50813], "glucose": [50931, 50809],
    "hemoglobin": [51222, 50811], "wbc": [51301, 51300],
}
# Physiologic plausibility ranges (after unit conversion).
RANGES = {
    "heart_rate": (0, 300), "sbp": (0, 400), "dbp": (0, 300), "mbp": (0, 300),
    "resp_rate": (0, 80), "spo2": (0, 100), "temp_c": (25, 45),
    "troponin_t": (0, 100), "troponin_i": (0, 1000), "ck_mb": (0, 1000),
    "creatinine": (0, 30), "potassium": (1, 12), "sodium": (90, 200),
    "bnp": (0, 100000), "lactate": (0, 40), "glucose": (10, 2000),
    "hemoglobin": (1, 25), "wbc": (0, 500),
}


def section(t):
    print(f"\n=== {t} ===")


patients = load("PATIENTS", parse_dates=["dob"])
adm = load("ADMISSIONS", parse_dates=["admittime", "dischtime", "deathtime"])
icu = load("ICUSTAYS", parse_dates=["intime", "outtime"])
dx = load("DIAGNOSES_ICD", dtype={"icd9_code": str})
d_items = load("D_ITEMS")
d_lab = load("D_LABITEMS")

section("Tables")
for n, df in [("PATIENTS", patients), ("ADMISSIONS", adm), ("ICUSTAYS", icu),
              ("DIAGNOSES_ICD", dx)]:
    print(f"{n:15s} {len(df):>7,} rows")

# ---------------------------------------------------------------- cohort
section("Cohort flow")
flow = []
st = icu.merge(patients[["subject_id", "gender", "dob"]], on="subject_id")
flow.append(("All ICU stays", len(st), st.subject_id.nunique()))

# Age from year arithmetic: shifted DOBs for >89 overflow ns timestamps.
st["age"] = (st.intime.dt.year - st.dob.dt.year) - (
    (st.intime.dt.month * 100 + st.intime.dt.day)
    < (st.dob.dt.month * 100 + st.dob.dt.day)).astype(int)
st["age"] = st.age.astype(float)
st["age_over_89"] = st.age > 89
st.loc[st.age_over_89, "age"] = 91.4  # MIMIC convention: shifted to ~300
st = st[st.age >= 18]
flow.append(("Adults (>= 18)", len(st), st.subject_id.nunique()))

st = st.sort_values("intime").groupby("subject_id", as_index=False).first()
flow.append(("First ICU stay per patient", len(st), st.subject_id.nunique()))

dx["prefix"] = dx.icd9_code.str[:3]
card = dx[dx.prefix.isin(CARDIAC)].copy()
card["group"] = card.prefix.map(CARDIAC)
cardiac_hadm = set(card.hadm_id)
st = st[st.hadm_id.isin(cardiac_hadm)]
flow.append(("With a cardiac ICD-9 code", len(st), st.subject_id.nunique()))

st = st.merge(adm[["hadm_id", "hospital_expire_flag", "admission_type",
                   "deathtime"]], on="hadm_id")
los_h = (st.outtime - st.intime).dt.total_seconds() / 3600
died_in_24h = st.deathtime.notna() & (
    (st.deathtime - st.intime).dt.total_seconds() / 3600 < 24)
st["short"] = (los_h < 24) | died_in_24h
cohort = st[~st.short].copy()
flow.append(("In ICU and alive at 24 h", len(cohort), cohort.subject_id.nunique()))

for name, n, s in flow:
    print(f"{name:32s} {n:>4} stays  ({s} patients)")

section("Cardiac groups in the final cohort (a stay can have several)")
g = card[card.hadm_id.isin(cohort.hadm_id)].groupby("group").hadm_id.nunique()
for k, v in g.sort_values(ascending=False).items():
    print(f"{k:15s} {v:>3}")

section("Label: in-hospital mortality")
vc = cohort.hospital_expire_flag.value_counts().sort_index()
n = len(cohort)
for k, v in vc.items():
    print(f"{'died' if k == 1 else 'survived':9s} {v:>3}  ({v / n:.0%})")

section("Patient details")
print(f"age: median {cohort.age.median():.0f}, range {cohort.age.min():.0f}-"
      f"{cohort.age.max():.0f}; over 89 (age masked by MIMIC): "
      f"{int(cohort.age_over_89.sum())}")
print("sex:", cohort.gender.value_counts().to_dict())
print("admission type:", cohort.admission_type.value_counts().to_dict())
print("ICU first care unit:", cohort.first_careunit.value_counts().to_dict())
print("charting system:", cohort.dbsource.value_counts().to_dict())

# ---------------------------------------------------------------- events
section("ITEMID check (labels from the dictionaries)")
for k, ids in VITALS.items():
    labs = d_items[d_items.itemid.isin(ids)].label.tolist()
    print(f"{k:11s} {labs}")
for k, ids in LABS.items():
    labs = d_lab[d_lab.itemid.isin(ids)].label.tolist()
    print(f"{k:11s} {labs}")

all_vital_ids = sorted({i for v in VITALS.values() for i in v})
all_lab_ids = sorted({i for v in LABS.values() for i in v})
win = cohort[["icustay_id", "hadm_id", "subject_id", "intime"]].copy()
win["end"] = win.intime + pd.Timedelta(hours=24)

chunks = []
for ch in pd.read_csv(next(p for p in ROOT.rglob("*.csv")
                           if p.stem.upper() == "CHARTEVENTS"),
                      usecols=lambda c: c.lower() in {
                          "icustay_id", "itemid", "charttime", "valuenum",
                          "error"},
                      chunksize=200_000):
    ch.columns = [c.lower() for c in ch.columns]
    ch = ch[ch.itemid.isin(all_vital_ids) & ch.icustay_id.isin(win.icustay_id)]
    chunks.append(ch)
ce = pd.concat(chunks)
ce["charttime"] = pd.to_datetime(ce.charttime)
n_err = int((ce.get("error", pd.Series(dtype=float)) == 1).sum())
ce = ce[ce.get("error", 0) != 1]
ce = ce.merge(win[["icustay_id", "intime", "end"]], on="icustay_id")
ce = ce[(ce.charttime >= ce.intime) & (ce.charttime < ce.end)]
inv = {i: k for k, ids in VITALS.items() for i in ids}
ce["var"] = ce.itemid.map(inv)
f = ce["var"] == "temp_f"
ce.loc[f, "valuenum"] = (ce.loc[f, "valuenum"] - 32) * 5 / 9
ce.loc[f, "var"] = "temp_c"

le = load("LABEVENTS", parse_dates=["charttime"])
le = le[le.itemid.isin(all_lab_ids) & le.hadm_id.isin(win.hadm_id)]
# Labs belong to the admission; keep those inside the ICU 24-h window.
le = le.merge(win[["hadm_id", "icustay_id", "intime", "end"]], on="hadm_id")
le = le[(le.charttime >= le.intime) & (le.charttime < le.end)]
invl = {i: k for k, ids in LABS.items() for i in ids}
le["var"] = le.itemid.map(invl)

ev = pd.concat([ce[["icustay_id", "var", "charttime", "valuenum"]],
                le[["icustay_id", "var", "charttime", "valuenum"]]])
ev = ev.dropna(subset=["valuenum"])

section("Cleaning")
print(f"chart rows flagged as errors and dropped: {n_err}")
out = []
for v, (lo, hi) in RANGES.items():
    m = (ev["var"] == v) & ((ev.valuenum < lo) | (ev.valuenum > hi))
    if m.sum():
        out.append(f"{v}: {int(m.sum())}")
    ev = ev[~m]
print("values outside physiologic range, dropped:", ", ".join(out) or "none")
print(f"measurements kept in the 24-h windows: {len(ev):,}")

# ---------------------------------------------------------------- features
section("Coverage: share of stays with at least one value in the first 24 h")
ev = ev.sort_values("charttime")
agg = ev.groupby(["icustay_id", "var"]).valuenum.agg(
    ["count", "mean", "min", "max", "last", "std"])
feat = agg.unstack("var")
order = list(RANGES)
cov = {v: (feat[("count", v)].notna().sum() / n if ("count", v) in feat else 0.0)
       for v in order}
for v in order:
    if ("count", v) in feat:
        med_n = feat[("count", v)].median()
        print(f"{v:11s} {cov[v]:>5.0%}   median {med_n:>4.0f} readings per stay")
    else:
        print(f"{v:11s} {0:>5.0%}")

section("Median of per-stay mean value, survivors vs died (aggregates)")
lab = cohort.set_index("icustay_id").hospital_expire_flag
print(f"{'variable':11s} {'survived':>9s} {'died':>9s}")
for v in order:
    if ("mean", v) not in feat:
        continue
    col = feat[("mean", v)].dropna()
    y = lab.reindex(col.index)
    med = [f"{col[y == k].median():>9.2f}" if (y == k).any() else f"{'-':>9s}"
           for k in (0, 1)]
    print(f"{v:11s} {med[0]} {med[1]}   (n={len(col)})")

n_feat = sum(1 for c in feat.columns if c[0] in
             ("mean", "min", "max", "last", "std"))
section("Model-ready table")
print(f"{len(feat)} stays x {n_feat} summary features "
      f"(mean/min/max/last/std per variable) + {len(order)} missingness flags "
      f"+ age, sex, admission type")
