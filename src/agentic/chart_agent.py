"""Chart agent: one ICU stay's first 24 hours, turned into findings.

Reads only what is known before the index time (ICU intime + 24 h): the
feature-table summaries the trained model also sees, the last lab values in
the same window, the reason for admission (recorded on arrival) and drugs
started on an earlier day. Never the label or any post-index column --
chart_data drops those before this module sees anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mimic_iii import config, features

from .chart_data import ChartSources
from .findings import Finding, FindingList
from .knowledge import Knowledge

UNITS = {v.name: v.unit for v in config.VARIABLES}
VITALS = tuple(v.name for v in config.VARIABLES if v.source == "chart")
LABELS = {
    "heart_rate": "heart rate", "sbp": "systolic blood pressure",
    "dbp": "diastolic blood pressure", "mbp": "mean arterial pressure",
    "resp_rate": "respiratory rate", "spo2": "oxygen saturation (SpO2)", "temp_c": "temperature",
    "troponin_t": "troponin T", "troponin_i": "troponin I", "ck_mb": "CK-MB",
    "creatinine": "creatinine", "potassium": "potassium", "sodium": "sodium", "bnp": "BNP",
    "lactate": "lactate", "glucose": "glucose", "hemoglobin": "hemoglobin",
    "wbc": "white blood cell count",
}
GROUP_LABELS = {"antiarrhythmic": "Antiarrhythmic", "beta_blocker": "Beta-blocker",
                "anticoagulant": "Anticoagulant", "diuretic": "Diuretic",
                "potassium_supplement": "Potassium supplement", "digoxin": "Digoxin",
                "vasopressor_inotrope": "Vasopressor/inotrope"}
MED_CAVEAT = ("dates only: a drug started on the day of the 24-hour mark is left out, "
              "because its time is unknown")


def _num(x) -> str:
    return f"{float(x):g}"


def _text(x, default: str = "not recorded") -> str:
    return default if pd.isna(x) else str(x)


def top_reasons(model, X: pd.DataFrame, k: int = 3) -> list[str]:
    """The k inputs that pushed the score most (coef x transformed value), as in
    scripts/mimic_iii/show_predictions.py; '+' raises the score, '-' lowers it."""
    names = [str(n) for n in model.named_steps["pre"].get_feature_names_out()]
    row = model.named_steps["pre"].transform(X)[0] * model.named_steps["clf"].coef_.ravel()
    top = np.argsort(-np.abs(row))[:k]
    return [f"{'+' if row[i] > 0 else '-'}{names[i]}" for i in top]


def _reason_terms(reasons: list[str]) -> set[str]:
    return {v for r in reasons for v in config.VARIABLE_NAMES if r[1:].startswith(v + "_")}


def _demographics(fl: FindingList, row) -> None:
    age = ">89" if int(row["age_over_89"]) == 1 else f"{float(row['age']):.0f}"
    sex = {"M": "male", "F": "female"}.get(_text(row["gender"]), "sex not recorded")
    fl.add("demographics", "",
           f"Age {age}, {sex}; {_text(row['admission_type']).lower()} admission; first ICU "
           f"unit {_text(row['first_careunit'])}",
           evidence={"age": age, "gender": _text(row["gender"]),
                     "admission_type": _text(row["admission_type"]),
                     "first_careunit": _text(row["first_careunit"])})


def _labs(fl: FindingList, labs: pd.DataFrame, knowledge: Knowledge) -> None:
    order = {v: i for i, v in enumerate(config.VARIABLE_NAMES)}
    labs = labs.assign(_o=labs["variable"].map(order)).sort_values("_o")
    for lab in labs.itertuples(index=False):
        var, value = lab.variable, float(lab.value)
        lo, hi = knowledge.lab_ranges.get(var, (None, None))
        direction = ("low" if lo is not None and value < lo
                     else "high" if hi is not None and value > hi else None)
        hospital = str(lab.flag).strip().lower() == "abnormal"
        if direction is None and not hospital:
            continue
        why = []
        if direction:
            why.append(f"outside the review range {_num(lo)} to {_num(hi)}")
        if hospital:
            why.append("flagged abnormal by the hospital lab")
        fl.add("lab_abnormal", f"{var}:{direction or 'flagged'}",
               f"Last {LABELS[var]} before the 24-hour mark: {_num(value)} {UNITS[var]} "
               f"({direction or 'abnormal'}; {' and '.join(why)})",
               value=float(_num(value)), unit=UNITS[var],
               evidence={"range": [lo, hi], "hospital_flag": hospital,
                         "direction": direction or "flagged"},
               terms=(var,))


def _vitals(fl: FindingList, row, knowledge: Knowledge) -> None:
    for var in VITALS:
        if var not in knowledge.vital_ranges or pd.isna(row[f"{var}_min"]):
            continue
        lo, hi = knowledge.vital_ranges[var]
        vmin, vmax = round(float(row[f"{var}_min"]), 1), round(float(row[f"{var}_max"]), 1)
        if vmin < lo:
            fl.add("vital_out_of_range", f"{var}:low",
                   f"Lowest {LABELS[var]} before the 24-hour mark: {_num(vmin)} {UNITS[var]} "
                   f"(review range {_num(lo)} to {_num(hi)})",
                   value=vmin, unit=UNITS[var], evidence={"min": vmin, "range": [lo, hi]},
                   terms=(var,))
        if vmax > hi:
            fl.add("vital_out_of_range", f"{var}:high",
                   f"Highest {LABELS[var]} before the 24-hour mark: {_num(vmax)} {UNITS[var]} "
                   f"(review range {_num(lo)} to {_num(hi)})",
                   value=vmax, unit=UNITS[var], evidence={"max": vmax, "range": [lo, hi]},
                   terms=(var,))


def _not_measured(fl: FindingList, row) -> None:
    missing = [v for v in config.VARIABLE_NAMES if int(row[f"{v}_missing"]) == 1]
    if missing:
        fl.add("not_measured", "",
               "Not measured before the 24-hour mark: " + ", ".join(LABELS[v] for v in missing),
               value=len(missing), unit="variables", evidence={"variables": missing},
               terms=tuple(missing))


def _medications(fl: FindingList, meds: pd.DataFrame, knowledge: Knowledge) -> None:
    names = sorted({str(d).strip() for d in meds["drug"]}, key=str.lower)
    grouped: dict[str, list[str]] = {}
    for group, patterns in knowledge.drug_groups.items():
        hits = [n for n in names if any(p in n.lower() for p in patterns)]
        if hits:
            grouped[group] = hits
    for group, hits in grouped.items():
        patterns = [p for p in knowledge.drug_groups[group] if any(p in h.lower() for h in hits)]
        fl.add("medication", group,
               f"{GROUP_LABELS.get(group, group)} started before the 24-hour mark: "
               f"{', '.join(hits)}",
               value=len(hits), unit="drugs", evidence={"drugs": hits},
               terms=("medication", group, *(f"drug:{p}" for p in patterns)),
               caveat=MED_CAVEAT)
    in_groups = {n for hits in grouped.values() for n in hits}
    statement = (f"{len(names)} different drugs started before the 24-hour mark "
                 f"({len(names) - len(in_groups)} outside the cardiac-relevant groups)"
                 if names else "No prescriptions found that started before the 24-hour mark")
    fl.add("medication_count", "", statement, value=len(names), unit="drugs",
           evidence={"drugs": names}, terms=("medication",), caveat=MED_CAVEAT)


def _risk(fl: FindingList, rows: pd.DataFrame, sources: ChartSources) -> None:
    X = rows[features.model_input_columns()]
    risk = float(sources.model.predict_proba(X)[0, 1])
    threshold = float(sources.meta["threshold"])
    flagged = bool(risk >= threshold)
    reasons = top_reasons(sources.model, X)
    split = sources.split_of.get(int(rows.iloc[0]["subject_id"]), "unknown")
    n_train = sources.meta.get("selection", {}).get("n_train")
    if sources.meta.get("is_demo"):
        caveat = ("MIMIC-III Demo model" + (f" trained on {n_train} patients" if n_train else "")
                  + "; not validated")
    else:
        caveat = "not validated for clinical use"
    if split == "train":
        caveat += "; this patient was in the model's training split, so the score is optimistic"
    fl.add("risk_score", "",
           f"Demo risk model score {risk:.2f} for in-hospital death (threshold {threshold:.2f}): "
           f"{'above' if flagged else 'below'} the threshold. Biggest pushes: {', '.join(reasons)}",
           value=round(risk, 2),
           evidence={"threshold": round(threshold, 2), "flagged": flagged, "reasons": reasons,
                     "split": split},
           terms=("risk_score", *sorted(_reason_terms(reasons))), caveat=caveat)


def analyze(stay: int, sources: ChartSources, knowledge: Knowledge) -> list[Finding]:
    rows = sources.table[sources.table["icustay_id"] == int(stay)]
    if rows.empty:
        raise KeyError(f"ICU stay {stay} is not in the cohort feature table")
    row = rows.iloc[0]
    fl = FindingList("chart")
    _demographics(fl, row)
    reason = sources.admission_reason.get(int(row["hadm_id"]))
    if reason:
        fl.add("admission_reason", "", f"Reason for admission, recorded on arrival: {reason}",
               value=reason, terms=("admission_reason",))
    _labs(fl, sources.last_labs[sources.last_labs["icustay_id"] == int(stay)], knowledge)
    _vitals(fl, row, knowledge)
    _not_measured(fl, row)
    _medications(fl, sources.medications[sources.medications["hadm_id"] == int(row["hadm_id"])],
                 knowledge)
    _risk(fl, rows, sources)
    return fl.items
