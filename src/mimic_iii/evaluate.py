"""Scoring the clinical model and writing its report.

The report states its data source at the top. On the MIMIC-III Demo the test
split is about eight patients drawn from 100 who had all died by the time
PhysioNet selected them, so every number in that report verifies that the
pipeline runs; none of them estimates clinical performance. The banner says so
in the report itself, not only in this docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common import metrics, provenance

from . import config, dataset, preprocessing

DEMO_BANNER = (
    "PIPELINE VERIFICATION ONLY - NOT CLINICAL PERFORMANCE. These numbers come "
    "from the MIMIC-III Demo (100 patients, all of whom had died by the time "
    "PhysioNet selected them). The test split has a handful of patients; the "
    "confidence intervals show how little the point estimates mean. Do not quote "
    "them as model performance."
)
RESEARCH_BANNER = (
    "Research artifact. Not clinically validated, not a medical device, not for "
    "patient care."
)


def score(model, table) -> np.ndarray:
    X, _ = dataset.xy(table)
    return model.predict_proba(X)[:, 1]


def evaluate_split(model, table, threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    _, y = dataset.xy(table)
    p = score(model, table)
    return {
        **metrics.binary_report(y, p, threshold=threshold),
        "auroc_ci": metrics.bootstrap_ci(y, p, "auroc", n_boot=n_boot, seed=seed),
        "auprc_ci": metrics.bootstrap_ci(y, p, "auprc", n_boot=n_boot, seed=seed),
        "reliability": metrics.reliability_bins(y, p, n_bins=5),
    }


def top_coefficients(model, k: int = 15) -> list[dict]:
    """Largest standardized logistic-regression weights, for interpretation."""
    names = preprocessing.output_feature_names(model)
    coef = model.named_steps["clf"].coef_.ravel()
    order = np.argsort(-np.abs(coef))[:k]
    return [{"feature": names[i], "coefficient": float(coef[i])} for i in order]


def build_report(source: str, task: config.TaskSpec, cohort_summary: dict,
                 split_counts: dict, selection: dict, results: dict,
                 coefficients: list[dict], inputs: dict) -> dict:
    is_demo = source == "demo"
    return {
        "title": "MIMIC-III clinical model: in-hospital mortality",
        "banner": DEMO_BANNER if is_demo else RESEARCH_BANNER,
        "research_disclaimer": RESEARCH_BANNER,
        "data_source": {"name": source, "is_demo": is_demo,
                        "description": config.SOURCE_DESCRIPTIONS[source]},
        "task": task.describe(),
        "cohort": cohort_summary,
        "split": split_counts,
        "selection": selection,
        "metrics": results,
        "top_coefficients": coefficients,
        "provenance": provenance.provenance(inputs),
    }


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict) -> str:
    lines = [f"# {report['title']}", "", f"> **{report['banner']}**", ""]
    src = report["data_source"]
    lines += [f"**Data source:** `{src['name']}` - {src['description']}", ""]

    task = report["task"]
    lines += ["## Task", "",
              f"- Target: {task['label']}",
              f"- Index time: {task['index_time']}",
              f"- Vitals window: {task['vitals_window']}; labs window: {task['labs_window']}",
              f"- Horizon: {task['horizon']}",
              f"- Exclusions: {', '.join(e for e in task['exclusions'] if e)}", ""]

    lines += ["## Cohort", "", "| Step | Stays | Patients |", "|---|---:|---:|"]
    for row in report["cohort"]["flow"]:
        lines.append(f"| {row['step']} | {row['stays']} | {row['subjects']} |")
    c = report["cohort"]
    lines += ["", f"Final cohort: {c['n_stays']} stays, {c['n_positive']} deaths "
                  f"({c['n_positive'] / max(c['n_stays'], 1):.0%}).", ""]

    lines += ["## Split (by patient)", "", "| Split | Patients | Stays | Deaths |",
              "|---|---:|---:|---:|"]
    for name, row in report["split"].items():
        lines.append(f"| {name} | {row['patients']} | {row['stays']} | {row['positive']} |")

    sel = report["selection"]
    lines += ["", "## Model selection (validation split only)", "",
              f"- Selected C = {sel['selected_C']} ({sel['selection_note']})",
              f"- Threshold = {sel['threshold']:.3f} ({sel['threshold_note']})", ""]

    lines += ["## Metrics", "",
              "`train` is in-sample. `val` chose C and the threshold, so it is not held out "
              "either. Only `test` was untouched until this evaluation.", "",
              "| Split | n | Deaths | AUROC (95% CI) | AUPRC (95% CI) | Brier | Sens | Spec | PPV |",
              "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for name, m in report["metrics"].items():
        def ci(key):
            b = m[f"{key}_ci"]
            if b["lower"] is None:
                return f"{_fmt(m[key])} (CI undefined)"
            return f"{_fmt(m[key])} ({b['lower']:.2f}-{b['upper']:.2f})"
        lines.append(
            f"| {name} | {m['n']} | {m['n_positive']} | {ci('auroc')} | {ci('auprc')} | "
            f"{_fmt(m['brier'])} | {_fmt(m['sensitivity'])} | {_fmt(m['specificity'])} | "
            f"{_fmt(m['ppv'])} |")
    undefined = {n: m["undefined"] for n, m in report["metrics"].items() if m["undefined"]}
    if undefined:
        lines += ["", "Undefined metrics (not computable on these rows):", ""]
        for name, reasons in undefined.items():
            for k, why in reasons.items():
                lines.append(f"- {name} {k}: {why}")

    lines += ["", "## Largest standardized coefficients", "",
              "| Feature | Coefficient |", "|---|---:|"]
    for row in report["top_coefficients"]:
        lines.append(f"| `{row['feature']}` | {row['coefficient']:+.3f} |")

    prov = report["provenance"]
    lines += ["", "## Provenance", "",
              f"- Generated: {prov['generated_utc']}",
              f"- Git commit: {prov['git']['commit']} (dirty: {prov['git']['dirty']})",
              f"- Environment: {prov['environment']}", "",
              f"_{report['research_disclaimer']}_", ""]
    return "\n".join(lines)


def write_report(out_dir: Path | str, report: dict) -> dict[str, Path]:
    out_dir = Path(out_dir)
    json_path = provenance.write_json(out_dir / "report.json", report)
    md_path = out_dir / "report.md"
    md_path.write_text(render_markdown(report))
    return {"json": json_path, "markdown": md_path}
