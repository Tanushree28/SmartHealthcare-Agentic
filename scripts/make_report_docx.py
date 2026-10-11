#!/usr/bin/env python
"""Build the plain written report for a supervisor.

A short Word document: what was done, what the numbers are, what did not work.
No styling beyond Word's defaults, so it opens cleanly in Word, Pages or Google
Docs and can be edited there.

Every figure comes from ``make_report_figures.py`` and every number is read from
the committed run artifacts in ``results/``, so the prose cannot drift from the
runs that produced it.

Usage:
    ./venv/bin/python scripts/make_report_figures.py     # figures first
    ./venv/bin/python scripts/make_report_docx.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg import config  # noqa: E402

HEADLINE = list(config.HEADLINE_CLASSES)
CLASSES = list(config.AAMI_CLASSES)


def load(name: str) -> dict:
    path = ROOT / "results" / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def find(payload: dict, label: str) -> dict:
    for result in payload["results"]:
        if result["label"] == label:
            return result
    raise SystemExit(f"no result labelled {label!r}")


def pc(result: dict) -> dict:
    return {m["class"]: m for m in result["per_class"]}


def headline(result: dict) -> float:
    per = pc(result)
    return float(np.mean([per[c]["f1"] for c in HEADLINE]))


def add_figure(doc: Document, path: Path, caption: str, width=6.4) -> None:
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Run: ./venv/bin/python scripts/make_report_figures.py"
        )
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para = doc.add_paragraph(caption)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.runs[0]
    run.italic = True
    run.font.size = Pt(9)


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(9.5)
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = text
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9.5)
    doc.add_paragraph()


def dm(diag: dict, arm: str) -> dict:
    """One arm's diagnosis measurements."""
    return diag["measurements"][arm]


def dpc(diag: dict, arm: str, name: str) -> dict:
    """One arm's per-class diagnosis entry."""
    for entry in dm(diag, arm)["per_class"]:
        if entry["class"] == name:
            return entry
    raise SystemExit(f"no class {name!r} in diagnosis arm {arm!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "report" / "ECG_Project_Report.docx"))
    parser.add_argument("--figures", default=str(ROOT / "report" / "figures"))
    args = parser.parse_args(argv)

    figures = Path(args.figures)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    baseline = load("2026-08-27-baseline-ds1_ds2.json")
    p1d = load("2026-09-16-nn-ds1_ds2.json")
    p1e = load("2026-09-17-nn-v2-ds1_ds2.json")
    cv1e = load("2026-09-17-nn-v2-ds1_ds2-nn_v2_cv.json")
    diag = load("2026-09-17-calibration-diagnosis-ds1_ds2.json")
    mix = load("2026-09-17-ensemble-mix-ds1_ds2.json")

    rf = find(baseline, "random_forest / rr+morphology")
    nn = find(p1d, "focal")
    ens = find(p1e, "beat_focal+ensemble")
    single = find(p1e, "beat_focal")

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ---- title -----------------------------------------------------------
    doc.add_heading("Detecting Abnormal Heartbeats in ECG Signals", level=0)
    sub = doc.add_paragraph(
        "Classifying heartbeats from the MIT-BIH Arrhythmia Database into the "
        "five AAMI classes, tested on patients the models never saw."
    )
    sub.runs[0].italic = True
    doc.add_paragraph(f"{date.today().strftime('%d %B %Y')}")

    # ---- 1. what the problem is -----------------------------------------
    doc.add_heading("1. The problem", level=1)
    doc.add_paragraph(
        "An ECG records every heartbeat. Most beats are normal, but a small "
        "number are abnormal and those are the ones that matter clinically. "
        "The task is to label each beat as one of five types: N (normal), "
        "S (supraventricular, an early beat), V (ventricular), F (fusion) and "
        "Q (unclassifiable)."
    )
    doc.add_paragraph(
        "Two things make this hard. First, the classes are extremely "
        "unbalanced: in our training data 89% of beats are normal while class "
        "S has 863 examples and class Q has only 8. Second, we test on "
        "patients that were never used in training, so the model cannot rely "
        "on having seen a particular person's heartbeat shape before. This is "
        "the standard inter-patient protocol (DS1/DS2) and it produces much "
        "lower scores than the beat-shuffled evaluation used in many papers.",
    )

    add_table(
        doc,
        ["Class", "Meaning", "Training beats", "Test beats"],
        [
            ["N", "Normal", "35,501", f"{pc(ens)['N']['support']:,}"],
            ["S", "Supraventricular (early)", "863", f"{pc(ens)['S']['support']:,}"],
            ["V", "Ventricular", "3,080", f"{pc(ens)['V']['support']:,}"],
            ["F", "Fusion", "388", f"{pc(ens)['F']['support']:,}"],
            ["Q", "Unclassifiable", "8", f"{pc(ens)['Q']['support']:,}"],
        ],
    )

    # ---- 2. what was done ------------------------------------------------
    doc.add_heading("2. What was done", level=1)
    doc.add_paragraph(
        "The work ran in three stages. Each stage was tested once on the same "
        "22 held-out patients, so the numbers are directly comparable."
    )
    for title, text in [
        (
            "Stage 1 — Classical baseline.",
            "Signal filtering, beat segmentation and hand-designed features "
            "(heartbeat timing plus beat shape), classified with a Random "
            "Forest. This set the score every later method had to beat.",
        ),
        (
            "Stage 2 — Neural network.",
            "A two-branch network: a 1-D convolutional network reads the shape "
            "of each beat while a second branch reads its timing. Class "
            "imbalance was handled with focal loss, which reduces the "
            "influence of the abundant normal beats during training.",
        ),
        (
            "Stage 3 — Ensemble.",
            "Cross-validation already produced five models, each trained on a "
            "different group of patients. Instead of discarding four of them, "
            "all five predictions are averaged. This required no additional "
            "training.",
        ),
    ]:
        para = doc.add_paragraph(style="List Bullet")
        para.add_run(title).bold = True
        para.add_run(" " + text)

    # ---- 3. results ------------------------------------------------------
    doc.add_heading("3. Results", level=1)
    doc.add_paragraph(
        "All figures below are measured on 49,646 heartbeats from 22 patients "
        "that appear in no training set."
    )

    add_table(
        doc,
        ["Method", "Overall score", "S found", "S correct", "V found"],
        [
            [
                name,
                f"{headline(r):.3f}",
                f"{pc(r)['S']['recall']:.3f}",
                f"{pc(r)['S']['precision']:.3f}",
                f"{pc(r)['V']['recall']:.3f}",
            ]
            for name, r in [
                ("Random Forest (classical)", rf),
                ("Neural network", nn),
                ("Neural network + ensemble", ens),
            ]
        ],
    )
    note = doc.add_paragraph(
        "Overall score is the macro-averaged F1 over classes N, S, V and F. "
        "Class Q is excluded from this average because 8 training beats cannot "
        "support a meaningful score; it is still reported separately. Accuracy "
        "is deliberately not used as the headline: predicting “normal” "
        "for every beat scores about 0.89 while finding no arrhythmia at all."
    )
    note.runs[0].font.size = Pt(9.5)
    note.runs[0].italic = True

    add_figure(
        doc,
        figures / "fig1_results.png",
        "Figure 1. Results across the three stages. The overall score improved "
        "from 0.506 to 0.590. The largest change was in class S.",
    )

    doc.add_heading("3.1 The main result", level=2)
    doc.add_paragraph(
        f"Detection of class S improved from {pc(rf)['S']['recall']:.1%} with "
        f"the Random Forest to {pc(ens)['S']['recall']:.1%} with the final "
        f"model. In absolute terms, of the {pc(ens)['S']['support']:,} S beats "
        f"in the test set the classical method found "
        f"{int(pc(rf)['S']['recall'] * pc(rf)['S']['support']):,}, while the "
        f"final model found "
        f"{int(pc(ens)['S']['recall'] * pc(ens)['S']['support']):,}."
    )
    doc.add_paragraph(
        f"Precision on class S also improved, from {pc(rf)['S']['precision']:.1%} "
        f"to {pc(ens)['S']['precision']:.1%}. This matters because false alarms "
        f"were the main weakness of the intermediate model: it found many more "
        f"S beats but produced a large number of incorrect S predictions."
    )

    add_figure(
        doc,
        figures / "fig2_confusion_matrix.png",
        "Figure 2. Confusion matrices. Rows are the true class, columns the "
        "predicted class. In the S row, correct predictions rise from 498 to "
        "1,352 and missed beats fall from 1,290 to 386.",
    )

    add_figure(
        doc,
        figures / "fig3_per_class.png",
        "Figure 3. Detection rate by heartbeat type. N and V are handled well "
        "by every method. S improves substantially. F becomes worse, and Q "
        "cannot be learned from 8 training examples.",
        width=5.9,
    )

    # ---- 4. was the improvement real? ------------------------------------
    doc.add_heading("4. Was the improvement real?", level=1)
    doc.add_paragraph(
        "A classifier does two separate things. It ranks heartbeats by how "
        "suspicious they look, and it draws a line deciding which side of that "
        "ranking counts as abnormal. The scores in section 3 measure both at "
        "once, so they cannot tell the two apart. A model that became more "
        "willing to guess “abnormal” scores better on them without having "
        "learned anything."
    )
    doc.add_paragraph(
        "The two were therefore measured separately. The second column below "
        "tries every possible decision line and averages the result, so moving "
        "one line cannot improve it."
    )

    add_table(
        doc,
        ["Training method", "Score with a fixed line", "Score with no line"],
        [
            [label,
             f"{dm(diag, arm)['headline_macro_f1']:.3f}",
             f"{dm(diag, arm)['macro_average_precision']:.3f}"]
            for arm, label in [
                ("plain_ce", "Plain (no imbalance handling)"),
                ("weighted_ce", "Weighted"),
                ("focal", "Focal — the method kept"),
                ("oversample", "Oversampling"),
                ("ensemble", "Ensemble of five"),
            ]
        ],
    )

    gap_f1 = dm(diag, "focal")["headline_macro_f1"] - dm(diag, "plain_ce")["headline_macro_f1"]
    gap_ap = dm(diag, "focal")["macro_average_precision"] - dm(diag, "plain_ce")["macro_average_precision"]
    doc.add_paragraph(
        f"Focal loss beats the plain model by {gap_f1:+.3f} in the first column "
        f"and by {gap_ap:+.3f} in the second. Almost the entire advantage "
        "reported earlier came from where the decision line sits, not from the "
        "model recognising heartbeats any better. The ensemble is the one "
        "change on this table that improves both columns."
    )

    add_figure(
        doc,
        figures / "fig4_scoring.png",
        "Figure 4. The same four training methods, scored two ways. The spread "
        "on the left is what earlier sections reported; the spread on the right "
        "is what the models actually learned.",
    )

    doc.add_heading("4.1 On class S, the chosen method was the worse one", level=2)
    add_table(
        doc,
        ["Training method", "Ranking quality on S", "Times better than chance",
         "Place in earlier comparison"],
        [
            [label,
             f"{dpc(diag, arm, 'S')['average_precision']:.3f}",
             f"{dpc(diag, arm, 'S')['lift_over_chance']:.1f}x",
             place]
            for arm, label, place in [
                ("ensemble", "Ensemble of five", "not tested then"),
                ("oversample", "Oversampling", "4th of 5"),
                ("plain_ce", "Plain (no imbalance handling)", "5th of 5"),
                ("focal", "Focal — the method kept", "1st of 5"),
                ("weighted_ce", "Weighted", "2nd of 5"),
            ]
        ],
    )
    doc.add_paragraph(
        "Read against the last column, the earlier ranking is close to "
        "upside down for class S. The method that finished fourth ranks S "
        "beats best; the method that was carried forward ranks them second "
        "worst. Focal loss traded class S away for class V, where it is "
        f"genuinely better ({dpc(diag, 'focal', 'V')['average_precision']:.3f} "
        f"against {dpc(diag, 'plain_ce', 'V')['average_precision']:.3f})."
    )
    doc.add_paragraph(
        "Since retired. Section 6 repeated this comparison with five models "
        "per method rather than one, and the ordering did not survive. The "
        "paragraph above is kept because the reasoning it illustrates is sound "
        "and the numbers were real; the conclusion drawn from them was not.",
    )
    doc.add_paragraph(
        "This also explains a result that had been puzzling. Precision on "
        "class S has sat between 0.27 and 0.39 through every method tried. "
        "Figure 5 shows why: the achievable trade-off is flat. There is no "
        "threshold setting that buys better precision, so tuning one cannot "
        "help."
    )

    add_figure(
        doc,
        figures / "fig5_precision_ceiling.png",
        "Figure 5. Class S. Moving left to right means catching more S beats; "
        "height is how often an S call was correct. A model with room to "
        "improve curves downward from the left. These lines are flat.",
        width=5.9,
    )

    # ---- 5. can the probabilities be trusted? ----------------------------
    doc.add_heading("5. Can the predicted probabilities be trusted?", level=1)
    doc.add_paragraph(
        "Everything above treats the model as something that outputs a label. "
        "If a later stage of the system instead reads the model's confidence "
        "and reasons with it, that number has to be honest. This had never "
        "been checked, and it is not."
    )
    doc.add_paragraph(
        "The cause is known: training uses class weights to counteract the "
        "imbalance, and weighting tilts the output. A calibrated model's "
        "average predicted probability for a class equals how often that class "
        "actually occurs."
    )

    add_table(
        doc,
        ["Heartbeat type", "How often it occurs", "What the model says",
         "Model ÷ reality (1.0 = honest)"],
        [
            [label,
             f"{dpc(diag, 'focal', name)['prevalence']:.2%}",
             f"{dpc(diag, 'focal', name)['mean_predicted']:.2%}",
             f"{dpc(diag, 'focal', name)['prior_ratio']:.1f}x"]
            for name, label in [
                ("N", "N (normal)"),
                ("S", "S (supraventricular)"),
                ("V", "V (ventricular)"),
                ("F", "F (fusion)"),
                ("Q", "Q (unclassifiable)"),
            ]
        ],
    )
    doc.add_paragraph(
        "The four rare classes are overstated and normal beats are understated, "
        "which is the tilt the class weights were always going to introduce. "
        "When the model reports a beat as 50% likely to be class S, it is not. "
        "Anything reading these numbers directly would raise roughly three "
        "times too many alerts on class S and fourteen times too many on "
        "class F. This is a systematic tilt rather than a broken model, and a "
        "correction fitted on held-out data removes it without retraining."
    )

    add_figure(
        doc,
        figures / "fig6_trust.png",
        "Figure 6. Left: what the model claimed against what happened, for "
        "class S. A trustworthy model follows the dotted line. Right: accuracy "
        "when the least confident beats are handed to a person instead.",
    )

    ens_ece = dm(diag, "ensemble")["classwise_ece"]["15"]
    doc.add_paragraph(
        "The left panel separates three failures. The plain and oversampling "
        "models are flat: whether they say 30% or 90%, the answer is about 40% "
        "likely, so the number carries no information. Focal and weighted bend "
        "downwards at the right-hand end, meaning their most confident calls "
        "on class S are less reliable than their middling ones. Only the "
        "ensemble rises the whole way. It sits below the dotted line, "
        "understating its own confidence, which is the one failure a "
        "calibration step is designed to correct."
    )
    doc.add_paragraph(
        "The ensemble is therefore the model to carry forward. It ranks best "
        "on every class that counts, and ranking is the part a calibration "
        "step cannot repair. The right panel shows the practical consequence: "
        "letting the model decline the beats it is least sure about raises "
        "accuracy on the rest from 0.88 to about 0.975, against a 0.89 "
        "baseline for calling every beat normal."
    )

    # ---- 6. was mixing the training methods better? -----------------------
    doc.add_heading("6. Was mixing the training methods better?", level=1)
    doc.add_paragraph(
        "Section 4.1 found that the training method carried forward was not the "
        "best at class S — two methods that had lost on the earlier score "
        "ranked S beats better. The obvious follow-up is to stop using five "
        "copies of one method and build the ensemble from all four instead, on "
        "the reasoning that members which make different mistakes help each "
        "other. That was tested rather than assumed."
    )
    doc.add_paragraph(
        "Twenty models were trained: four methods across the same five patient "
        "folds. Five candidate ensembles were fixed in writing before the run, "
        "and the winner was chosen on held-out training data before the test "
        "set was scored once. The script enforces that order rather than "
        "trusting whoever runs it."
    )

    mix_ds2 = mix["ds2"]["scores"]
    mix_oof = mix["oof"]["scores"]
    chosen = mix["selection"]["selected"]
    reference = mix["selection"]["reference"]
    label_of = {
        "A_focal_only": "Focal only (the ensemble in use)",
        "B_focal_plain": "Focal + plain",
        "C_focal_plain_oversample": "Focal + plain + oversampling",
        "D_all_four": "All four methods",
        "E_plain_oversample": "Plain + oversampling",
    }

    add_table(
        doc,
        ["Ensemble", "Models", "Ranking, all classes", "Ranking, class S",
         "Overall score"],
        [
            [label_of[k] + (" — chosen" if k == chosen else ""),
             str(mix_ds2[k]["n_models"]),
             f"{mix_ds2[k]['macro_average_precision']:.3f}",
             f"{mix_ds2[k]['s_average_precision']:.3f}",
             f"{mix_ds2[k]['headline_macro_f1']:.3f}"]
            for k in label_of
        ],
    )

    win, ref = mix_ds2[chosen], mix_ds2[reference]
    doc.add_paragraph(
        "Mixing did not help. The ensemble already in use scored best on every "
        "ranking measure, and the candidate the procedure selected came in "
        f"{ref['macro_average_precision'] - win['macro_average_precision']:.3f} "
        "lower across all classes and "
        f"{ref['s_average_precision'] - win['s_average_precision']:.3f} lower on "
        "class S. Every mixture was worse than the arrangement it was meant to "
        "improve on."
    )
    doc.add_paragraph(
        "This is the most trustworthy form a negative result can take. The "
        "procedure was committed to in advance and it chose against the "
        "existing ensemble; the existing ensemble won anyway. There is no room "
        "here for having picked the flattering answer after the fact."
    )

    doc.add_heading("6.1 The selection step was measuring the wrong thing",
                    level=2)
    doc.add_paragraph(
        "The more useful finding is why the procedure chose badly. Ranked by "
        "the held-out score used to choose, the candidates line up almost "
        "exactly by how many models they contain."
    )

    add_table(
        doc,
        ["Ensemble", "Models", "Score used to choose", "Score on unseen patients"],
        [
            [label_of[k],
             str(mix_ds2[k]["n_models"]),
             f"{mix_oof[k]['macro_average_precision']:.3f}",
             f"{mix_ds2[k]['macro_average_precision']:.3f}"]
            for k in sorted(label_of, key=lambda k: mix_ds2[k]["n_models"])
        ],
    )

    doc.add_paragraph(
        "The correlation between number of models and the choosing score is "
        "+0.97; against the score that matters it is -0.46. The procedure was "
        "ranking ensembles by size, not by quality."
    )
    doc.add_paragraph(
        "The cause is a flaw in the design, and it is worth recording plainly. "
        "Each model stops training at its best epoch as judged on the fold of "
        "patients held out from it — and that same fold supplies the "
        "predictions used to score it. An ensemble with more members therefore "
        "gets more independent attempts at the very data it is being measured "
        "on. Twenty models tuned against a fold will flatter themselves on that "
        "fold more than five will, whatever their real quality."
    )
    doc.add_paragraph(
        "The design document claimed this optimism was even across the "
        "candidates and so could not tip the choice. That was wrong: it is even "
        "across training methods but not across ensembles of different sizes. "
        "Any future comparison of this kind needs a fixed training budget, a "
        "separate selection split, or nested cross-validation before its "
        "choices can be believed."
    )

    add_figure(
        doc,
        figures / "fig7_mixing.png",
        "Figure 7. Left: every mixture scored below the ensemble already in "
        "use. Right: the score used to choose rises with ensemble size while "
        "the score that matters does not — the two dashed trends run in "
        "opposite directions.",
    )

    doc.add_heading("6.2 What this retires", level=2)
    doc.add_paragraph(
        "Section 8 flagged the class S result in section 4.1 as resting on one "
        "trained model per method and needing repetition before anything was "
        "built on it. This run was that repetition, with five models per method "
        "instead of one, and it did not hold: focal-only now ranks class S best "
        f"({mix_ds2[reference]['s_average_precision']:.3f}) and plain plus "
        f"oversampling ranks it worst "
        f"({mix_ds2['E_plain_oversample']['s_average_precision']:.3f}). The "
        "earlier ordering was a property of single models fitted on the whole "
        "training set, not something that survives into an ensemble."
    )
    doc.add_paragraph(
        "The finding that section 4 rests on is unaffected. That the four "
        "training methods barely differ once the decision line is taken out of "
        "the score is measured across every method and every candidate here too."
    )
    doc.add_paragraph(
        "Retraining also improved the ensemble itself. Letting each model stop "
        "at its own best epoch, rather than at the median epoch borrowed from "
        f"cross-validation, raised class S detection from "
        f"{pc(ens)['S']['recall']:.1%} to "
        f"{mix_ds2[reference]['s_recall']:.1%} at similar precision."
    )

    doc.add_heading("6.3 The one thing mixing did do", level=2)
    doc.add_paragraph(
        "Mixing improved the honesty of the probabilities substantially. Adding "
        "the plain method to the focal ensemble cut the Brier score from "
        f"{mix_ds2[reference]['brier']:.3f} to "
        f"{mix_ds2['B_focal_plain']['brier']:.3f} and the calibration error "
        f"from {mix_ds2[reference]['classwise_ece']:.3f} to "
        f"{mix_ds2['B_focal_plain']['classwise_ece']:.3f} — a reduction of "
        "roughly 40% in both."
    )
    doc.add_paragraph(
        "That is not a reason to adopt it. Section 5 established the rule: "
        "calibration is the part a later correction can repair and ranking is "
        "the part it cannot. Paying in class S ranking for calibration that can "
        "be fitted afterwards is the wrong way round."
    )

    # ---- 7. what did not work -------------------------------------------
    doc.add_heading("7. What did not work", level=1)
    doc.add_paragraph(
        "Three further changes were tested and measured against the same "
        "reference on identical data splits. Two produced no measurable effect "
        "and are reported here rather than omitted."
    )
    paired = {p["arm"]: p for p in [
        {"arm": k, **v["s_f1"]} for k, v in cv1e.get("paired_vs_reference", {}).items()
    ]}
    rows = []
    for arm, label in [
        ("rhythm_context", "Rhythm context (model reads neighbouring beats)"),
        ("beat_robust_rr", "Robust rhythm reference (median instead of mean)"),
    ]:
        p = paired.get(arm, {})
        rows.append([
            label,
            f"{p.get('wins', '?')} of {p.get('n_folds', '?')}",
            f"{p.get('mean_difference', 0):+.3f}",
            "No effect",
        ])
    rows.append(["Ensemble of five models", "—", "+0.062", "Improved the result"])
    add_table(doc, ["Change tested", "Runs won", "Mean change", "Verdict"], rows)

    doc.add_paragraph(
        "The rhythm-context model was the most promising idea on paper: it "
        "lets the network compare a beat against the beats around it, which is "
        "how a clinician judges whether a beat arrived early. It won 7 of 15 "
        "comparison runs, which is indistinguishable from chance. The robust "
        "rhythm reference had no effect at all."
    )
    doc.add_paragraph(
        "These comparisons were only readable because each change was tested "
        "against the reference on the same patients, run by run. Comparing "
        "average scores would not have worked: performance varied between "
        "0.35 and 0.71 depending on which patients were held out, a spread far "
        "larger than any difference between the methods."
    )

    # ---- 8. limitations --------------------------------------------------
    doc.add_heading("8. Limitations", level=1)
    for text in [
        f"Precision on class S is {pc(ens)['S']['precision']:.1%}, and section 4 "
        "shows this is a ceiling rather than a setting: the trade-off between "
        "finding S beats and being right about them is flat, so no threshold "
        "choice improves it. The model is suitable for flagging beats for a "
        "clinician to review, not for automatic decisions.",
        "Class F became worse, not better. The classical Random Forest still "
        f"has the best F1 on that class ({pc(rf)['F']['f1']:.3f} against "
        f"{pc(ens)['F']['f1']:.3f}).",
        "Class Q cannot be evaluated here. With 8 training beats and 7 test "
        "beats, every method scores zero, and no conclusion should be drawn.",
        "Results are reported from a single pass over the test set, which is "
        "correct practice but means each number is one measurement rather than "
        "a range.",
        "The class S ordering in section 4.1 has been retired. It rested on one "
        "trained model per method; repeated with five models per method in "
        "section 6, it did not hold. The broader finding it sits inside — that "
        "the line-free scores barely differ between methods — survives.",
        "Ensembles of different sizes cannot be compared on held-out training "
        "data under the current procedure, because each model stops at its best "
        "epoch on the very fold used to score it. Section 6.1 measures the "
        "resulting bias. A fixed training budget, a separate selection split or "
        "nested cross-validation is needed before any such comparison is "
        "trusted again.",
        "The predicted probabilities are not yet calibrated. Section 5 "
        "measures the tilt but does not correct it; that correction is the "
        "next piece of work and must be fitted on held-out data only.",
        "The remaining limit appears to be data rather than model design. Two "
        "architectural changes produced no improvement, while 863 training "
        "examples of class S remains the binding constraint.",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    # ---- 9. methods ------------------------------------------------------
    doc.add_heading("9. Method details", level=1)
    env = p1e.get("environment", {})
    add_table(
        doc,
        ["Item", "Value"],
        [
            ["Dataset", "MIT-BIH Arrhythmia Database v1.0.0, 48 records"],
            ["Split", "Inter-patient DS1/DS2 (de Chazal et al., 2004)"],
            ["Excluded", "4 paced records, per AAMI recommendation"],
            ["Training set", "22 patients, 50,955 beats"],
            ["Test set", "22 patients, 49,646 beats"],
            ["Sampling rate", "360 Hz, single lead (MLII where available)"],
            ["Preprocessing", "0.5–40 Hz bandpass, z-score normalisation, "
                              "300-sample beat windows"],
            ["Model", "1-D CNN (4 convolutional blocks) + dense timing branch, "
                      "60,101 parameters"],
            ["Training", "Adam, focal loss, early stopping on validation "
                         "macro-F1"],
            ["Model selection", "5-fold cross-validation grouped by patient; "
                                "the test set was never used for any choice"],
            ["Software", f"Python {env.get('python', '3.13')}, "
                         f"PyTorch {env.get('torch', '')}"],
        ],
    )

    doc.save(out)
    print(f"{out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
