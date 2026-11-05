"""Run one case through every stage, yielding one event per stage.

Stage numbers match the page in PlanDoc/agentic_with_fe.md §14:
1 raw ECG, 2 cleaned, 3 beats found, 4 beats cut out, 5 model calls,
6 heart agent, 7 chart agent, 8 rules, 9 report. The CLI prints the events;
the Part B web app will stream them. A stage that cannot run yields an event
whose payload is {"skipped": reason}, so a reader always sees all nine.

The cardiologist's labels travel only in stage 5's ``display_only`` and in
``Case.display_only``; no agent, rule or prompt reads them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mimic_iii.errors import MimicError
from mit_bih import filters
from mit_bih import inference as ecg_inference

from . import chart_agent, ecg_window, heart_agent, rules, template_report
from .chart_data import ChartSources, load_chart_sources
from .findings import Case, ReportResult, make_banner
from .knowledge import DEFAULT_PATH, Knowledge, load_knowledge

STAGES = {1: "raw_ecg", 2: "cleaned", 3: "beats_found", 4: "beats_cut", 5: "model_calls",
          6: "heart_agent", 7: "chart_agent", 8: "rules", 9: "report"}
TRUTH_TOLERANCE_S = 0.15
OBSERVATION_HOURS = 24.0


@dataclass(frozen=True)
class CaseRequest:
    record: str | None = None
    start_s: float = 0.0
    seconds: float = 20.0
    stay: int | None = None
    detect_peaks: bool = False


@dataclass(frozen=True)
class StageEvent:
    stage: int
    name: str
    payload: dict

    @property
    def skipped(self) -> bool:
        return "skipped" in self.payload


@dataclass
class Deps:
    knowledge: Knowledge
    classifier: object | None = None  # mit_bih.inference.EcgClassifier
    chart: ChartSources | None = None
    record_dir: Path | None = None
    source: str = "demo"
    write_report: Callable[[Case], ReportResult] = template_report.template_report
    unavailable: dict = field(default_factory=dict)  # "ecg"/"chart" -> why it is not loaded


def _event(stage: int, payload: dict) -> StageEvent:
    return StageEvent(stage, STAGES[stage], payload)


def match_truth(peaks: np.ndarray, win: ecg_window.EcgWindow) -> list[str | None]:
    """The cardiologist's label for each classified beat (display only)."""
    if win.r_peaks.size == 0:
        return [None] * len(peaks)
    tol = TRUTH_TOLERANCE_S * win.fs
    out = []
    for p in peaks:
        j = int(np.argmin(np.abs(win.r_peaks - p)))
        out.append(win.truth[j] if abs(int(win.r_peaks[j]) - int(p)) <= tol else None)
    return out


def _ecg_stages(req: CaseRequest, deps: Deps, findings: list, display: dict, context: list):
    win = ecg_window.load_window(req.record, req.start_s, req.seconds, deps.record_dir)
    context += [float(req.record), float(req.start_s), float(req.seconds)]
    yield _event(1, {"record": win.record, "lead": win.lead, "fs": win.fs,
                     "start_s": win.start_s, "signal": win.signal})
    b = deps.classifier.bundle
    yield _event(2, {"signal": filters.denoise(win.signal, fs=b.fs, low=b.bandpass_low_hz,
                                               high=b.bandpass_high_hz, order=b.bandpass_order,
                                               use_notch=b.notch)})
    if req.detect_peaks:
        peaks, source = ecg_inference.detect_r_peaks(win.signal, win.fs), "detected"
    else:
        peaks, source = win.r_peaks, "annotation"
    yield _event(3, {"r_peaks": peaks, "source": source})
    pred = deps.classifier.predict_signal(win.signal, win.fs, r_peaks=peaks)
    p = pred.prepared
    yield _event(4, {"n_input": p.n_input, "n_classified": p.n_beats,
                     "n_dropped": p.n_dropped_boundary + p.n_dropped_edge, "examples": p.X[:8]})
    truth = match_truth(pred.r_peaks, win)
    called = [pred.classes[i] for i in pred.predicted]
    pairs = [(c, t) for c, t in zip(called, truth) if t is not None]
    display.update({"truth": truth, "agree": sum(c == t for c, t in pairs), "of": len(pairs)})
    yield _event(5, {"classes": pred.classes, "called": called, "scores": pred.scores,
                     "r_peaks": pred.r_peaks, "display_only": dict(display)})
    heart = heart_agent.analyze(heart_agent.BeatCalls.from_prediction(pred, win.fs, source))
    findings += heart
    yield _event(6, {"findings": heart})


def run_case(req: CaseRequest, deps: Deps) -> Iterator[StageEvent]:
    if req.record is None and req.stay is None:
        raise ValueError("choose an ECG record, an ICU stay, or both")
    findings: list = []
    display: dict = {}
    context: list = [OBSERVATION_HOURS]

    if req.record is None or deps.classifier is None:
        reason = ("no ECG chosen" if req.record is None
                  else deps.unavailable.get("ecg", "ECG model not loaded"))
        for stage in range(1, 7):
            yield _event(stage, {"skipped": reason})
    else:
        yield from _ecg_stages(req, deps, findings, display, context)

    if req.stay is None or deps.chart is None:
        reason = ("no ICU stay chosen" if req.stay is None
                  else deps.unavailable.get("chart", "MIMIC outputs not loaded"))
        yield _event(7, {"skipped": reason})
    else:
        chart = chart_agent.analyze(req.stay, deps.chart, deps.knowledge)
        context.append(float(req.stay))
        findings += chart
        yield _event(7, {"findings": chart})

    matches = rules.evaluate(deps.knowledge.rules, findings)
    yield _event(8, {"matches": matches})
    case = Case(banner=make_banner(req.record, req.stay, deps.source), record=req.record,
                stay=req.stay, findings=findings, rules=matches,
                context_numbers=tuple(context), display_only=display)
    yield _event(9, {"case": case, "report": deps.write_report(case)})


def default_deps(*, need_ecg: bool, need_chart: bool, source: str = "demo", mimic_dir=None,
                 mimic_raw_dir=None, mitdb_dir=None, bundle_path=None,
                 knowledge_path=None) -> Deps:
    """Load whatever is available; record why anything is not."""
    from mimic_iii import config as mimic_config

    deps = Deps(knowledge=load_knowledge(knowledge_path or DEFAULT_PATH),
                record_dir=mitdb_dir, source=source)
    if need_ecg:
        try:
            deps.classifier = ecg_inference.EcgClassifier.from_bundle(
                bundle_path or ecg_inference.DEFAULT_BUNDLE_PATH)
        except (FileNotFoundError, ValueError) as exc:
            deps.unavailable["ecg"] = f"ECG model not available: {exc}"
    if need_chart:
        try:
            deps.chart = load_chart_sources(mimic_dir or mimic_config.output_dir(source),
                                            mimic_raw_dir or mimic_config.source_dir(source),
                                            source)
        except (FileNotFoundError, ValueError, KeyError, MimicError) as exc:
            deps.unavailable["chart"] = (f"MIMIC outputs not available ({exc}); run "
                                         f"scripts/mimic_iii/build_features.py, make_splits.py, "
                                         f"train.py")
    return deps
