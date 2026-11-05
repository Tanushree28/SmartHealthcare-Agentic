"""What the page can ask for, without any HTTP: options, a run, chat, outcome.

``App.run`` wraps ``agentic.orchestrator.run_case``: it yields ``(kind, data)``
pairs -- ``("stage", ...)`` nine times, then ``("done", {"case_id"})`` -- or an
``("error", {"message"})`` the page can show. Finished cases are kept in memory
(the last 20) so the chat can answer about them; nothing is written to disk.

One lock keeps ECG inference and LLM calls one at a time: Ollama serves one
request at a time anyway, and the torch models are not shared across threads.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import wfdb

from agentic import chat_agent, ecg_window, orchestrator, report_agent
from agentic import glossary as glossary_mod
from agentic.llm import LlmStatus
from mit_bih import config as ecg_config
from mit_bih.errors import EcgError

from . import serialize

MIN_SECONDS = float(ecg_window.MIN_SECONDS)
#: Longest window the page asks for: keeps one stream under about 1 MB.
MAX_SECONDS = 120.0
LLM_OFF = LlmStatus(False, "", "LLM off: template reports only, chat off")


def _one(query: dict, name: str) -> str:
    values = query.get(name) or [""]
    return str(values[0]).strip()


def parse_run(query: dict) -> orchestrator.CaseRequest:
    """A /api/run query string (as parsed by urllib.parse.parse_qs) -> a request."""
    record = _one(query, "record") or None
    stay_text = _one(query, "stay")
    try:
        stay = int(stay_text) if stay_text else None
    except ValueError:
        raise ValueError(f"stay must be a number, got {stay_text!r}") from None
    try:
        start = float(_one(query, "start") or 0)
        seconds = float(_one(query, "seconds") or 20)
    except ValueError:
        raise ValueError("start and seconds must be numbers") from None
    if not (math.isfinite(start) and math.isfinite(seconds)):
        raise ValueError("start and seconds must be numbers")
    if seconds > MAX_SECONDS:
        raise ValueError(f"seconds must be at most {MAX_SECONDS:g} on the page")
    peaks = _one(query, "peaks") or "annotation"
    if peaks not in ("annotation", "detected"):
        raise ValueError("peaks must be 'annotation' or 'detected'")
    if record is None and stay is None:
        raise ValueError("choose an ECG record, an ICU stay, or both")
    return orchestrator.CaseRequest(record=record, start_s=start, seconds=seconds, stay=stay,
                                    detect_peaks=peaks == "detected")


@dataclass
class _Stored:
    case: object
    history: list = field(default_factory=list)


class App:
    def __init__(self, deps: orchestrator.Deps, llm=None, glossary=None, outcome=None,
                 max_cases: int = 20, llm_status: LlmStatus | None = None) -> None:
        self.deps = deps
        self.llm = llm
        self.deps.write_report = report_agent.make_writer(llm)
        self.glossary = glossary if glossary is not None else glossary_mod.load_glossary()
        self.outcome_lookup = outcome
        self.max_cases = max_cases
        self.cases: OrderedDict[str, _Stored] = OrderedDict()
        self.llm_status = llm_status or (llm.status() if llm is not None else LLM_OFF)
        self._work_lock = threading.Lock()
        self._store_lock = threading.Lock()
        self._records: list[dict] | None = None

    # -- options -------------------------------------------------------------

    def _record_list(self) -> list[dict]:
        if self._records is None:
            directory = Path(self.deps.record_dir or ecg_config.mitdb_dir()).expanduser()
            out = []
            for r in ecg_window.offered_records():
                try:
                    header = wfdb.rdheader(str(directory / r.name))
                    seconds = round(header.sig_len / header.fs, 1)
                except (OSError, ValueError):
                    seconds = None  # not on disk: the page shows it disabled
                out.append({"name": r.name, "tag": r.tag, "note": r.note, "seconds": seconds})
            self._records = out
        return self._records

    def _stay_list(self) -> list[dict]:
        chart = self.deps.chart
        if chart is None:
            return []
        stays = []
        for row in chart.cohort.sort_values("icustay_id").itertuples(index=False):
            stays.append({
                "id": int(row.icustay_id),
                "age": ">89" if int(row.age_over_89) == 1 else f"{float(row.age):.0f}",
                "sex": str(row.gender),
                "reason": chart.admission_reason.get(int(row.hadm_id), ""),
                "split": chart.split_of.get(int(row.subject_id), "unknown")})
        return stays

    def options(self) -> dict:
        s = self.llm_status
        return {"records": self._record_list(), "stays": self._stay_list(),
                "llm": {"ok": s.ok, "model": s.model, "message": s.message},
                "limits": {"min_seconds": MIN_SECONDS, "max_seconds": MAX_SECONDS},
                "unavailable": dict(self.deps.unavailable),
                "outcome_available": self.outcome_lookup is not None}

    # -- a run ---------------------------------------------------------------

    def _store(self, case) -> str:
        case_id = uuid.uuid4().hex
        with self._store_lock:
            self.cases[case_id] = _Stored(case)
            while len(self.cases) > self.max_cases:
                self.cases.popitem(last=False)
        return case_id

    def run(self, query: dict):
        try:
            req = parse_run(query)
        except ValueError as exc:
            yield "error", {"message": str(exc)}
            return
        case_id = None
        with self._work_lock:
            try:
                for ev in orchestrator.run_case(req, self.deps):
                    data = serialize.event(ev)
                    if ev.stage == 9:
                        case_id = self._store(ev.payload["case"])
                        data["case_id"] = case_id
                    yield "stage", data
            except (ecg_window.WindowError, EcgError, KeyError, ValueError) as exc:
                yield "error", {"message": str(exc.args[0]) if exc.args else str(exc)}
                return
        yield "done", {"case_id": case_id}

    # -- chat and outcome ----------------------------------------------------

    def chat(self, case_id: str, question: str) -> dict:
        with self._store_lock:
            stored = self.cases.get(case_id)
        if stored is None:
            raise KeyError("this case is no longer in memory; run it again")
        agent = chat_agent.ChatAgent(stored.case, self.glossary, self.llm)
        with self._work_lock:
            answer = agent.ask(question, stored.history)
        stored.history.append((question, answer.text))
        return {"answer": answer.text, "cites": list(answer.cites), "verified": answer.verified,
                "refused": answer.refused, "reason": answer.reason}

    def outcome(self, stay: int) -> dict:
        if self.outcome_lookup is None:
            raise KeyError("the outcome is not available")
        return self.outcome_lookup.get(int(stay))
