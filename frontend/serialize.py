"""Stage events -> JSON-safe dicts for the page.

numpy arrays become lists of plain floats and ints, rounded (signals to 4
decimals, scores to 3) so a 20 s window is about 100 KB. The case itself stays
on the server; stage 9 sends only the banner and the report.
"""

from __future__ import annotations

import numpy as np

from agentic.orchestrator import StageEvent


def _floats(values, digits: int) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(values, dtype=float).ravel()]


def _ints(values) -> list[int]:
    return [int(x) for x in np.asarray(values).ravel()]


def event(ev: StageEvent) -> dict:
    out = {"stage": ev.stage, "name": ev.name}
    p = ev.payload
    if ev.skipped:
        out["skipped"] = p["skipped"]
        return out
    if ev.stage == 1:
        r = p["record"]
        out.update(record={"name": r.name, "tag": r.tag, "note": r.note}, lead=p["lead"],
                   fs=int(p["fs"]), start_s=float(p["start_s"]), signal=_floats(p["signal"], 4))
    elif ev.stage == 2:
        out["signal"] = _floats(p["signal"], 4)
    elif ev.stage == 3:
        out.update(r_peaks=_ints(p["r_peaks"]), source=p["source"])
    elif ev.stage == 4:
        out.update(n_input=int(p["n_input"]), n_classified=int(p["n_classified"]),
                   n_dropped=int(p["n_dropped"]),
                   examples=[_floats(row, 3) for row in p["examples"]])
    elif ev.stage == 5:
        shown = p["display_only"]
        out.update(classes=list(p["classes"]), called=list(p["called"]),
                   scores=[_floats(row, 3) for row in p["scores"]], r_peaks=_ints(p["r_peaks"]),
                   display_only={"truth": list(shown["truth"]), "agree": int(shown["agree"]),
                                 "of": int(shown["of"])})
    elif ev.stage in (6, 7):
        out["findings"] = [f.to_dict() for f in p["findings"]]
    elif ev.stage == 8:
        out["matches"] = [f.to_dict() for f in p["matches"]]
    elif ev.stage == 9:
        case, report = p["case"], p["report"]
        out.update(banner=case.banner, simulated=case.simulated, report={
            "kind": report.kind, "text": report.text, "model": report.model,
            "note": report.note, "attempts": report.attempts,
            "seconds": round(float(report.seconds), 1),
            "problems": [pr.detail for pr in report.problems]})
    return out
