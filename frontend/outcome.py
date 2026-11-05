"""The real in-hospital outcome of an ICU stay -- DISPLAY ONLY.

Shown on the page behind a "Reveal real outcome" button so a person can
compare the Demo risk score with what happened. It lives here, outside
src/agentic, on purpose: no agent, rule, report or chat imports this module,
and its answer is never stored in a case (a test checks both).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NOTE = "Display only: the agents, the report and the chat never see this."


class OutcomeLookup:
    def __init__(self, out_dir) -> None:
        self.path = Path(out_dir).expanduser() / "features.csv.gz"
        self._labels: dict[int, int] | None = None

    def _load(self) -> dict[int, int]:
        if self._labels is None:
            df = pd.read_csv(self.path, usecols=["icustay_id", "label"])
            self._labels = {int(s): int(v) for s, v in zip(df["icustay_id"], df["label"])}
        return self._labels

    def get(self, stay: int) -> dict:
        labels = self._load()
        if int(stay) not in labels:
            raise KeyError(f"ICU stay {stay} is not in the cohort")
        died = labels[int(stay)] == 1
        return {"stay": int(stay),
                "outcome": "died in hospital" if died else "survived to hospital discharge",
                "note": NOTE}
