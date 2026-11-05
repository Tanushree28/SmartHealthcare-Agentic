"""The one format every agent speaks.

A :class:`Finding` is one fact with its evidence. The heart agent, the chart
agent and the rules produce findings; the report, the chat and the
fact-checker read them. ``statement`` is always written by code from the
values -- never by the LLM -- so every number a reader sees traces back to a
model output or a chart value.

A :class:`Case` is everything known about one (possibly simulated) patient.
``display_only`` holds what a person may look at but no LLM may see: the
cardiologist's beat labels. :meth:`Case.facts_for_llm` is the only way facts
reach a prompt, and it never reads ``display_only`` or ``evidence``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PREFIX = {"heart": "H", "chart": "C"}


@dataclass(frozen=True)
class Finding:
    id: str
    agent: str  # "heart" | "chart" | "rule" | "glossary"
    kind: str
    key: str
    statement: str
    value: float | int | str | None = None
    unit: str = ""
    evidence: dict = field(default_factory=dict)
    #: Vocabulary keys this finding can back (see verifier.VOCAB), e.g. "beat:V".
    terms: tuple[str, ...] = ()
    caveat: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        out["terms"] = list(self.terms)
        return out


class FindingList:
    """Collects one agent's findings and numbers them H1, H2, ... or C1, C2, ..."""

    def __init__(self, agent: str) -> None:
        if agent not in PREFIX:
            raise ValueError(f"no ID prefix for agent {agent!r}")
        self.agent = agent
        self.items: list[Finding] = []

    def add(self, kind: str, key: str, statement: str, value=None, unit: str = "",
            evidence: dict | None = None, terms=(), caveat: str = "") -> Finding:
        finding = Finding(id=f"{PREFIX[self.agent]}{len(self.items) + 1}", agent=self.agent,
                          kind=kind, key=key, statement=statement, value=value, unit=unit,
                          evidence=dict(evidence or {}), terms=tuple(terms), caveat=caveat)
        self.items.append(finding)
        return finding


#: Given to the LLM with a simulated case's facts (report and chat).
SIMULATED_NOTE = ("The ECG facts (H...) and the chart facts (C...) come from two different "
                  "people. Never describe them as one patient: say 'the ECG' and 'the chart'.")


def make_banner(record: str | None, stay: int | None, source: str = "demo") -> str:
    data = "MIMIC-III Demo" if source == "demo" else "MIMIC-III"
    if record is not None and stay is not None:
        return (f"SIMULATED CASE: ECG from MIT-BIH record {record}; chart from {data} ICU stay "
                f"{stay}. These are different people. Research demo, not for patient care.")
    if record is not None:
        return f"ECG from MIT-BIH record {record}. Research demo, not for patient care."
    if stay is not None:
        return f"Chart from {data} ICU stay {stay}. Research demo, not for patient care."
    raise ValueError("a case needs an ECG record, an ICU stay, or both")


@dataclass
class Case:
    banner: str
    record: str | None = None
    stay: int | None = None
    findings: list[Finding] = field(default_factory=list)
    rules: list[Finding] = field(default_factory=list)
    #: Numbers a text may use besides the findings: record, stay, window, 24 h.
    context_numbers: tuple[float, ...] = ()
    #: For people only -- never given to any LLM.
    display_only: dict = field(default_factory=dict)

    @property
    def simulated(self) -> bool:
        """An MIT-BIH ECG with a MIMIC-III chart: always two different people."""
        return self.record is not None and self.stay is not None

    def all_findings(self) -> list[Finding]:
        return [*self.findings, *self.rules]

    def by_id(self) -> dict[str, Finding]:
        return {f.id: f for f in self.all_findings()}

    def facts_for_llm(self) -> list[dict]:
        """What a prompt may contain: ID, statement, value, unit, caveat. Nothing else."""
        return [{"id": f.id, "statement": f.statement, "value": f.value, "unit": f.unit,
                 **({"caveat": f.caveat} if f.caveat else {})} for f in self.all_findings()]


@dataclass(frozen=True)
class ReportResult:
    kind: str  # "llm" | "template"
    text: str
    model: str | None = None
    seconds: float = 0.0
    attempts: int = 0
    problems: tuple = ()  # verifier.Problem items from the last rejected draft
    note: str = ""
    draft: dict | None = None
