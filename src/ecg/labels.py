"""MIT-BIH annotation symbol to AAMI class mapping (PDF 2.6).

Three outcomes are possible for any annotation symbol, and they are kept
distinct on purpose:

``beat``
    A real heartbeat, mapped to one of the five AAMI classes.
``excluded``
    A non-beat annotation -- rhythm change, signal-quality marker, waveform
    landmark. These are **never** folded into class Q. Mislabelling a rhythm
    marker as an unclassifiable beat injects phantom beats into the dataset,
    and no downstream metric would reveal it.
``unknown``
    A symbol absent from both tables. Raises in strict mode, counted otherwise.
    A silent mapping gap must never quietly shrink the dataset.

Symbol descriptions were cross-checked against ``wfdb.io.annotation.ann_label_table``
rather than recalled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import config
from .errors import UnknownSymbolError

Outcome = Literal["beat", "excluded", "unknown"]

# ---------------------------------------------------------------------------
# Beat symbols -> AAMI class (ANSI/AAMI EC57)
# ---------------------------------------------------------------------------

#: The canonical 15 MIT-BIH beat types and their AAMI class.
BEAT_SYMBOL_TO_AAMI: dict[str, str] = {
    # N -- normal and bundle-branch-block beats, plus escape beats of
    # supraventricular origin.
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat
    # S -- supraventricular ectopic beats.
    "A": "S",  # Atrial premature contraction
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Premature or ectopic supraventricular beat
    # V -- ventricular ectopic beats.
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat
    # F -- fusion of ventricular and normal.
    "F": "F",
    # Q -- paced and unclassifiable.
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
}

#: WFDB beat codes that exist in the annotation standard but do **not** occur in
#: MIT-BIH Arrhythmia Database v1.0.0. Mapped anyway so a record from another
#: PhysioNet database cannot fall through into ``unknown`` unnoticed.
EXTENDED_BEAT_SYMBOL_TO_AAMI: dict[str, str] = {
    "B": "N",  # Left or right bundle branch block (unspecified)
    "n": "S",  # Supraventricular escape beat
    "r": "V",  # R-on-T premature ventricular contraction
    "?": "Q",  # Learning -- beat not classified during the learning period
}

# ---------------------------------------------------------------------------
# Non-beat symbols -- excluded, never classified
# ---------------------------------------------------------------------------

#: Annotations that mark something other than a heartbeat.
#:
#: ``!`` (ventricular flutter wave) is here deliberately. It marks a wave inside
#: a flutter episode rather than a discrete classifiable beat, and the canonical
#: AAMI class table does not list it. Counting these as class V would inflate V
#: with a morphology that is not a ventricular ectopic beat. They are counted as
#: excluded and visible in the manifest, not silently dropped.
NON_BEAT_SYMBOLS: dict[str, str] = {
    "+": "Rhythm change",
    "~": "Signal quality change",
    "|": "Isolated QRS-like artifact",
    "s": "ST change",
    "T": "T-wave change",
    "*": "Systole",
    "D": "Diastole",
    '"': "Comment annotation",
    "=": "Measurement annotation",
    "p": "P-wave peak",
    "t": "T-wave peak",
    "u": "U-wave peak",
    "^": "Non-conducted pacer spike",
    "!": "Ventricular flutter wave",
    "[": "Start of ventricular flutter/fibrillation",
    "]": "End of ventricular flutter/fibrillation",
    "@": "Link to external data",
    "x": "Non-conducted P-wave (blocked APB)",
    "(": "Waveform onset",
    ")": "Waveform end",
    "`": "PQ junction",
    "'": "J-point",
    "": "Not an actual annotation",
}

#: Full beat table: canonical MIT-BIH types plus the extended WFDB codes.
ALL_BEAT_SYMBOLS: dict[str, str] = {
    **BEAT_SYMBOL_TO_AAMI,
    **EXTENDED_BEAT_SYMBOL_TO_AAMI,
}


@dataclass(frozen=True)
class SymbolMapping:
    """The full result of mapping one annotation symbol."""

    symbol: str
    outcome: Outcome
    aami_class: str | None = None
    class_index: int | None = None
    reason: str | None = None

    @property
    def is_beat(self) -> bool:
        return self.outcome == "beat"


def class_index(aami_class: str) -> int:
    """Index of an AAMI class in the stable, documented order.

    The order in :data:`ecg.config.AAMI_CLASSES` is a contract. Reordering it
    would silently invalidate every saved label array and any trained model.
    """
    try:
        return config.AAMI_CLASSES.index(aami_class)
    except ValueError as exc:
        raise ValueError(
            f"{aami_class!r} is not an AAMI class; expected one of "
            f"{list(config.AAMI_CLASSES)}"
        ) from exc


def map_symbol(symbol: str, strict: bool = False) -> SymbolMapping:
    """Map one annotation symbol to a beat class, an exclusion, or unknown."""
    sym = str(symbol)

    aami = ALL_BEAT_SYMBOLS.get(sym)
    if aami is not None:
        return SymbolMapping(
            symbol=sym,
            outcome="beat",
            aami_class=aami,
            class_index=class_index(aami),
        )

    reason = NON_BEAT_SYMBOLS.get(sym)
    if reason is not None:
        return SymbolMapping(symbol=sym, outcome="excluded", reason=reason)

    if strict:
        raise UnknownSymbolError(
            f"annotation symbol {sym!r} is in neither the AAMI beat table nor "
            f"the non-beat table; refusing to guess. Add it to ecg.labels or "
            f"run with strict=False to count it as unknown."
        )
    return SymbolMapping(symbol=sym, outcome="unknown", reason="unmapped symbol")


def map_symbols(symbols, strict: bool = False) -> list[SymbolMapping]:
    """Map a sequence of symbols, preserving order."""
    return [map_symbol(s, strict=strict) for s in symbols]
