"""Beat neighbourhoods, so a model can see the rhythm it is judging against.

The Phase 1d model classified one beat at a time. But "premature" is a statement
about a beat *relative to the rhythm around it*, and a single beat carries only
its own RR numbers -- the model had to infer the reference rhythm from them. It
did not do that well: 3,015 normal beats in DS2 were called S, which is 98% of
that arm's false positives and the reason S precision sat at 0.29.

This builds a window of consecutive beats around each beat. Two rules matter:

* **Windows never cross a record boundary.** Two patients' beats inside one
  window would be a leakage bug that presents as a modelling result.
* **Windows are indices, not copies.** A 5-wide window over the 50,955 DS1 beats
  would be 306 MB if materialised; as indices it is 2 MB, and batches are
  gathered on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import BeatData

#: Beats either side of the centre beat. Five beats of context is enough to
#: establish a local rhythm without spanning a change of rhythm.
DEFAULT_CONTEXT: int = 2


@dataclass(frozen=True)
class SequenceData:
    """Beat windows expressed as indices into an underlying ``BeatData``."""

    beats: BeatData
    neighbours: np.ndarray  # (n, width) int64 -- indices into beats.X
    context: int

    @property
    def width(self) -> int:
        return int(self.neighbours.shape[1])

    @property
    def n(self) -> int:
        return int(self.neighbours.shape[0])

    @property
    def y(self) -> np.ndarray:
        """Label of the centre beat -- the beat being classified."""
        return self.beats.y

    @property
    def record(self) -> np.ndarray:
        return self.beats.record

    def gather(self, rows) -> tuple[np.ndarray, np.ndarray]:
        """Materialise the windows for the given rows.

        Returns ``(beats, rr)`` shaped ``(len(rows), width, BEAT_LENGTH)`` and
        ``(len(rows), width, 6)``.
        """
        index = self.neighbours[np.asarray(rows, dtype=np.int64)]
        return self.beats.X[index], self.beats.rr[index]


def make_sequences(
    data: BeatData,
    context: int = DEFAULT_CONTEXT,
) -> SequenceData:
    """Build a ``context``-wide neighbourhood around every beat.

    At a record's edges the window clamps -- the first beat repeats itself
    rather than borrowing the previous patient's last beat. Clamping keeps every
    beat classifiable, which matters because dropping edge beats would change
    the denominator and break comparability with Phase 1b.
    """
    if context < 0:
        raise ValueError(f"context must be >= 0, got {context}")

    n = data.n_kept
    width = 2 * context + 1
    neighbours = np.empty((n, width), dtype=np.int64)

    offsets = np.arange(-context, context + 1, dtype=np.int64)

    # Record boundaries: beats arrive grouped by record, so a change in the
    # record column marks a new patient.
    records = np.asarray(data.record, dtype=object)
    starts: list[int] = []
    stops: list[int] = []
    if n:
        boundaries = np.flatnonzero(records[1:] != records[:-1]) + 1
        edges = [0, *boundaries.tolist(), n]
        starts = edges[:-1]
        stops = edges[1:]

    for start, stop in zip(starts, stops):
        rows = np.arange(start, stop, dtype=np.int64)
        candidate = rows[:, None] + offsets[None, :]
        # Clamp inside this record only.
        neighbours[start:stop] = np.clip(candidate, start, stop - 1)

    return SequenceData(beats=data, neighbours=neighbours, context=context)
