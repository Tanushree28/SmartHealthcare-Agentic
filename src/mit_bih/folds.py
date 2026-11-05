"""Record-grouped cross-validation folds over DS1.

``splits.ds1_ds2_split`` carves 4 validation records out of DS1. Measured, those
4 records (112, 205, 215, 223) hold **81 S beats, 26 F beats and zero Q beats**
-- too thin to tune per-class decision thresholds against, since one beat
flipping moves S recall by 1.2 points.

These folds put every DS1 record in the validation position exactly once, so all
944 DS1 S beats serve as validation data across a study rather than 81. Grouping
is by record: no patient appears on both sides of any fold.

DS2 never appears here. Selection happens inside DS1 only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, splits


@dataclass(frozen=True)
class FoldAssignment:
    """Which records are held out in each cross-validation fold."""

    folds: tuple[tuple[str, ...], ...]
    seed: int

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def validate(self) -> None:
        """Assert the guarantees this class exists to provide."""
        flat = [r for fold in self.folds for r in fold]

        seen: set[str] = set()
        repeated: set[str] = set()
        for record in flat:
            if record in seen:
                repeated.add(record)
            seen.add(record)
        if repeated:
            raise ValueError(
                f"records {sorted(repeated)} appear in two folds: a patient "
                f"would be both trained on and validated against"
            )

        leaked = sorted(seen & set(splits.DS2_RECORDS))
        if leaked:
            raise ValueError(
                f"DS2 records {leaked} appear in a fold. DS2 is the test set "
                f"and must never enter selection."
            )

        paced = sorted(seen & set(config.PACED_RECORDS))
        if paced:
            raise ValueError(
                f"paced records {paced} appear in a fold; AAMI excludes them"
            )

    def train_val(self, index: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Training and validation records for one fold."""
        if not 0 <= index < self.n_folds:
            raise IndexError(f"fold {index} out of range for {self.n_folds} folds")
        val = self.folds[index]
        train = tuple(
            sorted(r for i, fold in enumerate(self.folds) if i != index for r in fold)
        )
        return train, val

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_folds": self.n_folds,
            "group_unit": "record",
            "folds": [list(f) for f in self.folds],
            "note": (
                "Grouped by record: no patient appears on both sides of any "
                "fold. DS2 records are excluded by construction. Replaces the "
                "single 4-record validation split, which held only 81 S beats."
            ),
        }

    def write_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path


def make_folds(
    records=splits.DS1_RECORDS,
    n_folds: int = 5,
    seed: int = config.PHASE_1D_SEED,
) -> FoldAssignment:
    """Partition records into ``n_folds`` record-grouped folds, deterministically."""
    ordered = sorted({str(r) for r in records})

    if n_folds < 2:
        raise ValueError(f"need at least 2 folds, got {n_folds}")
    if n_folds > len(ordered):
        raise ValueError(
            f"more folds than records: {n_folds} folds for {len(ordered)} records"
        )

    # Sorting before shuffling makes the result depend only on the seed and the
    # set of records, never on the order they happened to be passed in.
    shuffled = [str(r) for r in np.random.default_rng(seed).permutation(ordered)]
    chunks = np.array_split(np.arange(len(shuffled)), n_folds)

    assignment = FoldAssignment(
        folds=tuple(tuple(sorted(shuffled[i] for i in chunk)) for chunk in chunks),
        seed=seed,
    )
    assignment.validate()
    return assignment
