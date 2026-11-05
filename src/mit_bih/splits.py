"""Train/validation/test assignment (PDF 2.8 -- "This is very important").

**Splitting is record-wise, always.** Beat-level splitting puts beats from the
same patient on both sides of the split; a model then recognises the patient
rather than the arrhythmia, and reported accuracy becomes meaningless. The
safest way to prevent that is to make it unexpressible: no function here accepts
a split-unit argument, so there is no beat-level mode to reach for.

Two strategies:

``record_wise_split``
    A seeded, reproducible split over whatever records are present. Use this for
    development and for any record set that is not the standard 48.
``ds1_ds2_split``
    The inter-patient DS1/DS2 assignment from de Chazal et al. (2004), used
    throughout the arrhythmia literature so results are comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

# ---------------------------------------------------------------------------
# DS1 / DS2 -- inter-patient split
# ---------------------------------------------------------------------------
#
# Source: P. de Chazal, M. O'Dwyer, R. B. Reilly, "Automatic Classification of
# Heartbeats Using ECG Morphology and Heartbeat Interval Features",
# IEEE Trans. Biomed. Eng. 51(7):1196-1206, 2004, Section II-B.
#
# 44 records = the 48 MIT-BIH records minus the four paced records
# (102, 104, 107, 217), which the AAMI recommended practice excludes.
# 22 records per set, no patient in both.

DS1_RECORDS: tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
)

DS2_RECORDS: tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
)


@dataclass(frozen=True)
class SplitAssignment:
    """Which records belong to which split, plus what was excluded and why."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    excluded: tuple[str, ...]
    strategy: str
    seed: int | None = None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "split_unit": "record",
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "excluded": list(self.excluded),
            "note": (
                "Splitting is record-wise. No beat from a given record appears "
                "in more than one split (PDF 2.8)."
            ),
        }

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def split_of(self, record: str) -> str | None:
        for name in ("train", "val", "test"):
            if record in getattr(self, name):
                return name
        return None

    def validate(self) -> None:
        """Assert the guarantees this class exists to provide."""
        sets = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        for a in sets:
            for b in sets:
                if a < b and sets[a] & sets[b]:
                    raise ValueError(
                        f"record leakage: {sorted(sets[a] & sets[b])} appear in "
                        f"both {a} and {b}"
                    )
        overlap = set(self.excluded) & set().union(*sets.values())
        if overlap:
            raise ValueError(f"excluded records also assigned: {sorted(overlap)}")


def _eligible(records, exclude_paced: bool) -> tuple[list[str], list[str]]:
    # Deduplicate: a record id repeated in the input would otherwise be
    # assigned to two splits, which validate() reports as leakage.
    ordered = sorted({str(r) for r in records})
    if not exclude_paced:
        return ordered, []
    paced = set(config.PACED_RECORDS)
    kept = [r for r in ordered if r not in paced]
    dropped = [r for r in ordered if r in paced]
    return kept, dropped


def record_wise_split(
    records,
    fractions: tuple[float, float, float] = config.DEFAULT_SPLIT_FRACTIONS,
    seed: int = config.DEFAULT_SPLIT_SEED,
    exclude_paced: bool = True,
) -> SplitAssignment:
    """Seeded record-wise split. Reproducible for a given ``(records, seed)``."""
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, val, test)")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions)}")
    if any(f < 0 for f in fractions):
        raise ValueError("fractions must be non-negative")

    eligible, excluded = _eligible(records, exclude_paced)
    n = len(eligible)
    if n == 0:
        raise ValueError("no eligible records to split")

    order = np.random.default_rng(seed).permutation(n)
    shuffled = [eligible[i] for i in order]

    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))

    # Guarantee every requested split gets at least one record when there are
    # enough to go round. Rounding alone starves val on small record sets, and
    # an empty validation split fails silently at training time rather than here.
    wants_val = fractions[1] > 0
    wants_test = fractions[2] > 0
    min_needed = 1 + int(wants_val) + int(wants_test)

    if n >= min_needed:
        n_train = max(n_train, 1)
        n_val = max(n_val, 1) if wants_val else 0
        # Shrink train -- the largest split -- rather than starve val or test.
        overflow = n_train + n_val + int(wants_test) - n
        if overflow > 0:
            n_train = max(n_train - overflow, 1)
            overflow = n_train + n_val + int(wants_test) - n
            if overflow > 0 and wants_val:
                n_val = max(n_val - overflow, 1)

    assignment = SplitAssignment(
        train=tuple(sorted(shuffled[:n_train])),
        val=tuple(sorted(shuffled[n_train : n_train + n_val])),
        test=tuple(sorted(shuffled[n_train + n_val :])),
        excluded=tuple(excluded),
        strategy="record_wise",
        seed=seed,
    )
    assignment.validate()
    return assignment


def ds1_ds2_split(
    records=None,
    val_fraction: float = 0.2,
    seed: int = config.DEFAULT_SPLIT_SEED,
) -> SplitAssignment:
    """Inter-patient DS1/DS2 assignment (de Chazal et al., 2004).

    DS2 is the test set, untouched, so results stay comparable to published
    work. The validation set is carved deterministically out of DS1 -- never out
    of DS2, which would contaminate the comparison.
    """
    available = None if records is None else {str(r) for r in records}

    ds1 = [r for r in DS1_RECORDS if available is None or r in available]
    ds2 = [r for r in DS2_RECORDS if available is None or r in available]

    if not ds1 or not ds2:
        raise ValueError(
            "the DS1/DS2 split needs records from both sets; present records "
            f"cover {len(ds1)} of DS1 and {len(ds2)} of DS2"
        )

    n_val = int(round(val_fraction * len(ds1)))
    n_val = min(max(n_val, 1 if val_fraction > 0 else 0), len(ds1) - 1)

    order = np.random.default_rng(seed).permutation(len(ds1))
    shuffled = [ds1[i] for i in order]

    excluded = sorted(config.PACED_RECORDS)
    if available is not None:
        excluded = sorted(r for r in config.PACED_RECORDS if r in available)

    assignment = SplitAssignment(
        train=tuple(sorted(shuffled[n_val:])),
        val=tuple(sorted(shuffled[:n_val])),
        test=tuple(sorted(ds2)),
        excluded=tuple(excluded),
        strategy="ds1_ds2",
        seed=seed,
    )
    assignment.validate()
    return assignment
