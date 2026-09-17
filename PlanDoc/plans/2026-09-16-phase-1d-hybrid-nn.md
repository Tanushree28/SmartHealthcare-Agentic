# Phase 1d — Hybrid Neural Classifier and Imbalance Study: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-branch neural network (1D CNN over the beat waveform + a dedicated branch for RR timing features) and run a controlled six-arm study that separates the effect of imbalance handling from the effect of architecture, measured against the Phase 1b Random Forest floor.

**Architecture:** The Phase 1 pipeline already writes normalized 300-sample beats per record, so no preprocessing changes. A new `dataset.py` pairs each beat with its 6 RR features. A new `folds.py` partitions the 22 DS1 records into 5 record-grouped cross-validation folds. `src/ecg/nn/` holds the model, the loss functions, and the training loop. Every selection decision is made by cross-validation on DS1; DS2 is scored exactly once at the end.

**Tech Stack:** Python 3.13 (arm64), PyTorch 2.14.0 (CPU), NumPy 2.3.3, scikit-learn 1.7.2 (already present, used only for `confusion_matrix`), pytest 8.4.2.

**Spec:** `PlanDoc/specs/2026-09-16-phase-1d-hybrid-nn-design.md` — read it before starting. This plan argues from that spec; where they disagree, the spec wins and the plan is wrong.

## Global Constraints

- **Do not commit yet.** The user asked on 2026-09-16 that nothing be committed. Each task below ends with a commit step for when that hold is lifted — **stage the work and stop; do not run `git commit` until the user says so.**
- **Python interpreter is `./venv/bin/python`.** Not `python3` (that is 3.14.2 and lacks the dependencies).
- **Torch is pinned to `torch==2.14.0`.** Verified 2026-09-16 to install from a cp313 arm64 wheel with no source build. It pulls in `filelock`, `jinja2`, `MarkupSafe`, `networkx`, `sympy`, `mpmath`, `setuptools` transitively — record this in `requirements.txt` the way Deviation 5 recorded matplotlib.
- **The AAMI class order `("N", "S", "V", "F", "Q")` is a contract.** `config.AAMI_CLASSES`. Never reorder; saved label arrays depend on it.
- **The test suite runs offline.** No test may read the real dataset unless marked `@pytest.mark.requires_data`. `pytest.ini` already deselects that marker by default.
- **`filterwarnings = error::RuntimeWarning` is active.** A divide-by-zero in a metric raises rather than returning NaN. Guard divisions explicitly.
- **DS2 is never used for selection.** No hyperparameter, no threshold, no arm choice may reference a DS2 number. DS2 is scored once, in Task 10.
- **Processed data lives at `~/physionet-data/processed/mitdb-ds12`** (the DS1/DS2 output). Not `processed/mitdb`, which is the record-wise output and has different counts.
- **Fixed seed for this phase: `20260916`.**
- **TDD.** Write the failing test, watch it fail, implement, watch it pass. A test that has not been observed failing has not been shown to test anything (Problem #10).

---

### Task 1: Add `macro_f1` to the evaluator

Ranking seven rows on `macro_recall` alone would repeat a documented mistake: §3 of `results/2026-08-27-baseline-ds1_ds2.md` showed it ranked the worst model of six in first place. This task is first because every later task's tests read the metric.

**Files:**
- Modify: `src/ecg/evaluate.py:34-51` (the `Evaluation` dataclass), `:55-80` (`to_dict`), `:116-128` (the `evaluate` return), `:130-150` (`format_report`)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `Evaluation.macro_f1: float` — the mean F1 over classes with non-zero support, matching how `macro_recall` already averages
  - `Evaluation.macro_f1_of(names) -> float` — the mean F1 over the named classes only
  - `config.HEADLINE_CLASSES: tuple[str, ...] = ("N", "S", "V", "F")` — Q excluded
  - `Evaluation.to_dict()` gains `"macro_f1"` and `"macro_f1_headline"` keys

**Why two numbers.** Spec §7 requires Q out of the headline ranking: it has 8 training beats and 7 test beats, so its F1 is noise, yet averaged over five classes it moves the number that ranks every model. `macro_f1` stays defined over all present classes for continuity with `macro_recall`; `macro_f1_headline` over N/S/V/F is what Task 10's table ranks on.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluate.py`:

```python
def test_macro_f1_averages_over_present_classes_only():
    # 4 classes present, Q absent. Perfect prediction -> every present F1 is 1.0.
    y_true = np.array([0, 0, 1, 2, 3])
    y_pred = np.array([0, 0, 1, 2, 3])
    result = evaluate.evaluate(y_true, y_pred)
    assert result.macro_f1 == pytest.approx(1.0)
    # Q has zero support and must not drag the average to 0.8.
    assert result.recall_of("Q") == 0.0


def test_macro_f1_is_hand_computable():
    # N: 2 true, both found -> recall 1.0. Predicted-as-N is 3 -> precision 2/3.
    #    F1 = 2 * (2/3) * 1 / (2/3 + 1) = 0.8
    # S: 1 true, missed -> recall 0.0, F1 0.0
    # V: 1 true, found -> recall 1.0, predicted-as-V is 1 -> precision 1.0, F1 1.0
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 2])
    result = evaluate.evaluate(y_true, y_pred)
    assert result.macro_f1 == pytest.approx((0.8 + 0.0 + 1.0) / 3)


def test_macro_f1_appears_in_json_payload():
    result = evaluate.evaluate(np.array([0, 1]), np.array([0, 1]))
    assert "macro_f1" in result.to_dict()


def test_headline_macro_f1_excludes_q():
    """Q has 7 beats in DS2. It must not move the ranking metric."""
    # N, S, V, F all perfect; Q present but always missed.
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 0])
    result = evaluate.evaluate(y_true, y_pred)
    assert result.macro_f1_of(config.HEADLINE_CLASSES) == pytest.approx(1.0)
    assert result.macro_f1 < 1.0          # the all-classes average is dragged down


def test_macro_f1_of_rejects_an_unknown_class():
    result = evaluate.evaluate(np.array([0, 1]), np.array([0, 1]))
    with pytest.raises(KeyError):
        result.macro_f1_of(("N", "Z"))


def test_headline_macro_f1_appears_in_json_payload():
    result = evaluate.evaluate(np.array([0, 1]), np.array([0, 1]))
    assert "macro_f1_headline" in result.to_dict()


def test_macro_f1_differs_from_macro_recall_on_over_prediction():
    # Reproduces the lda/rr_only trap: a rare class caught by spraying
    # predictions. Recall is flattered; F1 is not.
    y_true = np.array([0] * 20 + [1])
    y_pred = np.array([1] * 21)
    result = evaluate.evaluate(y_true, y_pred)
    assert result.recall_of("S") == pytest.approx(1.0)
    assert result.macro_f1 < result.macro_recall
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_evaluate.py -k macro_f1 -v`
Expected: FAIL — `AttributeError: 'Evaluation' object has no attribute 'macro_f1'`

- [ ] **Step 3: Implement**

In `src/ecg/evaluate.py`, add the field to `Evaluation` immediately after `macro_recall` (it must stay ahead of `label`, which has a default):

```python
    macro_recall: float
    macro_f1: float
    n_test: int
    label: str = ""
```

Add to `src/ecg/config.py`, directly under `AAMI_CLASSES`:

```python
#: Classes the headline metric averages over. Q is excluded: it has 8 training
#: beats and 7 in DS2, so its F1 is noise, and averaged over five classes that
#: noise moves the number that ranks every model.
HEADLINE_CLASSES: tuple[str, ...] = ("N", "S", "V", "F")
```

Add this method to `Evaluation`, next to `recall_of`:

```python
    def macro_f1_of(self, names) -> float:
        """Mean F1 over the named classes only.

        Used for the headline ranking, which excludes Q -- see
        config.HEADLINE_CLASSES.
        """
        by_name = {m.name: m for m in self.per_class}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(f"unknown classes {missing}")
        chosen = [by_name[n].f1 for n in names]
        return float(np.mean(chosen)) if chosen else 0.0
```

In `to_dict`, after the `"macro_recall"` line:

```python
            "macro_f1": round(self.macro_f1, 4),
            "macro_f1_headline": round(
                self.macro_f1_of(config.HEADLINE_CLASSES), 4
            ),
            "macro_f1_headline_note": (
                "Averaged over N/S/V/F only. Q is excluded: 8 training beats "
                "and 7 in DS2 make its F1 noise that would move the ranking."
            ),
            "macro_f1_note": (
                "Headline ranking metric. Macro recall alone rewards "
                "over-predicting a rare class with no offsetting penalty -- it "
                "ranked the worst of six Phase 1b models first. See "
                "results/2026-08-27-baseline-ds1_ds2.md section 3."
            ),
```

In the `evaluate()` return, after the `macro_recall=` line:

```python
        macro_f1=float(np.mean([m.f1 for m in present])) if present else 0.0,
```

In `format_report`, after the macro-recall line:

```python
    headline = result.macro_f1_of(config.HEADLINE_CLASSES)
    lines.append(f"macro-average F1     : {result.macro_f1:.3f}   <- all present classes")
    lines.append(f"macro F1 (N/S/V/F)   : {headline:.3f}   <- the ranking metric, Q excluded")
```

- [ ] **Step 4: Run the full suite**

Run: `./venv/bin/python -m pytest -q`
Expected: PASS. Watch specifically for any existing test that builds `Evaluation(...)` positionally — inserting a field breaks positional construction, and that is the likeliest failure here. If one breaks, convert it to keyword arguments rather than moving the new field.

- [ ] **Step 5: Stage (do not commit — see Global Constraints)**

```bash
git add src/ecg/evaluate.py tests/test_evaluate.py
# git commit -m "feat(evaluate): add macro_f1 as the ranking metric"
```

---

### Task 2: `dataset.py` — pair beats with RR features

The network takes two inputs per beat. This module produces them, aligned, with the boundary-beat accounting made explicit rather than implicit.

**Files:**
- Create: `src/ecg/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `features.rr_features(r_peaks, fs) -> (array (n,6) float32, keep_index int64)` — already exists at `src/ecg/features.py:92`
- Produces:
  - `BeatData` frozen dataclass with fields `X: np.ndarray (n,300) float32`, `rr: np.ndarray (n,6) float32`, `y: np.ndarray (n,) int64`, `record: np.ndarray (n,) object`, `n_input: int`, `n_dropped_edge: int`, and property `n_kept -> int`
  - `load_record(beats_dir, record, fs=config.FS) -> BeatData`
  - `load_records(beats_dir, records, fs=config.FS) -> BeatData` — concatenation of the above, in the order given
  - `class_counts(data: BeatData) -> np.ndarray` — length-5 int64 array indexed by AAMI class

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset.py`:

```python
"""Beat + RR pairing. Offline: fixtures write real .npz files into tmp_path."""

from __future__ import annotations

import numpy as np
import pytest

from ecg import config, dataset


def write_beats(directory, record, n_beats, rr_samples=360, labels=None):
    """Write one record's .npz in exactly the layout pipeline.run produces."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(record)) % 2**32)
    X = rng.standard_normal((n_beats, config.BEAT_LENGTH)).astype(np.float32)
    y = (
        np.asarray(labels, dtype=np.int8)
        if labels is not None
        else np.zeros(n_beats, dtype=np.int8)
    )
    r_peaks = (200 + np.arange(n_beats) * rr_samples).astype(np.int64)
    np.savez(directory / f"{record}.npz", X=X, y=y, r_peaks=r_peaks)
    return X, y, r_peaks


def test_load_record_drops_first_and_last_beat(tmp_path):
    write_beats(tmp_path, "901", n_beats=10)
    data = dataset.load_record(tmp_path, "901")
    assert data.n_input == 10
    assert data.n_kept == 8          # both edge beats lack a neighbour
    assert data.n_dropped_edge == 2


def test_load_record_keeps_x_rr_and_y_aligned(tmp_path):
    X, y, _ = write_beats(tmp_path, "901", n_beats=10, labels=list(range(5)) * 2)
    data = dataset.load_record(tmp_path, "901")
    assert data.X.shape == (8, config.BEAT_LENGTH)
    assert data.rr.shape == (8, 6)
    assert data.y.shape == (8,)
    # Row i of the output must be input row i+1 -- the first beat was dropped.
    np.testing.assert_array_equal(data.X[0], X[1])
    np.testing.assert_array_equal(data.y, y[1:9])


def test_rr_values_are_physiological(tmp_path):
    # 360 samples between peaks at 360 Hz == exactly 1.0 s.
    write_beats(tmp_path, "901", n_beats=10, rr_samples=360)
    data = dataset.load_record(tmp_path, "901")
    assert data.rr[:, 0] == pytest.approx(1.0)   # pre_rr
    assert data.rr[:, 4] == pytest.approx(1.0)   # pre_rr_ratio


def test_load_records_concatenates_and_tags_the_record(tmp_path):
    write_beats(tmp_path, "901", n_beats=10)
    write_beats(tmp_path, "902", n_beats=6)
    data = dataset.load_records(tmp_path, ["901", "902"])
    assert data.n_kept == 8 + 4
    assert set(data.record.tolist()) == {"901", "902"}
    assert (data.record[:8] == "901").all()


def test_load_records_rejects_a_duplicate_record(tmp_path):
    write_beats(tmp_path, "901", n_beats=10)
    with pytest.raises(ValueError, match="duplicate"):
        dataset.load_records(tmp_path, ["901", "901"])


def test_load_record_rejects_a_beat_length_mismatch(tmp_path):
    directory = tmp_path
    np.savez(
        directory / "903.npz",
        X=np.zeros((5, 128), dtype=np.float32),
        y=np.zeros(5, dtype=np.int8),
        r_peaks=(200 + np.arange(5) * 360).astype(np.int64),
    )
    with pytest.raises(ValueError, match="beat length"):
        dataset.load_record(directory, "903")


def test_class_counts_covers_every_aami_class(tmp_path):
    write_beats(tmp_path, "901", n_beats=10, labels=[0, 0, 1, 2, 0, 0, 3, 0, 0, 0])
    counts = dataset.class_counts(dataset.load_record(tmp_path, "901"))
    assert counts.shape == (len(config.AAMI_CLASSES),)
    assert counts.sum() == 8
    assert counts[4] == 0          # Q absent, reported as zero not missing
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ecg.dataset'`

- [ ] **Step 3: Implement**

Create `src/ecg/dataset.py`:

```python
"""Pair each beat waveform with its RR timing features.

The hybrid network of Phase 1d takes two inputs per beat: the 300-sample
waveform the Phase 1 pipeline saved, and the 6 RR features describing when the
beat arrived. This module produces them aligned.

**Boundary beats are dropped.** RR features need a neighbour on both sides, so
the first and last beat of every record has none. Dropping them keeps the
denominator identical to the Phase 1b ``rr+morphology`` rows this study is
compared against -- see the spec, section 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, features


@dataclass(frozen=True)
class BeatData:
    """Beats, their RR features, labels, and the record each came from."""

    X: np.ndarray       # (n, BEAT_LENGTH) float32 -- normalized waveforms
    rr: np.ndarray      # (n, 6) float32 -- features.RR_FEATURE_NAMES order
    y: np.ndarray       # (n,) int64 -- AAMI class indices
    record: np.ndarray  # (n,) object -- record id per beat
    n_input: int        # beats before boundary dropping
    n_dropped_edge: int

    @property
    def n_kept(self) -> int:
        return int(self.y.size)

    def reconciles(self) -> bool:
        return self.n_kept + self.n_dropped_edge == self.n_input


def load_record(beats_dir, record: str, fs: int = config.FS) -> BeatData:
    """Load one record's .npz and attach RR features."""
    path = _require(beats_dir, record)
    with np.load(path) as handle:
        X = handle["X"].astype(np.float32)
        y = handle["y"].astype(np.int64)
        r_peaks = handle["r_peaks"].astype(np.int64)

    if X.ndim != 2 or X.shape[1] != config.BEAT_LENGTH:
        raise ValueError(
            f"record {record}: beat length {X.shape[1] if X.ndim == 2 else '?'} "
            f"!= expected {config.BEAT_LENGTH}; the model's convolution stack "
            f"assumes the configured length"
        )
    if not (X.shape[0] == y.size == r_peaks.size):
        raise ValueError(
            f"record {record}: X/y/r_peaks disagree "
            f"({X.shape[0]}, {y.size}, {r_peaks.size})"
        )

    rr, keep = features.rr_features(r_peaks, fs=fs)

    return BeatData(
        X=X[keep],
        rr=rr,
        y=y[keep],
        record=np.full(keep.size, record, dtype=object),
        n_input=int(y.size),
        n_dropped_edge=int(y.size - keep.size),
    )


def load_records(beats_dir, records, fs: int = config.FS) -> BeatData:
    """Load several records and concatenate them, in the order given."""
    names = [str(r) for r in records]
    seen = {n for n in names}
    if len(seen) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"duplicate record ids {duplicates}: the same beats would be "
            f"counted twice, which reads as leakage downstream"
        )
    if not names:
        raise ValueError("no records given")

    parts = [load_record(beats_dir, name, fs=fs) for name in names]

    return BeatData(
        X=np.concatenate([p.X for p in parts]),
        rr=np.concatenate([p.rr for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        record=np.concatenate([p.record for p in parts]),
        n_input=sum(p.n_input for p in parts),
        n_dropped_edge=sum(p.n_dropped_edge for p in parts),
    )


def class_counts(data: BeatData) -> np.ndarray:
    """Beats per AAMI class, including classes with none."""
    return np.bincount(data.y, minlength=len(config.AAMI_CLASSES)).astype(np.int64)


def _require(beats_dir, record: str):
    from pathlib import Path

    path = Path(beats_dir) / f"{record}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no beats for record {record} at {path}. Run: "
            f"./venv/bin/python scripts/preprocess_mitbih.py --strategy ds1_ds2 "
            f"--out-dir ~/physionet-data/processed/mitdb-ds12"
        )
    return path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_dataset.py -v && ./venv/bin/python -m pytest -q`
Expected: PASS, both.

- [ ] **Step 5: Stage**

```bash
git add src/ecg/dataset.py tests/test_dataset.py
# git commit -m "feat(dataset): pair beat waveforms with RR features"
```

---

### Task 3: `folds.py` — record-grouped cross-validation over DS1

The single 4-record validation split holds 81 S beats and zero Q beats — too thin to tune thresholds against. This replaces it with 5 folds so all 944 DS1 S beats serve as validation across the study.

**Files:**
- Create: `src/ecg/folds.py`
- Test: `tests/test_folds.py`

**Interfaces:**
- Consumes: `splits.DS1_RECORDS`, `splits.DS2_RECORDS` (`src/ecg/splits.py`), `config.PACED_RECORDS`
- Produces:
  - `FoldAssignment` frozen dataclass: `folds: tuple[tuple[str, ...], ...]`, `seed: int`, `n_folds: int` property, `validate() -> None`, `train_val(i) -> tuple[tuple[str,...], tuple[str,...]]`, `to_dict() -> dict`, `write_json(path) -> Path`
  - `make_folds(records=splits.DS1_RECORDS, n_folds=5, seed=config.PHASE_1D_SEED) -> FoldAssignment`
- Also adds `PHASE_1D_SEED: int = 20260916` to `src/ecg/config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_folds.py`:

```python
"""Record-grouped CV folds over DS1. The leakage guarantees are the point."""

from __future__ import annotations

import json

import pytest

from ecg import config, folds, splits


def test_folds_partition_ds1_exactly():
    assignment = folds.make_folds()
    flat = [r for fold in assignment.folds for r in fold]
    assert sorted(flat) == sorted(splits.DS1_RECORDS)
    assert len(flat) == len(set(flat))        # no record in two folds


def test_five_folds_over_twenty_two_records_are_balanced():
    assignment = folds.make_folds(n_folds=5)
    sizes = sorted(len(f) for f in assignment.folds)
    assert sizes == [4, 4, 4, 5, 5]


def test_no_ds2_record_can_enter_a_fold():
    assignment = folds.make_folds()
    flat = {r for fold in assignment.folds for r in fold}
    assert not (flat & set(splits.DS2_RECORDS))


def test_validate_rejects_a_ds2_record():
    bad = folds.FoldAssignment(folds=(("101",), ("100",)), seed=1)
    with pytest.raises(ValueError, match="DS2"):
        bad.validate()


def test_validate_rejects_a_record_in_two_folds():
    bad = folds.FoldAssignment(folds=(("101", "106"), ("106",)), seed=1)
    with pytest.raises(ValueError, match="two folds"):
        bad.validate()


def test_validate_rejects_a_paced_record():
    bad = folds.FoldAssignment(folds=(("101",), ("102",)), seed=1)
    with pytest.raises(ValueError, match="paced"):
        bad.validate()


def test_train_val_split_is_complementary():
    assignment = folds.make_folds()
    for i in range(assignment.n_folds):
        train, val = assignment.train_val(i)
        assert set(val) == set(assignment.folds[i])
        assert not (set(train) & set(val))
        assert sorted(set(train) | set(val)) == sorted(splits.DS1_RECORDS)


def test_fold_assignment_is_reproducible_from_the_seed():
    assert folds.make_folds(seed=7).folds == folds.make_folds(seed=7).folds


def test_a_different_seed_gives_a_different_assignment():
    assert folds.make_folds(seed=7).folds != folds.make_folds(seed=8).folds


def test_write_json_round_trips(tmp_path):
    assignment = folds.make_folds()
    path = assignment.write_json(tmp_path / "folds.json")
    payload = json.loads(path.read_text())
    assert payload["seed"] == config.PHASE_1D_SEED
    assert payload["group_unit"] == "record"
    assert len(payload["folds"]) == 5


def test_too_many_folds_is_rejected():
    with pytest.raises(ValueError, match="more folds than records"):
        folds.make_folds(records=["101", "106"], n_folds=5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_folds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ecg.folds'`

- [ ] **Step 3: Implement**

First add to `src/ecg/config.py`, under the Splits section after `DEFAULT_SPLIT_FRACTIONS`:

```python
#: Seed for Phase 1d -- CV fold assignment and network initialisation. Separate
#: from DEFAULT_SPLIT_SEED so re-running Phase 1d cannot disturb the DS1/DS2
#: assignment that Phase 1b's published numbers rest on.
PHASE_1D_SEED: int = 20260916
```

Then create `src/ecg/folds.py`:

```python
"""Record-grouped cross-validation folds over DS1.

``splits.ds1_ds2_split`` carves 4 validation records out of DS1. Measured, those
4 records hold 81 S beats, 26 F beats and zero Q beats -- too thin to tune
per-class decision thresholds against, since one beat flipping moves S recall by
1.2 points.

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
        flat: list[str] = [r for fold in self.folds for r in fold]

        seen: set[str] = set()
        repeated = sorted({r for r in flat if r in seen or seen.add(r)})
        if repeated:
            raise ValueError(
                f"records {repeated} appear in two folds: a patient would be "
                f"both trained on and validated against"
            )

        leaked = sorted(set(flat) & set(splits.DS2_RECORDS))
        if leaked:
            raise ValueError(
                f"DS2 records {leaked} appear in a fold. DS2 is the test set "
                f"and must never enter selection."
            )

        paced = sorted(set(flat) & set(config.PACED_RECORDS))
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
                "fold. DS2 records are excluded by construction."
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
    # set of records, never on the order they were passed in.
    shuffled = list(np.random.default_rng(seed).permutation(ordered))
    chunks = np.array_split(np.arange(len(shuffled)), n_folds)

    assignment = FoldAssignment(
        folds=tuple(tuple(sorted(shuffled[i] for i in chunk)) for chunk in chunks),
        seed=seed,
    )
    assignment.validate()
    return assignment
```

Note on `np.array_split` with 22 records and 5 folds: it yields sizes 5, 5, 4, 4, 4 — the test sorts them, so the assertion `[4, 4, 4, 5, 5]` holds.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_folds.py -v && ./venv/bin/python -m pytest -q`
Expected: PASS, both.

- [ ] **Step 5: Stage**

```bash
git add src/ecg/config.py src/ecg/folds.py tests/test_folds.py
# git commit -m "feat(folds): record-grouped CV folds over DS1"
```

---

### Task 4: Install torch and build the two-branch model

**Files:**
- Modify: `requirements.txt`
- Create: `src/ecg/nn/__init__.py`, `src/ecg/nn/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `config.BEAT_LENGTH` (300), `config.AAMI_CLASSES`, `features.RR_FEATURE_NAMES` (6 names)
- Produces:
  - `HybridBeatNet(n_classes=5, n_rr=6, use_rr=True, dropout=0.3)` — a `torch.nn.Module` whose `forward(beat, rr)` takes `beat: (B, 1, 300)` and `rr: (B, 6)` and returns logits `(B, 5)`. When `use_rr=False` the RR branch is absent and `rr` is ignored.
  - `count_parameters(model) -> int`

- [ ] **Step 1: Install the dependency**

```bash
./venv/bin/pip install torch==2.14.0
```

Then append to `requirements.txt`:

```
# Phase 1d: the hybrid neural classifier. CPU wheels; verified 2026-09-16 to
# install on Python 3.13 arm64 with no source build. Pulls in filelock, jinja2,
# MarkupSafe, networkx, sympy, mpmath and setuptools transitively -- recorded
# here so this file is not read as the complete installed set (cf. Deviation 5).
torch==2.14.0
```

Verify: `./venv/bin/python -c "import torch; print(torch.__version__)"` → `2.14.0`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_model.py`:

```python
"""The two-branch network. Shape, determinism, and that the RR branch is wired."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg import config
from ecg.nn import model as model_module


def make_batch(batch_size=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    beat = torch.randn(batch_size, 1, config.BEAT_LENGTH, generator=generator)
    rr = torch.randn(batch_size, 6, generator=generator)
    return beat, rr


def test_forward_returns_one_logit_per_class():
    net = model_module.HybridBeatNet()
    beat, rr = make_batch()
    out = net(beat, rr)
    assert out.shape == (8, len(config.AAMI_CLASSES))


def test_model_is_deterministic_under_a_fixed_seed():
    beat, rr = make_batch()
    torch.manual_seed(config.PHASE_1D_SEED)
    first = model_module.HybridBeatNet().eval()(beat, rr)
    torch.manual_seed(config.PHASE_1D_SEED)
    second = model_module.HybridBeatNet().eval()(beat, rr)
    torch.testing.assert_close(first, second)


def test_the_rr_branch_actually_changes_the_output():
    """If this fails, the RR branch is decorative and the whole design is void."""
    torch.manual_seed(config.PHASE_1D_SEED)
    net = model_module.HybridBeatNet().eval()
    beat, rr = make_batch()
    out_a = net(beat, rr)
    out_b = net(beat, rr + 5.0)
    assert not torch.allclose(out_a, out_b)


def test_waveform_only_model_ignores_rr():
    torch.manual_seed(config.PHASE_1D_SEED)
    net = model_module.HybridBeatNet(use_rr=False).eval()
    beat, rr = make_batch()
    torch.testing.assert_close(net(beat, rr), net(beat, rr + 5.0))


def test_waveform_only_model_accepts_rr_of_none():
    net = model_module.HybridBeatNet(use_rr=False).eval()
    beat, _ = make_batch()
    assert net(beat, None).shape == (8, 5)


def test_hybrid_model_rejects_rr_of_none():
    net = model_module.HybridBeatNet(use_rr=True).eval()
    beat, _ = make_batch()
    with pytest.raises(ValueError, match="requires rr"):
        net(beat, None)


def test_parameter_count_is_in_the_designed_range():
    n = model_module.count_parameters(model_module.HybridBeatNet())
    assert 50_000 < n < 200_000, f"expected ~100k parameters, got {n:,}"


def test_batch_size_one_works():
    """BatchNorm in train mode fails on a single sample -- eval mode must not."""
    net = model_module.HybridBeatNet().eval()
    beat, rr = make_batch(batch_size=1)
    assert net(beat, rr).shape == (1, 5)


def test_rejects_a_wrong_beat_length():
    net = model_module.HybridBeatNet()
    bad = torch.randn(4, 1, 128)
    with pytest.raises(ValueError, match="beat length"):
        net(bad, torch.randn(4, 6))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ecg.nn'`

- [ ] **Step 4: Implement**

Create `src/ecg/nn/__init__.py`:

```python
"""Neural components for Phase 1d.

Grouped as a subpackage rather than flat modules in ``src/ecg/`` because the
model, its losses and its training loop are tightly coupled and are read
together. A deliberate departure from the flat layout -- see the spec, section 9.
"""
```

Create `src/ecg/nn/model.py`:

```python
"""The two-branch beat classifier.

One branch reads shape, the other reads timing.

**Why two branches and not one input vector.** Phase 1b diagnosed the S-class
failure as feature dilution: 6 RR columns competing against 30 morphology
columns inside one model, with ``max_features='sqrt'`` offering a tree ~6 of 36
columns per split. Here the RR features cannot be crowded out -- they arrive on
their own path and are concatenated at full width with the waveform
representation.

**Why a waveform-only CNN would not do.** ``config.SAMPLES_BEFORE/AFTER`` crops
every beat around its R peak, so each beat is centred and the inter-beat timing
is destroyed by the cropping. A premature beat and an on-time beat produce the
identical window. S is defined by prematurity, so a waveform-only model is blind
to the feature that identifies it. ``use_rr=False`` builds exactly that model, so
the claim is measured rather than asserted.
"""

from __future__ import annotations

import torch
from torch import nn

from .. import config, features


def _conv_block(in_ch: int, out_ch: int, kernel: int, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class HybridBeatNet(nn.Module):
    """1D CNN over the beat, plus a dense branch over the RR features."""

    def __init__(
        self,
        n_classes: int = len(config.AAMI_CLASSES),
        n_rr: int = len(features.RR_FEATURE_NAMES),
        use_rr: bool = True,
        dropout: float = 0.3,
        beat_length: int = config.BEAT_LENGTH,
    ) -> None:
        super().__init__()
        self.use_rr = use_rr
        self.beat_length = beat_length

        self.waveform = nn.Sequential(
            _conv_block(1, 32, 7, pool=True),
            _conv_block(32, 64, 5, pool=True),
            _conv_block(64, 64, 3, pool=True),
            _conv_block(64, 128, 3, pool=False),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        head_in = 128

        if use_rr:
            self.rr_branch = nn.Sequential(
                nn.Linear(n_rr, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )
            head_in += 32
        else:
            self.rr_branch = None

        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, beat: torch.Tensor, rr: torch.Tensor | None) -> torch.Tensor:
        if beat.ndim != 3 or beat.shape[1] != 1:
            raise ValueError(
                f"expected beat of shape (batch, 1, {self.beat_length}), got "
                f"{tuple(beat.shape)}"
            )
        if beat.shape[2] != self.beat_length:
            raise ValueError(
                f"beat length {beat.shape[2]} != expected {self.beat_length}"
            )

        shape = self.waveform(beat)

        if not self.use_rr:
            return self.head(shape)

        if rr is None:
            raise ValueError("this model requires rr features; got None")
        return self.head(torch.cat([shape, self.rr_branch(rr)], dim=1))


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_model.py -v && ./venv/bin/python -m pytest -q`
Expected: PASS. If `test_parameter_count_is_in_the_designed_range` fails, report the actual count rather than widening the bound to fit — the bound encodes the design.

- [ ] **Step 6: Stage**

```bash
git add requirements.txt src/ecg/nn/__init__.py src/ecg/nn/model.py tests/test_model.py
# git commit -m "feat(nn): two-branch beat classifier"
```

---

### Task 5: `losses.py` — weighted cross-entropy, focal loss, capped weights

Arms 1 and 2 differ from arm 0 only in the object built here.

**Files:**
- Create: `src/ecg/nn/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: `config.AAMI_CLASSES`
- Produces:
  - `inverse_frequency_weights(counts, cap=DEFAULT_WEIGHT_CAP) -> np.ndarray` — length-5 float64. Absent classes get weight 0.0. Weights over `cap` are clipped. The result is rescaled so the mean weight over present classes is 1.0.
  - `FocalLoss(gamma=2.0, weight=None)` — a `torch.nn.Module` with `forward(logits, target) -> torch.Tensor` (scalar, mean-reduced)
  - `build_loss(kind, counts, cap=DEFAULT_WEIGHT_CAP, gamma=2.0) -> torch.nn.Module` where `kind` is one of `"plain"`, `"weighted"`, `"focal"`
  - `DEFAULT_WEIGHT_CAP: float = 50.0`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_losses.py`:

```python
"""Loss functions and class weighting."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg.nn import losses


# --- weights ---------------------------------------------------------------

def test_rarer_classes_get_larger_weights():
    w = losses.inverse_frequency_weights([100, 10, 50, 5, 0])
    assert w[1] > w[2] > w[0]
    assert w[3] > w[1]


def test_absent_class_gets_zero_weight_not_infinity():
    """Q has 8 training beats and can be absent from a fold entirely."""
    w = losses.inverse_frequency_weights([100, 10, 50, 5, 0])
    assert w[4] == 0.0
    assert np.isfinite(w).all()


def test_the_cap_is_applied():
    # Without a cap the last class would earn a weight in the hundreds.
    w = losses.inverse_frequency_weights([100_000, 10, 10, 10, 8], cap=50.0)
    assert w.max() <= 50.0


def test_uncapped_q_weight_would_have_been_enormous():
    """The reason the cap exists: 996.0 for 8 beats, from normalization.json."""
    uncapped = losses.inverse_frequency_weights(
        [35501, 863, 3080, 388, 8], cap=float("inf")
    )
    assert uncapped[4] > 100
    capped = losses.inverse_frequency_weights([35501, 863, 3080, 388, 8], cap=50.0)
    assert capped[4] == pytest.approx(capped.max())
    assert capped.max() <= 50.0


def test_weights_are_rescaled_to_mean_one_over_present_classes():
    w = losses.inverse_frequency_weights([100, 10, 50, 5, 0])
    assert w[w > 0].mean() == pytest.approx(1.0)


def test_all_equal_counts_give_all_equal_weights():
    w = losses.inverse_frequency_weights([10, 10, 10, 10, 10])
    assert w == pytest.approx(np.ones(5))


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        losses.inverse_frequency_weights([10, -1, 10, 10, 10])


# --- focal loss ------------------------------------------------------------

def test_focal_loss_at_gamma_zero_equals_cross_entropy():
    """The identity that proves the implementation is the real focal loss."""
    torch.manual_seed(0)
    logits = torch.randn(32, 5)
    target = torch.randint(0, 5, (32,))
    focal = losses.FocalLoss(gamma=0.0)(logits, target)
    plain = torch.nn.CrossEntropyLoss()(logits, target)
    torch.testing.assert_close(focal, plain)


def test_weighted_focal_at_gamma_zero_equals_weighted_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(32, 5)
    target = torch.randint(0, 5, (32,))
    w = torch.tensor([1.0, 4.0, 2.0, 8.0, 0.0])
    focal = losses.FocalLoss(gamma=0.0, weight=w)(logits, target)
    plain = torch.nn.CrossEntropyLoss(weight=w)(logits, target)
    torch.testing.assert_close(focal, plain)


def test_focal_loss_down_weights_confident_correct_predictions():
    confident = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])
    plain = torch.nn.CrossEntropyLoss()(confident, target)
    focal = losses.FocalLoss(gamma=2.0)(confident, target)
    assert focal < plain


def test_focal_loss_returns_a_scalar():
    loss = losses.FocalLoss(gamma=2.0)(torch.randn(8, 5), torch.randint(0, 5, (8,)))
    assert loss.ndim == 0


def test_negative_gamma_is_rejected():
    with pytest.raises(ValueError, match="gamma"):
        losses.FocalLoss(gamma=-1.0)


# --- factory ---------------------------------------------------------------

def test_build_loss_plain_ignores_the_counts():
    loss_fn = losses.build_loss("plain", counts=[100, 1, 1, 1, 1])
    assert getattr(loss_fn, "weight", None) is None


def test_build_loss_weighted_carries_the_weights():
    loss_fn = losses.build_loss("weighted", counts=[100, 10, 50, 5, 0])
    assert loss_fn.weight is not None
    assert loss_fn.weight[4] == 0.0


def test_build_loss_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown loss"):
        losses.build_loss("hinge", counts=[1, 1, 1, 1, 1])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_losses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ecg.nn.losses'`

- [ ] **Step 3: Implement**

Create `src/ecg/nn/losses.py`:

```python
"""Loss functions for the imbalance study.

Arms 0, 1 and 2 of Phase 1d differ only in the object this module builds.

**Why weights are capped.** ``normalization.json`` for the DS1/DS2 split
suggests a weight of 996.0 for class Q, which has 8 training beats. Applied
literally, those 8 beats would dominate the gradient and destabilise training.
The cap bounds any one class's influence; the cap value itself is selected by
cross-validation, never asserted.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .. import config

#: Starting point for the weight cap. Selected properly by cross-validation.
DEFAULT_WEIGHT_CAP: float = 50.0

LOSS_KINDS = ("plain", "weighted", "focal")


def inverse_frequency_weights(
    counts,
    cap: float = DEFAULT_WEIGHT_CAP,
) -> np.ndarray:
    """Per-class weights inversely proportional to frequency, capped.

    A class with no beats gets weight 0.0 rather than infinity -- it cannot
    contribute to the loss, so its weight is arbitrary, and 0.0 is the value
    that keeps the array finite.

    The result is rescaled so the mean weight over present classes is 1.0. That
    keeps the loss on the same scale as the unweighted arm, so arm 0 and arm 1
    can be compared without the learning rate meaning something different in
    each.
    """
    raw = np.asarray(counts, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {raw.shape}")
    if np.any(raw < 0):
        raise ValueError(f"negative class count in {raw.tolist()}")
    if raw.sum() <= 0:
        raise ValueError("no beats in any class")
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    present = raw > 0
    weights = np.zeros_like(raw)

    # total / (n_present * count): the standard "balanced" formula, restricted
    # to classes that are actually present.
    weights[present] = raw.sum() / (present.sum() * raw[present])
    weights = np.minimum(weights, cap)

    mean_present = weights[present].mean()
    if mean_present > 0:
        weights[present] /= mean_present
        weights = np.minimum(weights, cap)

    return weights


class FocalLoss(nn.Module):
    """Cross-entropy scaled by ``(1 - p_t) ** gamma``.

    At ``gamma=0`` this is exactly cross-entropy; ``test_losses.py`` asserts
    that identity, which is what proves the scaling term is right.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = float(gamma)
        self.register_buffer(
            "weight", weight if weight is None else weight.to(torch.float32)
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # reduction="none" so the focal term is applied per sample before any
        # averaging -- averaging first would make the term a no-op.
        ce = nn.functional.cross_entropy(
            logits, target, weight=self.weight, reduction="none"
        )

        # p_t recovered from the unweighted log-probability, so the class weight
        # scales the loss without also distorting the focal term.
        log_pt = nn.functional.log_softmax(logits, dim=1).gather(
            1, target.unsqueeze(1)
        ).squeeze(1)
        focal = (1.0 - log_pt.exp()).clamp(min=0.0) ** self.gamma

        loss = focal * ce

        if self.weight is None:
            return loss.mean()

        # Weighted cross-entropy normalises by the sum of weights, not the
        # count. Matching that is what makes the gamma=0 identity hold.
        denominator = self.weight[target].sum()
        if denominator <= 0:
            return loss.sum() * 0.0
        return loss.sum() / denominator


def build_loss(
    kind: str,
    counts,
    cap: float = DEFAULT_WEIGHT_CAP,
    gamma: float = 2.0,
) -> nn.Module:
    """Build the loss for one arm of the study."""
    if kind not in LOSS_KINDS:
        raise ValueError(f"unknown loss {kind!r}; expected one of {list(LOSS_KINDS)}")

    if kind == "plain":
        return nn.CrossEntropyLoss()

    weight = torch.tensor(inverse_frequency_weights(counts, cap), dtype=torch.float32)

    if kind == "weighted":
        return nn.CrossEntropyLoss(weight=weight)
    return FocalLoss(gamma=gamma, weight=weight)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_losses.py -v && ./venv/bin/python -m pytest -q`
Expected: PASS. The gamma=0 identity tests are the load-bearing ones — if either fails, the focal implementation is wrong and no amount of tuning will rescue the arm.

- [ ] **Step 5: Stage**

```bash
git add src/ecg/nn/losses.py tests/test_losses.py
# git commit -m "feat(nn): weighted CE, focal loss, capped class weights"
```

---

### Task 6: `train.py` — the training loop

**Files:**
- Create: `src/ecg/nn/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `dataset.BeatData` (Task 2), `model.HybridBeatNet` (Task 4), `losses.build_loss` (Task 5), `evaluate.evaluate` with `macro_f1` (Task 1)
- Produces:
  - `TrainConfig` dataclass: `epochs=50`, `lr=1e-3`, `batch_size=256`, `patience=8`, `seed=config.PHASE_1D_SEED`, `dropout=0.3`, `oversample=False`, `use_rr=True`
  - `TrainResult` dataclass: `state_dict: dict`, `best_epoch: int`, `best_val_macro_f1: float`, `history: tuple[dict, ...]`, `val_proba: np.ndarray (n_val, 5)`, `val_y: np.ndarray (n_val,)`, `val_record: np.ndarray (n_val,)`
  - `train_model(train_data, val_data, loss_kind, cfg=TrainConfig(), seed=None) -> TrainResult` — the weight cap and focal gamma come from `cfg`, not from separate arguments, so an arm cannot accidentally differ in two places at once
  - `predict_proba(model, data, batch_size=512) -> np.ndarray (n, 5)`
  - `fit_final(train_data, loss_kind, cfg, epochs) -> dict` — fixed-epoch fit with no validation, returning a state dict

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train.py`:

```python
"""Training loop: smoke, determinism, early stopping, checkpoint round-trip."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg import config, dataset
from ecg.nn import model as model_module
from ecg.nn import train


def make_data(n=200, seed=0, s_fraction=0.2):
    """Small two-class-ish dataset where S is marked by a short pre_rr."""
    rng = np.random.default_rng(seed)
    n_s = int(n * s_fraction)
    y = np.zeros(n, dtype=np.int64)
    y[:n_s] = 1
    rng.shuffle(y)

    X = rng.standard_normal((n, config.BEAT_LENGTH)).astype(np.float32)
    rr = np.ones((n, 6), dtype=np.float32) + 0.01 * rng.standard_normal((n, 6))
    rr[y == 1, 0] = 0.5          # premature: short pre_rr
    rr[y == 1, 4] = 0.5          # and a low pre_rr_ratio

    return dataset.BeatData(
        X=X,
        rr=rr.astype(np.float32),
        y=y,
        record=np.full(n, "sim", dtype=object),
        n_input=n,
        n_dropped_edge=0,
    )


def test_one_epoch_runs_and_returns_a_result():
    cfg = train.TrainConfig(epochs=1, batch_size=32)
    result = train.train_model(make_data(), make_data(seed=1), "plain", cfg)
    assert result.best_epoch == 1
    assert len(result.history) == 1
    assert 0.0 <= result.best_val_macro_f1 <= 1.0


def test_val_probabilities_have_the_right_shape_and_sum_to_one():
    cfg = train.TrainConfig(epochs=1, batch_size=32)
    val = make_data(n=120, seed=1)
    result = train.train_model(make_data(), val, "plain", cfg)
    assert result.val_proba.shape == (120, len(config.AAMI_CLASSES))
    np.testing.assert_allclose(result.val_proba.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(result.val_y, val.y)


def test_training_is_deterministic_under_a_fixed_seed():
    cfg = train.TrainConfig(epochs=2, batch_size=32, seed=123)
    a = train.train_model(make_data(), make_data(seed=1), "plain", cfg)
    b = train.train_model(make_data(), make_data(seed=1), "plain", cfg)
    np.testing.assert_allclose(a.val_proba, b.val_proba, atol=1e-6)


def test_early_stopping_halts_before_the_epoch_budget():
    # patience=1 on a task with no signal to keep improving on.
    cfg = train.TrainConfig(epochs=30, batch_size=32, patience=1)
    result = train.train_model(make_data(), make_data(seed=1), "plain", cfg)
    assert len(result.history) < 30


def test_the_returned_checkpoint_reproduces_the_returned_probabilities():
    cfg = train.TrainConfig(epochs=2, batch_size=32)
    val = make_data(n=120, seed=1)
    result = train.train_model(make_data(), val, "plain", cfg)

    net = model_module.HybridBeatNet(dropout=cfg.dropout)
    net.load_state_dict(result.state_dict)
    reloaded = train.predict_proba(net, val)

    np.testing.assert_allclose(reloaded, result.val_proba, atol=1e-6)


def test_oversampling_changes_the_batches_not_the_validation_set():
    cfg = train.TrainConfig(epochs=1, batch_size=32, oversample=True)
    val = make_data(n=120, seed=1)
    result = train.train_model(make_data(), val, "plain", cfg)
    assert result.val_proba.shape[0] == 120


def test_waveform_only_training_runs():
    cfg = train.TrainConfig(epochs=1, batch_size=32, use_rr=False)
    result = train.train_model(make_data(), make_data(seed=1), "plain", cfg)
    assert result.val_proba.shape[1] == len(config.AAMI_CLASSES)


def test_fit_final_returns_a_loadable_state_dict():
    cfg = train.TrainConfig(epochs=50, batch_size=32)
    state = train.fit_final(make_data(), "plain", cfg, epochs=2)
    model_module.HybridBeatNet(dropout=cfg.dropout).load_state_dict(state)


def test_empty_training_data_is_rejected():
    empty = dataset.BeatData(
        X=np.zeros((0, config.BEAT_LENGTH), dtype=np.float32),
        rr=np.zeros((0, 6), dtype=np.float32),
        y=np.zeros(0, dtype=np.int64),
        record=np.empty(0, dtype=object),
        n_input=0,
        n_dropped_edge=0,
    )
    with pytest.raises(ValueError, match="empty"):
        train.train_model(empty, make_data(), "plain", train.TrainConfig(epochs=1))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ecg.nn.train'`

- [ ] **Step 3: Implement**

Create `src/ecg/nn/train.py`:

```python
"""Training loop for the Phase 1d arms.

Every arm shares this loop. What differs between arms is the loss object and two
flags -- ``oversample`` and ``use_rr``. Keeping the loop identical is what makes
the comparison a comparison: if arms differed in their optimiser or epoch budget
too, no row could be attributed to its one named change.

Selection is on validation macro-F1, never macro recall: macro recall rewards
over-predicting a rare class with no offsetting penalty, and ranked the worst of
six Phase 1b models first.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .. import config, evaluate
from ..dataset import BeatData, class_counts
from . import losses as losses_module
from .model import HybridBeatNet


@dataclass(frozen=True)
class TrainConfig:
    """Everything held constant across arms, plus the two flags that are not."""

    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 256
    patience: int = 8
    seed: int = config.PHASE_1D_SEED
    dropout: float = 0.3
    weight_cap: float = losses_module.DEFAULT_WEIGHT_CAP
    gamma: float = 2.0
    oversample: bool = False     # arm 4
    use_rr: bool = True          # arm 5 sets this False


@dataclass(frozen=True)
class TrainResult:
    """The trained model plus everything needed to score and audit it."""

    state_dict: dict
    best_epoch: int
    best_val_macro_f1: float
    history: tuple[dict, ...]
    val_proba: np.ndarray
    val_y: np.ndarray
    val_record: np.ndarray
    train_counts: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.int64))


def _tensors(data: BeatData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beat = torch.from_numpy(np.ascontiguousarray(data.X)).unsqueeze(1).float()
    rr = torch.from_numpy(np.ascontiguousarray(data.rr)).float()
    y = torch.from_numpy(np.ascontiguousarray(data.y)).long()
    return beat, rr, y


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


def _loader(data: BeatData, cfg: TrainConfig, generator: torch.Generator) -> DataLoader:
    beat, rr, y = _tensors(data)
    tensor_set = TensorDataset(beat, rr, y)

    if not cfg.oversample:
        return DataLoader(
            tensor_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

    # Arm 4: sample rare classes more often. Note this duplicates beats from the
    # same patients, which is close to the patient-memorisation failure recorded
    # as Problem #9 -- a win here is to be checked, not celebrated.
    counts = np.bincount(data.y, minlength=len(config.AAMI_CLASSES)).astype(np.float64)
    per_class = np.zeros_like(counts)
    present = counts > 0
    per_class[present] = 1.0 / counts[present]
    sample_weights = torch.from_numpy(per_class[data.y]).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(tensor_set),
        replacement=True,
        generator=generator,
    )
    return DataLoader(tensor_set, batch_size=cfg.batch_size, sampler=sampler)


def predict_proba(model: nn.Module, data: BeatData, batch_size: int = 512) -> np.ndarray:
    """Class probabilities for every beat, in input order."""
    beat, rr, _ = _tensors(data)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, beat.shape[0], batch_size):
            stop = start + batch_size
            logits = model(beat[start:stop], rr[start:stop] if model.use_rr else None)
            chunks.append(torch.softmax(logits, dim=1).numpy())
    if not chunks:
        return np.zeros((0, len(config.AAMI_CLASSES)), dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def train_model(
    train_data: BeatData,
    val_data: BeatData,
    loss_kind: str,
    cfg: TrainConfig = TrainConfig(),
    seed: int | None = None,
) -> TrainResult:
    """Train one arm on one fold, early-stopping on validation macro-F1."""
    if train_data.n_kept == 0:
        raise ValueError("empty training set")
    if val_data.n_kept == 0:
        raise ValueError("empty validation set")

    seed = cfg.seed if seed is None else seed
    _seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)

    counts = class_counts(train_data)
    model = HybridBeatNet(dropout=cfg.dropout, use_rr=cfg.use_rr)
    loss_fn = losses_module.build_loss(
        loss_kind, counts, cap=cfg.weight_cap, gamma=cfg.gamma
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = _loader(train_data, cfg, generator)

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_proba = None
    history: list[dict] = []
    since_improved = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for beat, rr, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(beat, rr if cfg.use_rr else None), y)
            loss.backward()
            optimizer.step()
            running += float(loss) * y.shape[0]

        proba = predict_proba(model, val_data)
        result = evaluate.evaluate(val_data.y, proba.argmax(axis=1))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / train_data.n_kept,
                "val_macro_f1": result.macro_f1,
                "val_macro_recall": result.macro_recall,
            }
        )

        if result.macro_f1 > best_score:
            best_score = result.macro_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_proba = proba
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= cfg.patience:
                break

    assert best_proba is not None  # at least one epoch always runs

    return TrainResult(
        state_dict=best_state,
        best_epoch=best_epoch,
        best_val_macro_f1=float(best_score),
        history=tuple(history),
        val_proba=best_proba,
        val_y=val_data.y.copy(),
        val_record=val_data.record.copy(),
        train_counts=counts,
    )


def fit_final(
    train_data: BeatData,
    loss_kind: str,
    cfg: TrainConfig,
    epochs: int,
) -> dict:
    """Fit on all of DS1 for a fixed number of epochs.

    There is no held-out set left at this point, so there is nothing to
    early-stop against. ``epochs`` is the median stopping epoch across the
    arm's folds. This substitution must be stated in the results, not glossed --
    see the spec, section 6 step 5.
    """
    if train_data.n_kept == 0:
        raise ValueError("empty training set")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    _seed_everything(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    model = HybridBeatNet(dropout=cfg.dropout, use_rr=cfg.use_rr)
    loss_fn = losses_module.build_loss(
        loss_kind, class_counts(train_data), cap=cfg.weight_cap, gamma=cfg.gamma
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = _loader(train_data, cfg, generator)

    for _ in range(epochs):
        model.train()
        for beat, rr, y in loader:
            optimizer.zero_grad()
            loss_fn(model(beat, rr if cfg.use_rr else None), y).backward()
            optimizer.step()

    return copy.deepcopy(model.state_dict())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_train.py -v && ./venv/bin/python -m pytest -q`
Expected: PASS. These tests run real training on 200 synthetic beats; the file should finish in well under a minute. If it takes minutes, the batch size or epoch count in the fixtures is wrong.

- [ ] **Step 5: Stage**

```bash
git add src/ecg/nn/train.py tests/test_train.py
# git commit -m "feat(nn): training loop with early stopping on macro-F1"
```

---

### Task 7: The imbalance regression test — verified to fail

This is the most important test in the phase and it gets its own task because of how it must be built. Problem #10 records a regression test that passed with both the buggy and the fixed parameter, and therefore asserted nothing; the missing ingredient was class imbalance in the fixture. **This test is not accepted until it has been observed failing on arm 0's configuration.**

**Files:**
- Create: `tests/test_imbalance_regression.py`

**Interfaces:**
- Consumes: `train.train_model`, `train.TrainConfig` (Task 6), `dataset.BeatData` (Task 2), `evaluate.evaluate` (Task 1)
- Produces: nothing importable — this is a test-only task

- [ ] **Step 1: Write the fixture and the test**

Create `tests/test_imbalance_regression.py`:

```python
"""The regression test for imbalance handling.

**Read this before changing it.** Problem #10 in PlanDoc/IMPLEMENTATION.md
records a regression test that passed with both the buggy parameter and the
fixed one, and so asserted nothing. The missing ingredient was class imbalance:
only once S prevalence matched the real split did the failure reproduce.

The fixture below therefore mirrors the real DS1 training split: S is ~2.2% of
beats (863 of 39,840), it is identified by timing rather than shape, and its
waveform is drawn from the same distribution as N so there is nothing to
memorise. Under plain cross-entropy the model should ignore S almost entirely.

Step 2 of this task requires observing the assertion FAIL under plain
cross-entropy before the weighted case is accepted as passing.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg import config, dataset, evaluate
from ecg.nn import train

S_PREVALENCE = 0.022          # 863 / 39,840 in the real DS1 training split


def imbalanced_beats(n=4000, seed=0, s_prevalence=S_PREVALENCE):
    """N and S with identical waveform distributions, separated only by timing."""
    rng = np.random.default_rng(seed)
    n_s = max(int(round(n * s_prevalence)), 1)

    y = np.zeros(n, dtype=np.int64)
    y[:n_s] = 1
    rng.shuffle(y)

    # Waveforms carry no class information at all -- if the model succeeds, it
    # succeeded on timing, which is the property under test.
    X = rng.standard_normal((n, config.BEAT_LENGTH)).astype(np.float32)

    rr = np.empty((n, 6), dtype=np.float32)
    normal_rr = 0.85 + 0.05 * rng.standard_normal(n)
    rr[:, 0] = np.where(y == 1, 0.50 + 0.04 * rng.standard_normal(n), normal_rr)
    rr[:, 1] = np.where(y == 1, 1.15 + 0.05 * rng.standard_normal(n), normal_rr)
    rr[:, 2] = normal_rr
    rr[:, 3] = 0.85
    rr[:, 4] = rr[:, 0] / 0.85
    rr[:, 5] = rr[:, 1] / 0.85

    return dataset.BeatData(
        X=X,
        rr=rr.astype(np.float32),
        y=y,
        record=np.full(n, "sim", dtype=object),
        n_input=n,
        n_dropped_edge=0,
    )


def s_recall(loss_kind, seed, cfg_kwargs=None):
    cfg = train.TrainConfig(epochs=8, batch_size=128, patience=8, seed=seed,
                            **(cfg_kwargs or {}))
    result = train.train_model(
        imbalanced_beats(seed=seed),
        imbalanced_beats(n=2000, seed=seed + 100),
        loss_kind,
        cfg,
    )
    scored = evaluate.evaluate(result.val_y, result.val_proba.argmax(axis=1))
    return scored.recall_of("S")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_plain_cross_entropy_ignores_the_minority_class(seed):
    """Arm 0's expected behaviour: the collapse this study exists to measure."""
    assert s_recall("plain", seed) < 0.35


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_class_weighting_recovers_the_minority_class(seed):
    """Arm 1 must find what arm 0 misses. If this passes for arm 0 too, the
    fixture is not imbalanced enough and the test asserts nothing."""
    assert s_recall("weighted", seed) > 0.60


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_focal_loss_recovers_the_minority_class(seed):
    assert s_recall("focal", seed) > 0.60


def test_weighting_beats_plain_on_every_seed():
    """The comparison stated directly, so a marginal fixture cannot hide it."""
    for seed in (1, 2, 3):
        assert s_recall("weighted", seed) > s_recall("plain", seed)
```

- [ ] **Step 2: Prove the test can fail — this step is mandatory**

Temporarily change `test_class_weighting_recovers_the_minority_class` to pass `"plain"` instead of `"weighted"`, then run:

```bash
./venv/bin/python -m pytest tests/test_imbalance_regression.py -k class_weighting -v
```

Expected: **FAIL** on all three seeds, with S recall below 0.35.

If it PASSES with `"plain"`, the fixture does not reproduce the problem and the test is worthless — as in Problem #10. In that case increase the imbalance (lower `S_PREVALENCE` toward 0.01) or reduce the timing separation, and repeat this step until it fails. **Do not proceed until this step has failed as expected.** Then revert the temporary change.

- [ ] **Step 3: Run the real test**

Run: `./venv/bin/python -m pytest tests/test_imbalance_regression.py -v`
Expected: PASS, all 10 tests.

If the weighted case cannot clear 0.60, do not lower the threshold to make it pass. Report the observed numbers — the arm is telling you something about the design.

- [ ] **Step 4: Record the observation**

Write the observed S recalls (plain vs weighted vs focal, all three seeds) into a scratch note for Task 11's writeup. These numbers are the evidence that the test discriminates, and Problem #10 exists because that evidence was not captured last time.

- [ ] **Step 5: Stage**

```bash
git add tests/test_imbalance_regression.py
# git commit -m "test: imbalance regression, verified to fail under plain CE"
```

---

### Task 8: `thresholds.py` — per-class decision tuning

Arm 3. Trains nothing: it re-scores saved out-of-fold probabilities. Expected to be the largest single gain for S, because `pre_rr_ratio` alone separates S at 0.972 recall while argmax over the softmax does not.

**Files:**
- Create: `src/ecg/thresholds.py`
- Test: `tests/test_thresholds.py`

**Interfaces:**
- Consumes: `evaluate.evaluate` with `macro_f1` (Task 1)
- Produces:
  - `apply_thresholds(proba, thresholds) -> np.ndarray` — predicted class indices; `argmax(proba / thresholds)`
  - `tune_thresholds(y_true, proba, grid=DEFAULT_GRID, rounds=3) -> np.ndarray` — length-5 float64, coordinate ascent on macro-F1
  - `DEFAULT_GRID: tuple[float, ...]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_thresholds.py`:

```python
"""Per-class decision thresholds, tuned out-of-fold."""

from __future__ import annotations

import numpy as np
import pytest

from ecg import config, evaluate, thresholds


def test_thresholds_of_one_reproduce_argmax():
    proba = np.array([[0.6, 0.3, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0]])
    ones = np.ones(len(config.AAMI_CLASSES))
    np.testing.assert_array_equal(
        thresholds.apply_thresholds(proba, ones), proba.argmax(axis=1)
    )


def test_lowering_a_threshold_makes_that_class_easier_to_predict():
    proba = np.array([[0.6, 0.4, 0.0, 0.0, 0.0]])
    t = np.ones(len(config.AAMI_CLASSES))
    assert thresholds.apply_thresholds(proba, t)[0] == 0
    t[1] = 0.5                    # S needs only half the evidence now
    assert thresholds.apply_thresholds(proba, t)[0] == 1


def test_tuning_improves_macro_f1_on_an_imbalanced_set():
    rng = np.random.default_rng(0)
    n = 2000
    y = np.zeros(n, dtype=np.int64)
    y[: int(n * 0.03)] = 1
    rng.shuffle(y)

    # S is genuinely separable but the model is under-confident about it, so
    # argmax misses most of the class -- exactly the real situation.
    proba = np.zeros((n, len(config.AAMI_CLASSES)))
    proba[:, 0] = np.where(y == 1, 0.55, 0.95)
    proba[:, 1] = 1.0 - proba[:, 0]

    before = evaluate.evaluate(y, proba.argmax(axis=1)).macro_f1
    tuned = thresholds.tune_thresholds(y, proba)
    after = evaluate.evaluate(y, thresholds.apply_thresholds(proba, tuned)).macro_f1

    assert after > before


def test_tuning_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=500)
    proba = rng.dirichlet(np.ones(5), size=500)
    np.testing.assert_array_equal(
        thresholds.tune_thresholds(y, proba), thresholds.tune_thresholds(y, proba)
    )


def test_absent_class_threshold_stays_at_one():
    """Q is absent from most folds. Tuning must not invent a threshold for it."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, size=400)          # no class 3 or 4
    proba = rng.dirichlet(np.ones(5), size=400)
    tuned = thresholds.tune_thresholds(y, proba)
    assert tuned[4] == 1.0


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        thresholds.tune_thresholds(np.zeros(10, dtype=np.int64), np.zeros((9, 5)))


def test_thresholds_must_be_positive():
    proba = np.array([[0.6, 0.4, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="positive"):
        thresholds.apply_thresholds(proba, np.array([1.0, 0.0, 1.0, 1.0, 1.0]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_thresholds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ecg.thresholds'`

- [ ] **Step 3: Implement**

Create `src/ecg/thresholds.py`:

```python
"""Per-class decision thresholds.

``argmax`` over a softmax is a decision rule, and on an imbalanced problem it is
usually the wrong one: a class that is 2% of the data has to overcome a prior
that argmax takes at face value. Dividing each class's probability by a
per-class threshold before the argmax moves that boundary.

This costs no retraining -- it re-scores probabilities a model has already
produced. Given that a single threshold on ``pre_rr_ratio`` separates S at 0.972
recall while the trained model's argmax reaches 0.271, this is the cheapest
lever in the phase.

**Thresholds are tuned on pooled out-of-fold predictions only.** Tuning them on
the test split would be selecting on DS2, which the spec forbids.
"""

from __future__ import annotations

import numpy as np

from . import config, evaluate

#: Candidate thresholds. Below 1.0 makes a class easier to predict; above,
#: harder. Log-spaced because the useful range for a 2% class is well under 1.
DEFAULT_GRID: tuple[float, ...] = (
    0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
    0.70, 1.00, 1.50, 2.00, 3.00,
)


def apply_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Predict by ``argmax(proba / thresholds)``."""
    p = np.asarray(proba, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)

    if p.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {p.shape}")
    if t.shape != (p.shape[1],):
        raise ValueError(
            f"shape mismatch: {p.shape[1]} classes but {t.shape} thresholds"
        )
    if np.any(t <= 0):
        raise ValueError(f"thresholds must be positive, got {t.tolist()}")

    return (p / t).argmax(axis=1).astype(np.int64)


def tune_thresholds(
    y_true,
    proba: np.ndarray,
    grid: tuple[float, ...] = DEFAULT_GRID,
    rounds: int = 3,
    classes: tuple[str, ...] = config.AAMI_CLASSES,
) -> np.ndarray:
    """Coordinate ascent on macro-F1 over per-class thresholds.

    One class at a time, the best grid value is chosen with the others held
    fixed; ``rounds`` passes let earlier choices be revised. Deterministic: the
    grid is fixed and ties keep the incumbent.

    A class absent from ``y_true`` keeps a threshold of 1.0. Q has 8 training
    beats and is absent from most folds; inventing a threshold from no positive
    examples would be fitting noise.
    """
    true = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)

    if p.ndim != 2 or true.shape[0] != p.shape[0]:
        raise ValueError(
            f"shape mismatch: y_true {true.shape} vs proba {p.shape}"
        )
    if true.size == 0:
        raise ValueError("cannot tune on an empty set")

    n_classes = p.shape[1]
    present = np.bincount(true, minlength=n_classes) > 0

    best = np.ones(n_classes, dtype=np.float64)
    best_score = evaluate.evaluate(
        true, apply_thresholds(p, best), classes=classes
    ).macro_f1

    for _ in range(rounds):
        improved = False
        for k in range(n_classes):
            if not present[k]:
                continue
            for candidate in grid:
                trial = best.copy()
                trial[k] = candidate
                score = evaluate.evaluate(
                    true, apply_thresholds(p, trial), classes=classes
                ).macro_f1
                if score > best_score:
                    best_score, best, improved = score, trial, True
        if not improved:
            break

    return best
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_thresholds.py -v && ./venv/bin/python -m pytest -q`
Expected: PASS, both.

- [ ] **Step 5: Stage**

```bash
git add src/ecg/thresholds.py tests/test_thresholds.py
# git commit -m "feat(thresholds): per-class decision tuning on out-of-fold predictions"
```

---

### Task 9: `scripts/train_nn.py` — run the study

Orchestration only: no new algorithms. It runs 25 fold fits, selects on the fold means, fits 5 final models, and scores DS2 once.

**Files:**
- Create: `scripts/train_nn.py`
- Test: `tests/test_train_nn_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8
- Produces:
  - `ARMS: tuple[Arm, ...]` where `Arm` is a frozen dataclass with `name: str`, `loss: str`, `oversample: bool`, `use_rr: bool`, `tune: bool`
  - `run_cv(beats_dir, assignment, arm, cfg) -> dict` — per-arm fold results, including pooled out-of-fold probabilities
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_nn_cli.py`:

```python
"""The orchestration script. Structure and guardrails, not the numbers."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_nn                                       # noqa: E402
from ecg import splits                                # noqa: E402


def test_there_are_six_arms():
    assert len(train_nn.ARMS) == 6


def test_arm_names_are_unique():
    names = [a.name for a in train_nn.ARMS]
    assert len(names) == len(set(names))


def test_exactly_one_arm_has_no_imbalance_handling():
    plain = [a for a in train_nn.ARMS if a.loss == "plain" and not a.oversample]
    assert len(plain) == 1
    assert plain[0].name == "plain_ce"


def test_exactly_one_arm_is_waveform_only():
    waveform_only = [a for a in train_nn.ARMS if not a.use_rr]
    assert len(waveform_only) == 1


def test_the_threshold_arm_trains_nothing_of_its_own():
    tuned = [a for a in train_nn.ARMS if a.tune]
    assert len(tuned) == 1
    assert tuned[0].reuses is not None


def test_missing_data_names_the_command_that_produces_it(tmp_path, capsys):
    code = train_nn.main(["--beats-dir", str(tmp_path / "nope")])
    assert code != 0
    assert "preprocess_mitbih.py" in capsys.readouterr().err


def test_ds2_records_never_appear_in_any_fold():
    from ecg import folds

    assignment = folds.make_folds()
    flat = {r for fold in assignment.folds for r in fold}
    assert not (flat & set(splits.DS2_RECORDS))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_train_nn_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_nn'`

- [ ] **Step 3: Implement**

Create `scripts/train_nn.py`:

```python
#!/usr/bin/env python
"""Run the Phase 1d imbalance study.

Six rows, one variable each:

    plain_ce        no imbalance handling at all -- the "before"
    weighted_ce     class-weighted cross-entropy, capped weights
    focal           focal loss, gamma=2
    tuned           the better of weighted_ce / focal, plus per-class thresholds
    oversample      rare beats resampled to balance the batches
    waveform_only   the RR branch removed, best imbalance setting

Protocol, from the spec section 6: every decision is made on cross-validation
over DS1. DS2 is scored exactly once, at the end, for reporting only.

Usage:
    ./venv/bin/python scripts/train_nn.py \
        --beats-dir ~/physionet-data/processed/mitdb-ds12/beats \
        --out-dir   ~/physionet-data/processed/mitdb-ds12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg import config, dataset, evaluate, folds, splits, thresholds  # noqa: E402
from ecg.nn import model as model_module                              # noqa: E402
from ecg.nn import train                                              # noqa: E402


@dataclass(frozen=True)
class Arm:
    """One row of the study. Exactly one thing differs from plain_ce."""

    name: str
    loss: str
    oversample: bool = False
    use_rr: bool = True
    tune: bool = False
    reuses: str | None = None      # set only for the threshold arm
    note: str = ""


ARMS: tuple[Arm, ...] = (
    Arm("plain_ce", "plain", note="no imbalance handling -- the before-picture"),
    Arm("weighted_ce", "weighted", note="class-weighted CE, capped weights"),
    Arm("focal", "focal", note="focal loss, gamma=2"),
    Arm(
        "tuned",
        "weighted",
        tune=True,
        reuses="best_of(weighted_ce, focal)",
        note="per-class thresholds on pooled out-of-fold predictions",
    ),
    Arm("oversample", "plain", oversample=True, note="rare beats resampled"),
    Arm(
        "waveform_only",
        "weighted",
        use_rr=False,
        note="RR branch removed -- isolates architecture from imbalance",
    ),
)


def run_cv(beats_dir, assignment: folds.FoldAssignment, arm: Arm, cfg: train.TrainConfig) -> dict:
    """Train one arm across every fold; return fold scores and pooled OOF output."""
    fold_rows: list[dict] = []
    oof_proba: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []

    for index in range(assignment.n_folds):
        train_records, val_records = assignment.train_val(index)
        train_data = dataset.load_records(beats_dir, train_records)
        val_data = dataset.load_records(beats_dir, val_records)

        started = time.time()
        result = train.train_model(train_data, val_data, arm.loss, cfg)
        scored = evaluate.evaluate(
            result.val_y, result.val_proba.argmax(axis=1), label=f"{arm.name} fold{index}"
        )

        fold_rows.append(
            {
                "fold": index,
                "val_records": list(val_records),
                "best_epoch": result.best_epoch,
                "macro_f1": scored.macro_f1,
                "macro_recall": scored.macro_recall,
                "s_recall": scored.recall_of("S"),
                "v_recall": scored.recall_of("V"),
                "seconds": round(time.time() - started, 1),
            }
        )
        oof_proba.append(result.val_proba)
        oof_y.append(result.val_y)

        print(
            f"  {arm.name} fold {index}: macro-F1 {scored.macro_f1:.3f} "
            f"S {scored.recall_of('S'):.3f} (epoch {result.best_epoch}, "
            f"{fold_rows[-1]['seconds']}s)",
            flush=True,
        )

    scores = np.array([r["macro_f1"] for r in fold_rows])
    return {
        "arm": arm.name,
        "loss": arm.loss,
        "oversample": arm.oversample,
        "use_rr": arm.use_rr,
        "note": arm.note,
        "folds": fold_rows,
        "mean_macro_f1": float(scores.mean()),
        "std_macro_f1": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
        "mean_s_recall": float(np.mean([r["s_recall"] for r in fold_rows])),
        "median_best_epoch": int(np.median([r["best_epoch"] for r in fold_rows])),
        "oof_proba": np.concatenate(oof_proba),
        "oof_y": np.concatenate(oof_y),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.PHASE_1D_SEED)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="run cross-validation only; do not touch DS2",
    )
    args = parser.parse_args(argv)

    beats_dir = Path(args.beats_dir).expanduser()
    if not beats_dir.is_dir():
        print(
            f"no beats at {beats_dir}\n"
            f"Run: ./venv/bin/python scripts/preprocess_mitbih.py "
            f"--strategy ds1_ds2 --out-dir ~/physionet-data/processed/mitdb-ds12",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else beats_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment = folds.make_folds(n_folds=args.folds, seed=args.seed)
    assignment.write_json(out_dir / "nn_folds.json")

    cfg_for = lambda arm: train.TrainConfig(     # noqa: E731
        epochs=args.epochs,
        seed=args.seed,
        oversample=arm.oversample,
        use_rr=arm.use_rr,
    )

    # --- cross-validation: every selection decision happens here -------------
    cv: dict[str, dict] = {}
    for arm in ARMS:
        if arm.tune:
            continue                              # handled below, trains nothing
        print(f"[cv] {arm.name}: {arm.note}", flush=True)
        cv[arm.name] = run_cv(beats_dir, assignment, arm, cfg_for(arm))

    # --- arm 3: thresholds on pooled out-of-fold predictions -----------------
    source = max(("weighted_ce", "focal"), key=lambda n: cv[n]["mean_macro_f1"])
    tuned_thresholds = thresholds.tune_thresholds(
        cv[source]["oof_y"], cv[source]["oof_proba"]
    )
    tuned_oof = evaluate.evaluate(
        cv[source]["oof_y"],
        thresholds.apply_thresholds(cv[source]["oof_proba"], tuned_thresholds),
    )
    cv["tuned"] = {
        "arm": "tuned",
        "reuses": source,
        "thresholds": tuned_thresholds.tolist(),
        "oof_macro_f1": tuned_oof.macro_f1,
        "oof_s_recall": tuned_oof.recall_of("S"),
        "median_best_epoch": cv[source]["median_best_epoch"],
        "note": f"per-class thresholds tuned on {source}'s pooled OOF predictions",
    }
    print(
        f"[cv] tuned: reuses {source}, OOF macro-F1 {tuned_oof.macro_f1:.3f}, "
        f"S {tuned_oof.recall_of('S'):.3f}, thresholds "
        f"{np.round(tuned_thresholds, 3).tolist()}",
        flush=True,
    )

    payload = {
        "phase": "1d",
        "seed": args.seed,
        "folds": assignment.to_dict(),
        "cv": {
            name: {k: v for k, v in row.items() if not k.startswith("oof_proba")}
            for name, row in cv.items()
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
        },
        "command": " ".join(sys.argv),
    }

    def _clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(type(obj))

    (out_dir / "nn_cv.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_clean) + "\n"
    )
    print(f"[cv] wrote {out_dir / 'nn_cv.json'}", flush=True)

    if args.skip_test:
        print("[test] skipped by --skip-test; DS2 untouched", flush=True)
        return 0

    # --- final fits on all of DS1, then ONE pass over DS2 --------------------
    print("[final] fitting on all of DS1, then scoring DS2 once", flush=True)
    ds1 = dataset.load_records(beats_dir, splits.DS1_RECORDS)
    ds2 = dataset.load_records(beats_dir, splits.DS2_RECORDS)

    evaluations = []
    for arm in ARMS:
        train_arm = arm
        if arm.tune:
            # Reuses the source arm's configuration; only the decision rule differs.
            train_arm = next(a for a in ARMS if a.name == cv["tuned"]["reuses"])

        epochs = cv[train_arm.name]["median_best_epoch"]
        state = train.fit_final(ds1, train_arm.loss, cfg_for(train_arm), epochs=epochs)

        net = model_module.HybridBeatNet(use_rr=train_arm.use_rr)
        net.load_state_dict(state)
        proba = train.predict_proba(net, ds2)

        if arm.tune:
            y_pred = thresholds.apply_thresholds(
                proba, np.asarray(cv["tuned"]["thresholds"])
            )
        else:
            y_pred = proba.argmax(axis=1)

        scored = evaluate.evaluate(ds2.y, y_pred, label=arm.name)
        evaluations.append(scored)
        print(evaluate.format_report(scored), flush=True)
        print("", flush=True)

    evaluate.write_json(
        evaluations,
        out_dir / "nn_metrics.json",
        extra={
            **payload,
            "test_records": list(splits.DS2_RECORDS),
            "n_test_beats": int(ds2.n_kept),
            "final_fit_note": (
                "Final models were fitted on all of DS1 for a fixed number of "
                "epochs -- the median stopping epoch across that arm's folds -- "
                "because no held-out set remains to early-stop against."
            ),
        },
    )
    print(f"[test] wrote {out_dir / 'nn_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_train_nn_cli.py -v && ./venv/bin/python -m pytest -q`
Expected: PASS, both.

- [ ] **Step 5: Smoke-test the CLI on real data, cross-validation only**

```bash
./venv/bin/python scripts/train_nn.py \
  --beats-dir ~/physionet-data/processed/mitdb-ds12/beats \
  --epochs 2 --folds 2 --skip-test
```

Expected: completes, prints per-fold lines, writes `nn_cv.json`, and does **not** touch DS2. This is a plumbing check on 2 epochs — the numbers are meaningless and must not be reported anywhere.

- [ ] **Step 6: Stage**

```bash
git add scripts/train_nn.py tests/test_train_nn_cli.py
# git commit -m "feat(scripts): Phase 1d study orchestration"
```

---

### Task 10: Run the study and write the results

**Files:**
- Create: `results/<today>-nn-ds1_ds2.json` (written by the script), `results/<today>-nn-ds1_ds2.md` (written by hand from it)

**Interfaces:**
- Consumes: `scripts/train_nn.py` (Task 9)
- Produces: the comparison table Task 11 renders

- [ ] **Step 1: Confirm the DS1/DS2 processed data is present**

```bash
ls ~/physionet-data/processed/mitdb-ds12/beats | wc -l
```

Expected: 48. If not, run the preprocessing command from `COMMANDS.md` first.

- [ ] **Step 2: Run the full study in the background**

```bash
./venv/bin/python scripts/train_nn.py \
  --beats-dir ~/physionet-data/processed/mitdb-ds12/beats \
  --out-dir   ~/physionet-data/processed/mitdb-ds12 \
  2>&1 | tee /tmp/phase1d-run.log
```

Expected: 25 fold fits + 5 final fits, roughly 1–2 hours. Run it in the background and do not block on it.

- [ ] **Step 3: Copy the artifacts into the repo**

```bash
cp ~/physionet-data/processed/mitdb-ds12/nn_metrics.json \
   "results/$(date +%Y-%m-%d)-nn-ds1_ds2.json"
```

- [ ] **Step 4: Write the results markdown**

Create `results/<today>-nn-ds1_ds2.md` following the structure of `results/2026-08-27-baseline-ds1_ds2.md`. It must contain:

1. **The full table** — all seven rows (the Phase 1b Random Forest plus the six arms), ranked by **`macro_f1_headline`** (N/S/V/F, Q excluded), with all-class macro-F1, macro recall, S recall, V recall and n_test alongside.
2. **Before vs after** — `plain_ce` against each treated arm, stating which single change each row represents.
3. **The architecture result** — `waveform_only` against its hybrid twin. If the hybrid wins, the timing argument held. **If it does not, say so plainly and retract the claim in §4 of the spec** rather than burying it.
4. **Fold variance** — the standard deviation of macro-F1 across folds for each arm. If that spread is wider than the gap between two arms, state that the comparison cannot separate them rather than ranking them anyway.
5. **Q** — reported separately with the 8-training-beat explanation, excluded from the headline ranking.
6. **The final-fit caveat** — final models used a fixed epoch budget (the median fold stopping epoch), not early stopping, because no held-out set remained.
7. **Provenance** — commit, command, package versions, timestamps, and the fold assignment, copied from the JSON.

- [ ] **Step 5: Check the claims against the numbers**

Re-read the markdown against the JSON. Every number in prose must appear in the JSON. Any claim of improvement must name the comparison it is against. Do not describe an arm as "better" when the fold spread does not support it.

- [ ] **Step 6: Stage**

```bash
git add results/
# git commit -m "results: Phase 1d hybrid NN and imbalance study"
```

---

### Task 11: Publish and record

**Files:**
- Modify: `site/body.html`, `site/script.html`, `scripts/build_report.py`, `docs/index.html` (generated)
- Modify: `PlanDoc/IMPLEMENTATION.md`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `results/<today>-nn-ds1_ds2.json` (Task 10)
- Produces: the published page and the phase record

- [ ] **Step 1: Read the existing report builder**

```bash
./venv/bin/python -c "print(open('scripts/build_report.py').read())" | head -80
```

Note two rules it already enforces, both from Problem #11 and #12: output must be pure ASCII with escaping done inside `assemble`, and SVG text colour overrides must use inline `style`, never a `fill` attribute, because CSS `fill` wins over presentation attributes.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_report.py`:

```python
def test_report_includes_the_phase_1d_comparison(tmp_path):
    """The published page must show the NN study, not just the baseline."""
    html = build_report.assemble(**build_report.fixture_context())
    assert "plain_ce" in html
    assert "waveform_only" in html
    assert "macro-F1" in html or "macro F1" in html
```

Adapt `fixture_context` to whatever the existing tests in that file already use to build a context without real data — read the file first rather than assuming the helper's name.

- [ ] **Step 3: Run it to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_report.py -k phase_1d -v`
Expected: FAIL.

- [ ] **Step 4: Add the charts**

Add to the generated page:
- A grouped bar chart of macro-F1 per arm, with the Random Forest as a labelled reference line
- A before/after chart of S recall: `plain_ce` against every treated arm
- The confusion matrix of the winning arm, next to the Random Forest's for comparison
- An error-bar chart of per-arm fold spread, so the variance caveat is visible rather than only stated

Follow the existing chart code exactly. Inline `style` for any text colour override; choose text colour from the cell's relative luminance, not from the ramp index (Problem #14).

- [ ] **Step 5: Rebuild and verify**

```bash
./venv/bin/python scripts/build_report.py
./venv/bin/python -m pytest -q
```

Expected: the suite passes including `test_output_is_pure_ascii`. Open `docs/index.html` in a browser and confirm no chart is clipped — measure with `getBBox()` rather than judging by eye (Problem #13), and check both light and dark mode (Problem #14).

- [ ] **Step 6: Write the Phase 1d section of IMPLEMENTATION.md**

Append a `# Phase 1d` section matching the existing structure: what was built, test results (verbatim, observed), results on DS2, **Problems Discovered**, and **Deviations**. Record at minimum:

- The corrected class counts and the `processed/mitdb` vs `processed/mitdb-ds12` confusion that produced the wrong ones (spec §2)
- The abandonment of the single 4-record validation split, with the 81-S-beat measurement that forced it
- The fixed-epoch final fit standing in for early stopping
- Whether the RR branch claim survived arm 5
- The observed plain-vs-weighted S recalls from Task 7, as evidence the regression test discriminates

- [ ] **Step 7: Stage**

```bash
git add site/ scripts/build_report.py docs/index.html tests/test_report.py PlanDoc/IMPLEMENTATION.md
# git commit -m "docs: publish Phase 1d results and record the phase"
```

---

## Verification

Before claiming the phase complete, run and paste the output of:

```bash
./venv/bin/python -m pytest -q
./venv/bin/python -m pytest -q -m requires_data
```

Expected: the offline suite passes with roughly 181 + ~60 new tests; the real-data suite still passes its 9.

Do not report a number that was not observed. Do not describe an arm as an improvement without naming the comparison and the fold spread it clears.
