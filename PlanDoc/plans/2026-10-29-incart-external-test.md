# INCART External Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the unchanged MIT-BIH `D_all_four` ensemble once on the St Petersburg INCART database (lead II, resampled 257 → 360 Hz). Compare the result with DS2, alongside a resampling check and a beat-position check.

**Architecture:**
- **New module:** `src/incart/`, which imports `mit_bih` and `common`; `mit_bih` never imports it.
- **What goes where:** `resample.py` moves signals and annotation positions between rates. `io.py` loads labelled records, downloads them and reads the patient grouping. `evaluate.py` prepares beats, runs the position check and computes metrics, the patient bootstrap and the decision rules.
- **Scripts:** three thin scripts in `scripts/incart/` wire these together.
- **Outputs:** per-beat outputs stay in `~/physionet-data/processed/incartdb/`. Aggregate JSON is copied into `results/` by hand, following `results/README.md`.

**Tech Stack:**
- Python 3.13 in `venv/`.
- Libraries: numpy, scipy 1.16.2 (`scipy.signal.resample_poly`), wfdb 4.3.0 and torch, all already installed.
- Tests: pytest.

**Spec:** `PlanDoc/specs/2026-10-29-incart-external-test-design.md`

## Global Constraints

- **No new dependencies.** Never `pip install`.
- **Running things:** run tests as `venv/bin/python -m pytest …` from the project root; `pytest.ini` sets `pythonpath = src .`. Scripts use the existing `sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))` line.
- **No commits.** PERMISSIONS.md requires the user's OK for every commit, so each task ends with a **Checkpoint** step instead.
- **Nothing in `src/mit_bih/` changes.** The inference bundle is not re-exported. The existing 793 tests must still pass.
- **Fixed values (from the spec):**
  - INCART is recorded at 257 Hz; the model rate is 360 Hz (`mit_bih.config.FS`).
  - Lead II only, chosen by name.
  - Resample with `resample_poly(x, 360, 257)`; positions move with `round(sample * 360 / 257)`.
  - 95% percentile bootstrap, 1,000 draws, seed `mit_bih.config.PHASE_1D_SEED`.
  - "Resampling matters" means the absolute difference in headline macro-F1 is > 0.01.
  - The position rule fires when the medians differ by > 3 samples at 360 Hz (8.3 ms).
  - The position window is ±50 ms (`inference.PEAK_REFINE_SECONDS`).
  - DS2 published headline macro-F1 is 0.5554 (`inference.PUBLISHED_DS2_MACRO_F1["D_all_four"]`).
- **Scoring rules:**
  - INCART is scored once. Nothing is tuned on it: no thresholds, no member choice, and no preprocessing change chosen after seeing INCART numbers.
  - Predicted class is the argmax, as in `scripts/mit_bih/evaluate_bundle.py`.
  - The headline metric is macro-F1 over N/S/V/F (`mit_bih.config.HEADLINE_CLASSES`).
- **Patient data stays local.**
  - Scripts print aggregate numbers only.
  - No signal values, per-beat rows, header contents or patient-file contents are printed, logged or read into this conversation.
  - Per-beat `.npz` files stay under the data root.
- **Labels:** `mit_bih.labels.map_symbols(..., strict=True)` is used everywhere. An unknown symbol stops the run and is never guessed.
- **Every output carries:**
  - `inference.CALIBRATION_NOTE` (the scores are uncalibrated);
  - the line `Research result on a public dataset; not clinical validation and not a medical device.`

## Review Focus

1. **A record or patient with no beats of some class, or a bootstrap draw with none**, must give F1 0 for that class, exactly as `mit_bih.evaluate` does. It must never raise a RuntimeWarning, because pytest turns RuntimeWarning into an error. Test: Task 5, `test_matrix_f1_matches_evaluate_with_an_absent_class`.
2. **Two marks that refine onto the same peak** in the moved-to-the-peak variant: drop the later one, count it, and keep each label with its own beat. Test: Task 4, `test_refined_marks_that_collide_are_dropped_and_counted`.
3. **A patient file that is missing, fails to download, or contains diagnosis codes that look like record ids** (ICD-10 `I10`, `I25`): fall back to grouping by record, with a written note. It must never crash or produce a wrong grouping. Tests: Task 3, `test_duplicate_ids_fall_back_to_records` and `test_patient_file_download_failure_returns_false`.
4. **Running the INCART evaluation before the resampling check has passed, or a second time**, must be refused with a plain message. This protects "scored once" and the DS2 comparison. Tests: Task 5, `test_missing_resampling_check_is_refused`, `test_failed_reproduction_is_refused` and `test_existing_output_is_not_overwritten`.
5. **A record whose beats are all dropped** must appear in the per-record table with `n_beats: 0` and no metrics, rather than crashing `mit_bih.evaluate.evaluate` on an empty set. Test: Task 5, `test_record_row_with_no_beats`.

---

### Task 1: Package, config and resampling

**Files:**
- Create: `src/incart/__init__.py`, `src/incart/config.py`, `src/incart/resample.py`
- Create: `tests/incart/__init__.py` (empty), `tests/incart/test_resample.py`

**Interfaces:**
- Produces, in `incart.config`:
  - `FS = 257`, `MODEL_FS = mit_bih.config.FS`, `LEAD = "II"`
  - `DATABASE = "incartdb"`, `DIRNAME = "incartdb"`
  - `RECORDS = tuple(f"I{i:02d}" for i in range(1, 76))`
  - `PATIENT_FILE = "files-patients-diagnoses.txt"`, `N_PATIENTS = 32`
  - `PACED_SYMBOLS = ("/", "f")`
  - `N_BOOT = 1000`, `ALPHA = 0.05`, `SEED = mit_bih.config.PHASE_1D_SEED`
  - `RESAMPLING_THRESHOLD = 0.01`, `POSITION_THRESHOLD_SAMPLES = 3`
  - `NOT_CLINICAL = "Research result on a public dataset; not clinical validation and not a medical device."`
  - `raw_dir() -> Path` (= `common.paths.data_root() / DIRNAME`) and `processed_dir() -> Path` (= `common.paths.processed_root() / DIRNAME`)
- Produces, in `incart.resample`:
  - `MovedPositions(samples: np.ndarray[int64], keep: np.ndarray[bool])`, a frozen dataclass, with property `n_dropped -> int`
  - `resample_signal(signal, fs_in: int, fs_out: int) -> np.ndarray`
  - `move_positions(samples, fs_in: int, fs_out: int, n_out: int) -> MovedPositions`
  - `resample_record(signal, samples, fs_in: int, fs_out: int) -> tuple[np.ndarray, MovedPositions]`
  - `round_trip(signal, samples, fs: int, via_fs: int) -> tuple[np.ndarray, MovedPositions]`. Its `keep` refers to the original positions.

- [ ] **Step 1: Write the failing tests** in `tests/incart/test_resample.py`. Build signals with `from tests.mit_bih.synthetic import make_ecg`.

```python
def test_spike_lands_on_the_moved_position():
    sig, peaks = make_ecg(n_beats=10, fs=257, rr_samples=257)
    out = resample.resample_signal(sig, 257, 360)
    for p in peaks:
        expected = round(p * 360 / 257)
        lo, hi = expected - 10, expected + 11
        assert abs(lo + int(np.argmax(out[lo:hi])) - expected) <= 1
    assert abs(out.size - round(sig.size * 360 / 257)) <= 1

def test_positions_move_and_out_of_range_ones_are_dropped():
    moved = resample.move_positions(np.array([0, 257, 1000]), 257, 360, n_out=1000)
    assert moved.samples.tolist() == [0, 360]
    assert moved.keep.tolist() == [True, True, False]
    assert moved.n_dropped == 1

def test_rounding_is_half_up_not_bankers():
    # 180 * 257 / 360 = 128.5 exactly
    assert resample.move_positions(np.array([180]), 360, 257, n_out=10**6).samples.tolist() == [129]

def test_round_trip_keeps_positions_within_one_sample():
    sig, peaks = make_ecg(n_beats=20, fs=360)
    out, moved = resample.round_trip(sig, peaks, 360, 257)
    assert moved.keep.all()
    assert np.abs(moved.samples - peaks).max() <= 1
    assert abs(out.size - sig.size) <= 2
```

- [ ] **Step 2: Run the tests and see them fail.** Run `venv/bin/python -m pytest tests/incart/test_resample.py -v`. Expected: FAIL with `ModuleNotFoundError: No module named 'incart'`.

- [ ] **Step 3: Implement `config.py` and `resample.py`.**
  - `resample_signal`: reduce `fs_out`/`fs_in` by their gcd and call `scipy.signal.resample_poly(np.asarray(signal, float64), up, down)` with the default window and padding, as the spec says.
  - `move_positions`: round half up in integer arithmetic, `(2 * s * fs_out + fs_in) // (2 * fs_in)`, so the result never depends on float or banker's rounding. Keep positions in `[0, n_out)`.
  - `round_trip`: call `resample_record` twice, then compose the two `keep` masks so that `keep` indexes the original positions.
  - `__init__.py`: a docstring stating the purpose, the one-way dependency (`incart` → `mit_bih`, `common`) and "not clinical validation".

- [ ] **Step 4: Run the tests and see them pass.** Same command. Expected: 4 passed.

- [ ] **Step 5: Checkpoint.** No commit. Run `git status --short src/incart tests/incart`. Only the new files should be listed.

### Task 2: Loading a labelled record

**Files:**
- Create: `src/incart/io.py` (the loading part)
- Create: `tests/incart/conftest.py`, `tests/incart/test_io.py`

**Interfaces:**
- Consumes:
  - `mit_bih.io.load_record(record_dir, name, expected_fs, preference)` and `mit_bih.io.load_annotations(record_dir, name)`
  - `mit_bih.labels.map_symbols(symbols, strict=True)`
- Produces, in `incart.io`:
  - `LabelledRecord`, a frozen dataclass with fields:
    - `name: str`, `signal: np.ndarray`, `fs: int`, `lead: str`
    - `beat_samples: np.ndarray[int64]`, `beat_labels: np.ndarray[int64]` (indices into `AAMI_CLASSES`)
    - `symbol_counts: dict[str, int]` (every annotation symbol), `n_excluded: int`
  - `LabelledRecord` properties: `n_beats -> int` and `has_paced -> bool`. `has_paced` is true if any `config.PACED_SYMBOLS` symbol has count > 0.
  - `label_beats(samples, symbols) -> tuple[np.ndarray, np.ndarray, int]`, returning `(beat_samples, beat_labels, n_excluded)`. It raises `mit_bih.errors.UnknownSymbolError`.
  - `load_labelled_record(record_dir, name, fs: int = config.FS, preference: tuple[str, ...] = (config.LEAD,)) -> LabelledRecord`. DS2 calls it with `fs=360, preference=mit_bih.config.LEAD_PREFERENCE`.
- Produces, in `tests/incart/conftest.py`:
  - The fixture `incart_record(tmp_path) -> dict` with keys `dir`, `name` (`"I99"`), `lead_ii`, `n_beats` (20), `labels` (np.int64 array), `n_excluded` (1).
  - The fixture `bundle -> InferenceBundle`, built like `_bundle()` in `tests/mit_bih/test_inference.py`, with `members=()`.
  - The constant `LEADS_12 = ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")`.
  - The re-exported fixture `from tests.mit_bih.conftest import no_network`.

- [ ] **Step 1: Write the fixture and the failing tests.**
  - **The `incart_record` fixture:**
    - Call `make_ecg(n_beats=20, fs=257, rr_samples=257)`.
    - Write 12 leads with `write_record(..., fs=257)`. Lead II is the signal itself; every other lead is the signal scaled by a different factor.
    - Annotations: a `+` 50 samples before the first beat, then beats cycling `N, N, V`.

```python
def test_lead_ii_is_picked_by_name_from_twelve_leads(incart_record):
    rec = io.load_labelled_record(incart_record["dir"], "I99")
    assert rec.lead == "II" and rec.fs == 257
    assert np.allclose(rec.signal, incart_record["lead_ii"], atol=0.01)
    assert rec.n_beats == 20 and rec.n_excluded == 1
    assert np.array_equal(rec.beat_labels, incart_record["labels"])
    assert not rec.has_paced

def test_record_at_another_rate_is_rejected(tmp_path):  # same leads written at fs=360
    with pytest.raises(SamplingRateError):
        io.load_labelled_record(tmp_path / "x", "I98")

def test_unknown_symbol_stops_the_run():
    with pytest.raises(UnknownSymbolError):
        io.label_beats(np.array([10, 20]), ("N", "Z"))

def test_paced_symbol_marks_the_record():  # "/" among the beats
    s, l, n_ex = io.label_beats(np.array([10, 20, 30]), ("N", "/", "+"))
    assert s.tolist() == [10, 20] and n_ex == 1
    # and via load_labelled_record on a written record containing "/": rec.has_paced is True
```

- [ ] **Step 2: Run the tests and see them fail.** Run `venv/bin/python -m pytest tests/incart/test_io.py -v`. Expected: FAIL, `incart.io` has no attribute `load_labelled_record`.

- [ ] **Step 3: Implement the loading part of `io.py`.** Reuse the `mit_bih.io` readers. Count `symbol_counts` over all annotations with `collections.Counter`. Do not write a second WFDB reader.

- [ ] **Step 4: Run the tests and see them pass.** Expected: 4 passed.

- [ ] **Step 5: Checkpoint.** No commit.

### Task 3: Download, check and patient grouping

**Files:**
- Modify: `src/incart/io.py` (add acquisition and the patient map)
- Create: `scripts/incart/download_incart.py`
- Create: `tests/incart/test_acquire.py`
- Modify: `tests/test_scripts.py`, adding `"scripts/incart/download_incart.py"` to `SCRIPTS`

**Interfaces:**
- Consumes: `mit_bih.acquire.check_record(directory, name, expected_fs)`, `mit_bih.acquire.RecordCheck` and `mit_bih.acquire.directory_size_bytes`.
- Produces, in `incart.io`:
  - `check_record(directory, name) -> RecordCheck`. It runs `acquire.check_record(..., expected_fs=config.FS)`, then reads the header with `wfdb.rdheader` and requires lead `II`, with `units == "mV"` for that lead.
  - `verify(directory, records=config.RECORDS) -> list[RecordCheck]` and `missing_records(directory, records=config.RECORDS) -> list[str]`. Neither touches the network.
  - `download(directory=None, records=config.RECORDS, force=False) -> list[RecordCheck]`:
    - calls `wfdb.dl_database(config.DATABASE, dl_dir=..., records=todo)` for missing records only;
    - raises `RuntimeError` if anything fails verification afterwards.
  - `fetch_patient_file(directory) -> bool`:
    - if the file is already present, returns True without any network call;
    - otherwise calls `wfdb.dl_files(config.DATABASE, str(directory), [config.PATIENT_FILE])`;
    - any exception gives False.
  - `PatientMap(patient_of: dict[str, str], by_patient: bool, note: str)`, with property `groups -> dict[str, list[str]]`.
  - `patient_groups_from_text(text) -> list[tuple[str, ...]]`: record ids matching `\bI\d{2}\b`, one tuple per line that contains any; lines without ids are skipped.
  - `build_patient_map(text: str | None, records=config.RECORDS, expected_patients=config.N_PATIENTS) -> PatientMap`:
    - Grouping by patient is used only if every record appears exactly once and there are `expected_patients` groups.
    - Patient ids are `P01, P02, …` in order of first appearance.
    - Otherwise each record is its own group, and the note says: `Resampled by record: no valid record-to-patient mapping (<reason>). This understates the uncertainty, since some records share a patient.`
  - `load_patient_map(directory, records=config.RECORDS) -> PatientMap`, which reads `directory / config.PATIENT_FILE` if present.

- [ ] **Step 1: Write the failing tests.**

```python
def test_complete_directory_makes_no_network_call(incart_record, monkeypatch, no_network):
    monkeypatch.setattr(io.wfdb, "dl_database", lambda *a, **k: pytest.fail("fetched"))
    assert io.download(incart_record["dir"], records=["I99"])[0].ok

def test_only_missing_records_are_fetched(tmp_path, monkeypatch):
    # fake dl_database records its `records` argument and writes a valid I98 into dl_dir
    io.download(tmp_path, records=["I98"])
    assert calls == [["I98"]]

def test_check_fails_without_lead_ii(tmp_path):        # reason contains "lead II"
def test_check_fails_at_another_rate(tmp_path):        # reason contains "257"
def test_patient_file_already_present_needs_no_network(tmp_path, monkeypatch, no_network):
def test_patient_file_download_failure_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(io.wfdb, "dl_files", raise_oserror)
    assert io.fetch_patient_file(tmp_path) is False

def test_one_line_per_patient_groups_records():
    pm = io.build_patient_map("Patient 1: I01, I02\n\nPatient 2: I03\n", ("I01","I02","I03"), 2)
    assert pm.by_patient and pm.patient_of == {"I01": "P01", "I02": "P01", "I03": "P02"}

def test_duplicate_ids_fall_back_to_records():  # e.g. an ICD-10 code "I02" in a diagnosis
    pm = io.build_patient_map("I01 I02\nI02 I03\n", ("I01","I02","I03"), 2)
    assert not pm.by_patient and pm.patient_of == {"I01": "I01", "I02": "I02", "I03": "I03"}
    assert "understates" in pm.note

def test_wrong_patient_count_falls_back(): ...
def test_no_file_falls_back(): assert not io.build_patient_map(None, ("I01",), 1).by_patient
```

- [ ] **Step 2: Run the tests and see them fail.** Run `venv/bin/python -m pytest tests/incart/test_acquire.py -v`. Expected: FAIL with AttributeError.

- [ ] **Step 3: Implement the acquisition functions and the patient map in `io.py`.**

- [ ] **Step 4: Write `scripts/incart/download_incart.py`.**
  - Arguments: `--dir` (default `config.raw_dir()`), `--verify` (no network) and `--force`. Use the same flow and output style as `scripts/mit_bih/download_mitbih.py`.
  - After downloading, call `fetch_patient_file` and print `Patient file: present` or `Patient file: not available (results will be grouped by record)`.
  - The docstring states: open access, ODC Attribution License v1.0, about 800 MB, and the source PhysioNet `incartdb` v1.0.0.

- [ ] **Step 5: Add the script to `SCRIPTS` in `tests/test_scripts.py`, then run the tests and see them pass.** Run `venv/bin/python -m pytest tests/incart tests/test_scripts.py -v`. Expected: all pass, including `test_help_exits_cleanly[scripts/incart/download_incart.py]`.

- [ ] **Step 6: Checkpoint.** No commit.

### Task 4: Beats at the model rate and the position check

**Files:**
- Create: `src/incart/evaluate.py` (the beats and positions part)
- Create: `tests/incart/test_beats.py`

**Interfaces:**
- Consumes:
  - `incart.resample.resample_record` and `incart.resample.round_trip`
  - `mit_bih.inference.prepare_beats(signal, fs, r_peaks, bundle) -> PreparedBeats`
  - `mit_bih.filters.denoise`
- Produces, in `incart.evaluate`:
  - `RecordBeats`, a frozen dataclass with fields:
    - `name: str`
    - `X: np.ndarray`, `rr: np.ndarray`, `y: np.ndarray[int64]`, `r_peaks: np.ndarray[int64]`
    - `n_annotated: int`, `n_out_of_range: int`, `n_collided: int`
    - `n_dropped_boundary: int`, `n_dropped_edge: int`
  - `RecordBeats` property `n_beats` and methods:
    - `reconciles() -> bool`: `n_beats + n_out_of_range + n_collided + n_dropped_boundary + n_dropped_edge == n_annotated`
    - `counts() -> dict` of those six numbers
  - `band_passed(signal, bundle) -> np.ndarray`: `filters.denoise` with the bundle's band-pass settings and `use_notch=bundle.notch`.
  - `position_offsets(filtered, marks, fs: int, window_s: float = inference.PEAK_REFINE_SECONDS) -> np.ndarray[int64]`: the signed sample offset from each mark to the argmax of `|filtered|` within ±`round(window_s*fs)`, with the window clipped at the edges as in `detect_r_peaks`.
  - `peak_amplitudes(filtered, marks, fs: int, window_s=...) -> np.ndarray`: the largest `|filtered|` in the same window, in mV.
  - `refined_marks(filtered, marks, labels, fs: int) -> tuple[np.ndarray, np.ndarray, int]`: the marks moved to the local maximum. When two marks refine onto the same sample, the first is kept and the later one dropped; the int is the dropped count.
  - `at_model_rate(rec: LabelledRecord, bundle, round_trip_via: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]`:
    - returns `(signal, marks, labels, n_out_of_range)`, all at `bundle.fs`;
    - resamples when `rec.fs != bundle.fs`;
    - `round_trip_via` is only valid when `rec.fs == bundle.fs`, and raises ValueError otherwise.
  - `record_beats(rec, bundle, *, round_trip_via: int | None = None, refine: bool = False) -> RecordBeats`. Labels are attached by peak as in `evaluate_bundle.raw_beats`, and `n_annotated = rec.n_beats`.
  - `record_positions(rec, bundle) -> tuple[np.ndarray, np.ndarray]`: offsets in samples at `bundle.fs`, and peak amplitudes in mV.
  - `summarise_positions(offsets, peaks_mv, fs: int) -> dict`, with keys `n`, `median_samples`, `median_ms`, `q1_ms`, `q3_ms`, `median_peak_mv`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_marks_on_the_peaks_give_zero_offset(bundle):
    sig, peaks = make_ecg(n_beats=20, fs=360)
    assert np.all(ev.position_offsets(ev.band_passed(sig, bundle), peaks, 360) == 0)

def test_marks_off_the_peaks_give_the_shift(bundle):
    sig, peaks = make_ecg(n_beats=20, fs=360)
    f = ev.band_passed(sig, bundle)
    off = ev.position_offsets(f, peaks - 5, 360)
    assert np.all(off == 5)
    s = ev.summarise_positions(off, ev.peak_amplitudes(f, peaks - 5, 360), 360)
    assert s["median_ms"] == pytest.approx(5 / 360 * 1000) and s["n"] == 20

def test_refined_marks_that_collide_are_dropped_and_counted(bundle):
    sig, peaks = make_ecg(n_beats=10, fs=360)
    marks = np.sort(np.append(peaks, peaks[3] + 4))
    labels = np.arange(marks.size, dtype=np.int64) % 4
    new, new_labels, n = ev.refined_marks(ev.band_passed(sig, bundle), marks, labels, 360)
    assert n == 1 and np.array_equal(new, peaks)
    assert new_labels.tolist() == np.delete(labels, 4).tolist()

def test_beat_counts_reconcile_from_file_to_classified_beats(incart_record, bundle):
    rec = io.load_labelled_record(incart_record["dir"], "I99")
    beats = ev.record_beats(rec, bundle)
    assert beats.reconciles() and beats.n_annotated == 20 and beats.n_dropped_edge == 2
    assert beats.X.shape == (18, 300)
    assert np.array_equal(beats.y, incart_record["labels"][1:-1])

def test_round_trip_beats_reconcile(tmp_path, bundle):  # a 360 Hz MLII record from write_record
    beats = ev.record_beats(rec360, bundle, round_trip_via=257)
    assert beats.reconciles() and beats.n_beats == rec360.n_beats - 2

def test_round_trip_on_a_257_hz_record_is_refused(incart_record, bundle):
    with pytest.raises(ValueError): ev.at_model_rate(rec257, bundle, round_trip_via=257)
```

- [ ] **Step 2: Run the tests and see them fail.** Run `venv/bin/python -m pytest tests/incart/test_beats.py -v`. Expected: FAIL with `ModuleNotFoundError: No module named 'incart.evaluate'`.

- [ ] **Step 3: Implement the beats and positions part of `evaluate.py`.**
  - Compute positions and refinement on the signal at `bundle.fs`, after any resampling.
  - The position check uses as-marked positions only: `record_positions` never refines.

- [ ] **Step 4: Run the tests and see them pass.** Expected: 6 passed.

- [ ] **Step 5: Checkpoint.** No commit.

### Task 5: Metrics, patient bootstrap, decision rules and guards

**Files:**
- Modify: `src/incart/evaluate.py` (the metrics part)
- Create: `tests/incart/test_metrics.py`

**Interfaces:**
- Consumes: `mit_bih.evaluate.evaluate`, and `mit_bih.stats.metric_ci` (in the test only, for comparison).
- Produces, in `incart.evaluate`:
  - `confusion(y, pred, n_classes: int = 5) -> np.ndarray[int64]`: rows are true, columns are predicted, built with `np.bincount`. Empty inputs give zeros.
  - `headline_f1_from_matrix(matrix) -> float`: the mean F1 over the `HEADLINE_CLASSES` indices. An undefined precision, recall or F1 counts as 0, as in `mit_bih.evaluate`. Use `np.divide(..., where=...)` so that no RuntimeWarning is raised. Vectorised over a leading batch axis.
  - `sum_by_group(matrices: dict[str, np.ndarray], group_of: dict[str, str]) -> dict[str, np.ndarray]`.
  - `grouped_bootstrap_ci(matrices: Mapping[str, np.ndarray], n_boot: int = config.N_BOOT, seed: int = config.SEED, alpha: float = config.ALPHA) -> tuple[float, float]`:
    - `rng = np.random.default_rng(seed)`; each draw picks `len(matrices)` groups with replacement and sums their matrices;
    - takes the percentile interval of `headline_f1_from_matrix`.
  - `holds_up(reference: float, ci: tuple[float, float]) -> bool`, inclusive at both ends.
  - `resampling_matters(original: float, round_trip: float, threshold: float = config.RESAMPLING_THRESHOLD) -> bool`, true when `abs(diff) > threshold`.
  - `position_rule_fires(incart_median_samples: float, ds2_median_samples: float, threshold: int = config.POSITION_THRESHOLD_SAMPLES) -> bool`, true when `abs(diff) > threshold`.
  - `record_row(name: str, y, pred) -> dict`: `{"record", "n_beats", "headline_macro_f1", "metrics"}` with `metrics = evaluate(...).to_dict()`. With no beats it gives `n_beats: 0`, `headline_macro_f1: None` and `metrics: None`.
  - `class PreconditionError(mit_bih.errors.EcgError)`.
  - `read_resampling_verdict(path) -> dict`. It raises `PreconditionError` if the file is absent (the message names `scripts/incart/resampling_check.py`) or if `payload["reproduces_published"] is not True`.
  - `refuse_to_overwrite(path, overwrite: bool) -> None`. It raises `PreconditionError` (the message names `--overwrite` and says INCART is scored once) if the path exists and `overwrite` is false.
- **The `resampling_check.json` schema**, which Task 6 writes and Task 7 reads. These keys are fixed:
  - `reproduces_published: bool | None`
  - `original` and `round_trip`, each `{"headline_macro_f1": float, "ci95": [lo, hi], "metrics": dict, "counts": dict}`
  - `difference: float` (round trip minus original), `threshold: 0.01`, `resampling_matters: bool`, `records: list[str]`

- [ ] **Step 1: Write the failing tests.**

```python
def test_matrix_f1_matches_evaluate_with_an_absent_class():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500); p = rng.integers(0, 3, 500)  # no F, no Q anywhere
    expected = evaluate(y, p).macro_f1_of(HEADLINE_CLASSES)
    assert ev.headline_f1_from_matrix(ev.confusion(y, p)) == pytest.approx(expected)

def test_every_patient_perfect_gives_a_point_interval():
    m = {f"P{i}": ev.confusion(np.arange(4).repeat(10), np.arange(4).repeat(10)) for i in range(5)}
    assert ev.grouped_bootstrap_ci(m, n_boot=200, seed=1) == (1.0, 1.0)

def test_one_all_wrong_patient_widens_the_interval_beyond_beat_resampling():
    # 10 perfect patients x 200 beats (classes 0-3) + 1 patient with pred = (y + 1) % 4
    lo, hi = ev.grouped_bootstrap_ci(m, n_boot=1000, seed=1)
    blo, bhi = stats.metric_ci(y_all, p_all, n_boot=200, seed=1)
    assert lo < 1.0 and (hi - lo) > 2 * (bhi - blo)

def test_bootstrap_is_reproducible_with_the_seed(): ...
def test_decision_rules():
    assert ev.holds_up(0.5554, (0.50, 0.60)) and not ev.holds_up(0.5554, (0.56, 0.60))
    assert not ev.resampling_matters(0.5554, 0.5604) and ev.resampling_matters(0.5554, 0.5404)
    assert not ev.position_rule_fires(0, 2.5) and ev.position_rule_fires(-1, 3)

def test_record_row_with_no_beats():
    assert ev.record_row("I01", [], []) == {"record": "I01", "n_beats": 0,
                                            "headline_macro_f1": None, "metrics": None}

def test_missing_resampling_check_is_refused(tmp_path): ...    # match="resampling_check.py"
def test_failed_reproduction_is_refused(tmp_path): ...         # reproduces_published False
def test_existing_output_is_not_overwritten(tmp_path): ...     # match="--overwrite"; overwrite=True passes
```

- [ ] **Step 2: Run the tests and see them fail.** Run `venv/bin/python -m pytest tests/incart/test_metrics.py -v`. Expected: FAIL with AttributeError.

- [ ] **Step 3: Implement the metrics part of `evaluate.py`.**

- [ ] **Step 4: Run the tests and see them pass.** Expected: 9 passed.

- [ ] **Step 5: Checkpoint.** No commit.

### Task 6: The resampling check script

**Files:**
- Create: `scripts/incart/resampling_check.py`
- Modify: `tests/test_scripts.py`, adding `"scripts/incart/resampling_check.py"`

**Interfaces:**
- Consumes:
  - from Task 2: `io.load_labelled_record(..., fs=360, preference=mit_bih.config.LEAD_PREFERENCE)`;
  - from Task 4: `record_beats` with `round_trip_via=incart.config.FS`;
  - from Task 5: `confusion`, `headline_f1_from_matrix`, `grouped_bootstrap_ci`, `resampling_matters` and the schema;
  - `mit_bih.inference.EcgClassifier.from_bundle` and `mit_bih.splits.DS2_RECORDS`.
- Produces: `<out-dir>/resampling_check.json` (default `incart.config.processed_dir()`). The script exits 1 if the full-DS2 original does not reproduce 0.5554.

- [ ] **Step 1: Add the script to `SCRIPTS` and run `venv/bin/python -m pytest tests/test_scripts.py -v`.** Expected: FAIL for `resampling_check.py` (the file is missing).

- [ ] **Step 2: Write the script.**
  - **Arguments:** `--bundle` (default `inference.DEFAULT_BUNDLE_PATH`), `--mitdb-dir` (default `mit_bih.config.mitdb_dir()`), `--out-dir` and `--records`.
  - **Records:** the default is DS2 without `PACED_RECORDS` (22 records).
  - **Per record:** score the original and the round trip with `clf.score_beats(...).argmax(axis=1)`; keep per-record confusion matrices; check `reconciles()` and stop with an error if it is false.
  - **Intervals:** one group per record (DS2: 22 records = 22 patients).
  - **Pass condition:** `reproduces_published = round(original_headline, 4) == 0.5554`, set only on the full 22-record run. A subset run writes `resampling_check_subset.json` with `reproduces_published: None`.
  - **Output:** print the two headlines, their CIs, the difference, the verdict and `CALIBRATION_NOTE`. Write the provenance block with `common.provenance.provenance({"bundle": args.bundle})`.

- [ ] **Step 3: Run the tests and see them pass.** Run `venv/bin/python -m pytest tests/test_scripts.py -v`. Expected: all pass.

- [ ] **Step 4: Checkpoint.** No commit.

### Task 7: The INCART evaluation script

**Files:**
- Create: `scripts/incart/evaluate_on_incart.py`
- Modify: `tests/test_scripts.py`, adding `"scripts/incart/evaluate_on_incart.py"`

**Interfaces:**
- Consumes: everything from Tasks 2–5, plus `EcgClassifier.from_bundle`.
- Produces:
  - `<out-dir>/external_test.json`
  - `<out-dir>/predictions/<variant>/<record>.npz`, holding `y`, `pred`, `scores` (float32) and `r_peaks`. `<variant>` is `as_marked` or `refined`.

- [ ] **Step 1: Add the script to `SCRIPTS` and run the test.** Expected: FAIL, the file is missing.

- [ ] **Step 2: Write the script, in this order.**
  1. **Arguments:** `--bundle`, `--dir` (default `config.raw_dir()`), `--mitdb-dir`, `--out-dir`, `--smoke N` and `--overwrite`.
  2. **Smoke mode** (`--smoke N`): the first N INCART records only. Prints beat counts, the reconcile result and seconds per record, and nothing else: no metrics, no predictions, no files written. A smoke run must not show INCART performance.
  3. **Guards:** call `refuse_to_overwrite(out/"external_test.json", args.overwrite)`, then `verdict = read_resampling_verdict(out/"resampling_check.json")`. Smoke mode skips both.
  4. **First pass (no model):**
     - Patient map from `load_patient_map(args.dir)`.
     - For each INCART record, `load_labelled_record`. Leave out and list records with `has_paced`. Sum `symbol_counts`, and collect `record_positions` and `record_beats(rec, bundle)`.
     - For each of the 22 DS2 records, collect `record_positions` at 360 Hz.
  5. **Position rule:** `summarise_positions` for INCART and DS2, then `position_rule_fires(incart["median_samples"], ds2["median_samples"])`.
  6. **Scoring:** score as-marked per record and save the `.npz`. If the rule fired, prepare `record_beats(rec, bundle, refine=True)` and score it as the second variant. If any record's `reconciles()` is false, stop with an error that names the record.
  7. **Per variant:** the pooled `mit_bih.evaluate.evaluate(...).to_dict()`, the headline, the `grouped_bootstrap_ci` over `sum_by_group(record matrices, patient_map.patient_of)`, and a `record_row` for every record.
  8. **Decisions:** `holds_up(verdict["original"]["headline_macro_f1"], as_marked_ci)` and `difference_from_ds2`. Copy `resampling_matters` and `difference` from the verdict. Only the as-marked result is the headline.
  9. **Write `external_test.json`:**
     - dataset name and version, `lead`, both rates, ensemble and `n_models`;
     - `records_scored` and `records_excluded_paced`;
     - `patient_grouping` (`by_patient`, `n_groups`, `note`, `groups`);
     - `symbol_counts`, `beat_counts` (summed `counts()` plus `reconciles`), `position_check` and `as_marked`;
     - `refined` (`None` if the rule did not fire) and `ds2_reference`;
     - `holds_up`, `resampling_check`, `calibration`, `not_clinical`, `seconds`;
     - `provenance` (inputs: the bundle and `resampling_check.json`).
  10. **Print:** the aggregates and both verdicts.

- [ ] **Step 3: Run the tests and see them pass.** Run `venv/bin/python -m pytest tests/incart tests/test_scripts.py -v`. Expected: all pass.

- [ ] **Step 4: Run the full suite.** Run `venv/bin/python -m pytest -q`. Expected: 793 + the new tests pass, 0 failures.

- [ ] **Step 5: Checkpoint.** No commit.

### Task 8: The real-data runs (approved by the user on 2026-10-29)

**Files:**
- Create: `tests/incart/test_real_data.py`

- [ ] **Step 1: Download.** Run `venv/bin/python scripts/incart/download_incart.py`. This is the approved network request, about 800 MB. Expected: `75 records verified`, and the patient-file line.

- [ ] **Step 2: Check what is on disk.** Run `venv/bin/python scripts/incart/download_incart.py --verify`. Expected: `75/75 records verify.` and exit 0.

- [ ] **Step 3: Write the `requires_data` test.** Load `I01` with `load_labelled_record`. Assert `fs == 257`, `lead == "II"`, `n_beats > 0`, and that mapping raised nothing. Skip if `config.raw_dir()/"I01.hea"` is absent. Run `venv/bin/python -m pytest tests/incart/test_real_data.py -m requires_data -v`. Expected: 1 passed.
  - **If `UnknownSymbolError` is raised here or later, stop and tell the user the symbol.** The spec forbids guessing it.

- [ ] **Step 4: Resampling check, about 19 minutes.** Run `venv/bin/python scripts/incart/resampling_check.py` in the background. Expected: original `0.5554`, `reproduces_published: true` and exit 0. If it does not reproduce, stop and report; do not run Step 6.

- [ ] **Step 5: Smoke run.** Run `venv/bin/python scripts/incart/evaluate_on_incart.py --smoke 2`. Expected: counts reconcile for both records, and no metrics printed.

- [ ] **Step 6: Full INCART run, about 33 minutes,** plus about 33 more if the position rule fires. Run `venv/bin/python scripts/incart/evaluate_on_incart.py` in the background. Expected: exit 0, and `beat_counts.reconciles` is true. If the patient grouping fell back to records, note it for the write-up and the user.

- [ ] **Step 7: Checkpoint.** No commit. The data files are under `~/physionet-data/` only. Run `git status --short` and confirm that no data file appears.

### Task 9: Results files and documentation

**Files:**
- Create:
  - `results/<run-date>-incart-external-test.json`
  - `results/<run-date>-incart-resampling-check.json`
  - `results/<run-date>-incart-external-test.md`
- Modify:
  - `results/README.md` (contents table)
  - `README.md`: Status row, Layout tree (`src/incart/`, `scripts/incart/`, `tests/incart/`), the "Dependencies run one way" sentence, and a short INCART section
  - `COMMANDS.md`: a new section "4d. External test on INCART", with the three commands and their run times
  - `PlanDoc/IMPLEMENTATION.md`: a new section with what was built, test output, real runs and deviations
  - `PlanDoc/PLAN.md`: step 2 status
  - `PlanDoc/specs/2026-10-29-incart-external-test-design.md`: the Status line

- [ ] **Step 1: Copy the two run JSONs into `results/`.** Name them by run date and add a top-level `"run"` block: date, nearest commit with dirty flag, exact commands, package versions and run time. Confirm that neither file holds per-beat data: `grep -c '"y"' results/<run-date>-incart-*.json` must print 0.

- [ ] **Step 2: Write the analysis `.md`.**
  - **Opening banner:** the not-clinical line and the calibration note.
  - **The two pre-registered rules**, quoted from the spec, each with its verdict.
  - **INCART and DS2 side by side:** per-class recall, precision and F1, the headline, and the 95% CI.
  - **Diagnostics:** the position check (medians and IQR in ms, peak amplitude in mV) and the resampling check.
  - **Patient grouping:** what was used, with its note.
  - **Other contents:** excluded records, symbol counts, and where the per-record table is (in the JSON).
  - **Limitations.**
  - A drop is not explained away without both checks behind it.

- [ ] **Step 3: Update the documents listed under Files.**

- [ ] **Step 4: Run the full suite.** Run `venv/bin/python -m pytest -q`. Expected: all pass.

- [ ] **Step 5: Checkpoint.** No commit. Report to the user: the verdicts, the INCART headline with its CI next to DS2, the files written, and anything that deviated from the spec.

---

## Calls made in this plan (not in the spec)

- **Download code** lives in `src/incart/io.py`, keeping the approved four-file module layout; it is not in a new `acquire.py`.
- **The patient file name** `files-patients-diagnoses.txt` is a best guess and has not been verified. The grouping is used only if it maps all 75 records exactly once into 32 patients, the published count; otherwise the fallback and its note apply.
- **Run order:** the resampling check runs before the INCART evaluation, and the evaluation reads its result. This way both decision rules are applied by code, and the DS2 reference is fixed before INCART is scored.
- **Smoke mode** prints counts only, so testing the pipeline does not reveal INCART performance.
- **The amplitude diagnostic** is the peak `|band-passed|` within ±50 ms of the mark, not the value at the mark itself. The value at the mark would mix a position difference into the scale comparison.
- **Rounding of moved positions** is half up, in integer arithmetic. Python's `round` rounds halves to even, and exact halves occur in the 360 → 257 direction of the round trip.
- **Run-time estimates** are based on the raw DS2 run (9 min 24 s for 49,646 beats): INCART about 33 min, resampling check about 19 min. The spec's 26 and 15 minutes were based on the faster saved-beats run.
