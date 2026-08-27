# IMPLEMENTATION.md — Smart Healthcare

Tracks actual execution against [`PLAN.md`](PLAN.md), under the policy in
[`../PERMISSIONS.md`](../PERMISSIONS.md).

**Overall status: Phase 0 and Phase 1 COMPLETE.**
123 offline tests pass; 9 real-data tests pass; all 48 MIT-BIH records downloaded
and preprocessed.

Last updated: 2026-08-26

---

## Implementation Steps

### Phase 0 — Environment and data

| # | Step | Status | Note |
|---|------|--------|------|
| 0.1 | Scope approval | Completed | Approved 2026-08-26 ("implement phase 0 and 1") |
| 0.2 | Version control | Completed | Already a git repo; no `git init` needed |
| 0.3 | Python environment | Completed | Existing `venv/` (3.13.9) adopted — see Deviation 3 |
| 0.4 | Write and install `requirements.txt` | Completed | 5 direct pins; all wheels, no source builds |
| 0.5 | Create data root `~/physionet-data` | Completed | Outside the project tree |
| 0.6 | Download script | Completed | `scripts/download_mitbih.py` + `src/ecg/acquire.py` |
| 0.7 | Run the MIT-BIH download | Completed | 48 records, 93.9 MB, all verified |
| 0.8 | `README.md` incl. MIMIC-III access options | Completed | 190 lines |

### Phase 1 — MIT-BIH preprocessing

| # | Step | Status | Files | Tests |
|---|------|--------|-------|-------|
| 1.1 | Constants and paths | Completed | `src/ecg/config.py` | indirect |
| 1.2 | Synthetic ECG + WFDB fixtures | Completed | `tests/conftest.py` | — |
| 1.3 | Record/annotation loading, lead-by-name | Completed | `src/ecg/io.py` | 10 (F1–F5+) |
| 1.4 | Bandpass + optional notch | Completed | `src/ecg/filters.py` | 8 (A1–A6) |
| 1.5 | Z-score fit/apply | Completed | `src/ecg/normalize.py` | 10 (B1–B4+) |
| 1.6 | AAMI symbol mapping | Completed | `src/ecg/labels.py` | 37 (C1–C4+) |
| 1.7 | Beat and window segmentation | Completed | `src/ecg/segment.py` | 15 (D1–D6+) |
| 1.8 | Record-wise + DS1/DS2 splits | Completed | `src/ecg/splits.py` | 16 (E1–E6+) |
| 1.9 | **Verify DS1/DS2 against the published reference** | Completed | — | E4 committed |
| 1.10 | Pipeline orchestration + manifest | Completed | `src/ecg/pipeline.py` | 14 (G1–G13) |
| 1.11 | CLI entry point | Completed | `scripts/preprocess_mitbih.py` | via G1 |
| 1.12 | Full suite run | Completed | — | 123 pass |
| 1.13 | Real 48-record preprocessing pass | Completed | outputs on disk | 9 `requires_data` pass |
| 1.14 | Final review of the complete diff | Completed | — | — |

### Later phases
Phases 2–4 (MIMIC-III, waveform/signal quality, fusion + agentic layer) remain
**not planned in detail and not approved.** Each gets its own planning pass.

---

## Test Results

Verbatim, observed. Nothing here is reported as passing that was not seen passing.

| Run | Date | Command | Result |
|-----|------|---------|--------|
| Offline suite | 2026-08-26 | `pytest -q` | **123 passed, 9 deselected in 0.88s** |
| Offline, no data root | 2026-08-26 | `SMART_HEALTH_DATA=/tmp/definitely-no-data-here pytest -q` | **123 passed, 9 deselected in 0.76s** |
| Real-data suite | 2026-08-26 | `pytest -q -m requires_data` | **9 passed, 123 deselected in 0.38s** |
| Download | 2026-08-26 | `python scripts/download_mitbih.py` | **48 records verified, 93.9 MB** |
| Full preprocessing | 2026-08-26 | `python scripts/preprocess_mitbih.py` | **48 ok, 0 failed, 109,446 beats, 2.3 s** |
| Determinism | 2026-08-26 | second run to a different output dir | **all 48 `.npz` arrays + all 3 text artifacts identical** |

Test counts by group:

| Group | Module | Tests |
|-------|--------|-------|
| A | filters | 8 |
| B | normalize | 10 |
| C | labels | 37 |
| D | segment | 15 |
| E | splits | 16 |
| F | io | 10 |
| G | pipeline | 14 |
| H | download | 12 |
| — | real data (`requires_data`) | 9 |

No lint or type checker is configured for this project, so none was run. Nothing
was skipped, `xfail`ed, or weakened to make an implementation pass.

---

## Results on the real database

All 48 records at 360 Hz, two channels. Figures match the published description
of MIT-BIH Arrhythmia Database v1.0.0.

| Quantity | Value |
|---|---|
| Total annotations | 112,647 |
| Beats kept | 109,446 |
| Dropped at record boundary | 48 |
| Excluded non-beat annotations | 3,153 |
| Unknown symbols | **0** |
| Reconciles | Yes, exactly |

Beat annotations = 109,446 + 48 = **109,494** — the documented total.

Two records (102, 104) carry no MLII lead and fell back to V5, exactly the case
name-based lead selection exists to handle. Both are paced records, excluded from
the splits.

Largest excluded non-beat categories: `+` rhythm change (1,291), `~` signal
quality (616), `!` ventricular flutter wave (472), `"` comment (437).

---

## Changes Made

Files created:

| Path | What |
|------|------|
| `requirements.txt` | 5 pinned direct dependencies |
| `pytest.ini` | testpaths, `pythonpath=src`, `requires_data` marker, `RuntimeWarning`-as-error |
| `README.md` | Setup, run instructions, guarantees, results, MIMIC-III options |
| `src/ecg/{__init__,config,errors,io,filters,normalize,labels,segment,splits,acquire,pipeline}.py` | The pipeline |
| `scripts/{download_mitbih,preprocess_mitbih}.py` | CLI entry points |
| `tests/{conftest,test_io,test_filters,test_normalize,test_labels,test_segment,test_splits,test_pipeline,test_download,test_real_data}.py` | 132 tests |

Files moved: `PLAN.md` and `IMPLEMENTATION.md` → `PlanDoc/` (via `git mv`, at your
request). `PERMISSIONS.md` stayed in the root — say the word and it moves too.

Machine state changed outside the project: `venv/` populated with the pinned
dependencies; `~/physionet-data/` created holding 93.9 MB raw + processed output.
The `base` and `fairml312` conda environments and the system Python were not
touched. `Database use.docx.pdf` was not modified.

**Nothing was committed.** The working tree holds the full diff for your review.

---

## Deviations

**Deviation 3 — Python 3.13.9 instead of the planned 3.12 (accepted 2026-08-24).**
Carried over from the earlier draft. *Outcome:* confirmed harmless — every
dependency installed from a prebuilt cp313 arm64 wheel, no source builds.

**Deviation 4 — no relocation of the project folder (2026-08-24).**
The iCloud premise was false. Retained: the separate data root stands on its own
merits.

**Deviation 5 — `matplotlib` entered the environment transitively.**
*What changed:* `PLAN.md` deliberately excluded `matplotlib` from Phase 0/1.
`wfdb==4.3.0` depends on it, so it installed anyway, along with `soundfile`,
`aiohttp` and `requests`.
*Why it stands:* removing it would mean vendoring or forking `wfdb`, which is a
far worse trade than an unused transitive dependency.
*Added risk:* negligible — it is not imported anywhere in `src/`. Recorded so the
dependency list in `PLAN.md` is not read as the complete installed set.

**Deviation 6 — `src/ecg/errors.py` and `src/ecg/acquire.py` added.**
*What changed:* two modules `PLAN.md`'s file listing did not name.
*Why:* `errors.py` holds the typed exceptions the plan's edge-case list requires
(e.g. "a specific exception naming the length problem"), which had no home in the
planned layout. `acquire.py` holds the download and integrity logic so it can be
unit-tested without importing a script by path; `scripts/download_mitbih.py` is a
thin CLI over it.
*Added risk:* none. No public API beyond what the task needs.

**Deviation 7 — determinism is verified on array contents, not file bytes.**
*What changed:* `PLAN.md` test G3 said "identical file hashes".
*Why:* `np.savez` writes a zip archive whose member headers embed a wall-clock
timestamp, so two identical runs always differ in file bytes. Hashing the array
payloads (dtype, shape, contents) measures the determinism that actually matters.
The three text artifacts — `manifest.csv`, `splits.json`, `normalization.json` —
*are* compared byte-for-byte and are identical across runs.
*Added risk:* none. This is a stricter reading of the same property, and the
reason is documented in the test itself.

**Deviation 8 — `!` (ventricular flutter wave) is excluded, not mapped to V.**
*What changed:* `PLAN.md` did not rule on this symbol.
*Why:* it marks a wave inside a flutter episode, not a discrete classifiable
beat, and the canonical AAMI class table does not list it. Counting its 472
occurrences (all in record 207) as class V would inflate V with a morphology that
is not a ventricular ectopic beat. They are counted as excluded and visible in
the manifest, never silently dropped.
*Added risk:* a reader expecting the looser convention will see 472 fewer V
beats. Flagged here because it is a judgement call, not a fact.

**Anticipated deviation 1 (DS1/DS2 lists) — resolved, did not occur.**
The lists were verified before test E4 was committed. See below.

---

## Problems Discovered

1. ~~`~/Desktop` is iCloud-synced.~~ **RETRACTED 2026-08-24** — the claim was
   inferred from the existence of a CloudDocs path and was wrong. Verified by
   inode comparison and absence of cloud xattrs: Desktop & Documents sync is off,
   the project is on plain local disk.

2. **Default `python3` is 3.14.2.** Resolved by using `venv/` on 3.13.9.

3. ~~No version control.~~ The folder *is* a git repository (remote
   `github.com/Tanushree28/SmartHealthcare-Agentic`, branch `master`, 4 commits).
   The earlier "not a git repo" finding was from before the repo was initialised.

4. **No data present.** Resolved — 48 records downloaded and verified.

5. **`wget` is absent.** Moot: `wfdb.dl_database` was used.

6. **Class weights were computed over held-out data (found and fixed 2026-08-26).**
   The first real run reported `train_class_counts` totalling 109,446 — every
   processed beat, including the validation split, the test split, and the four
   excluded paced records. The training split actually holds 73,116 beats. The
   suggested class weights were therefore shaped in part by test data: mild
   leakage, but exactly the category `PLAN.md` is built to prevent, and it would
   have propagated silently into a loss function later.
   *Caught by:* the implausible Q share (7.35%) in the first run's summary.
   *Fixed by:* `PipelineResult.class_totals(split=...)`; `normalization.json` now
   reports train-only counts and adds a separate `all_processed_class_counts`
   field. Regression test G11 asserts the train counts equal the beats actually on
   disk for the training records and differ from the overall total.

7. **Duplicate record ids produced a split-leakage error (found and fixed
   2026-08-26).** Passing a record id twice put it in two splits;
   `SplitAssignment.validate()` correctly raised, but the message read as a
   leakage bug rather than bad input. Record ids are now deduplicated in
   `splits._eligible` and `pipeline.run`. Regression test G13.

8. **Rounding starved the validation split on small record sets (found and fixed
   2026-08-26).** With 3 records and the default (0.7, 0.15, 0.15) fractions,
   validation rounded to zero. Allocation now shrinks the training split rather
   than starving val or test. Caught by a test written before the fix.

---

## DS1/DS2 verification (step 1.9)

`PLAN.md` required this list be verified against the published source rather than
recalled, and that test E4 be dropped otherwise. **It was verified, so E4 ships.**

Source: de Chazal, O'Dwyer & Reilly, "Automatic Classification of Heartbeats Using
ECG Morphology and Heartbeat Interval Features", *IEEE Trans. Biomed. Eng.*
51(7):1196–1206, 2004, §II-B. Checked 2026-08-26 against two independent
publications restating the lists verbatim:

- <https://arxiv.org/html/2503.07276v1>
- <https://par.nsf.gov/servlets/purl/10086899>

Both give identical membership, matching the committed lists exactly:

- **DS1** (22): 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201,
  203, 205, 207, 208, 209, 215, 220, 223, 230
- **DS2** (22): 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213,
  214, 219, 221, 222, 228, 231, 232, 233, 234

Structural check: disjoint, 22 each, union is exactly the 48 records minus the
four paced ones (102, 104, 107, 217).

---

## Confirmed facts vs. assumptions

### Now confirmed (previously assumed)
- MIT-BIH download size: **93.9 MB** on disk (plan estimated ~100 MB).
- `wfdb` 4.3.0 installs and works on Python 3.13 / arm64 from a wheel.
- DS1/DS2 record membership — verified, see above.
- **All 48 records are 360 Hz and two-channel** — asserted, not assumed, by
  `test_real_data.py::test_every_record_is_360_hz_two_channel`.
- Exactly two records (102, 104) lack MLII.
- Zero annotation symbols fell outside the AAMI mapping table.
- Full preprocessing takes ~2.3 s, not "many minutes" — the long-running-command
  concern in `PERMISSIONS.md` did not apply.

### Still assumed / not verified
- MIMIC-III full-database size and current credentialing steps — not checked; no
  MIMIC-III work was done.
- The `!`-as-non-beat decision follows the canonical AAMI table, but conventions
  in the literature vary (Deviation 8).
