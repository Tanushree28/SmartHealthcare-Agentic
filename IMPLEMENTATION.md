# IMPLEMENTATION.md — Smart Healthcare

Tracks actual execution against `PLAN.md`, under the policy in `PERMISSIONS.md`.

**Overall status: NOT STARTED — awaiting scope approval.**
No source file has been created, no dependency installed, no data downloaded.

Last updated: 2026-08-22

---

## Implementation Steps

Execution order. "Gate" marks a step that cannot proceed without your approval per
`PERMISSIONS.md`.

### Phase 0 — Environment and data

| # | Step | Files expected | Tests expected | Gate |
|---|------|----------------|----------------|------|
| 0.1 | Confirm scope, env choice, data root, git (PLAN.md open questions) | — | — | **Yes** |
| 0.2 | `git init` (optional, if you want a reviewable diff) | `.gitignore` | — | **Yes** |
| 0.3 | Create conda env `smart-health` (Python 3.12) | — | — | **Yes** |
| 0.4 | Write and install `requirements.txt` | `requirements.txt` | — | **Yes** |
| 0.5 | Create data root `~/physionet-data` | — | — | **Yes** |
| 0.6 | Write download script | `scripts/download_mitbih.py` | `tests/test_download.py` (H1–H3) | No |
| 0.7 | Run the MIT-BIH download (~100 MB, network) | data on disk | — | **Yes** |
| 0.8 | Write `README.md` incl. MIMIC-III access options | `README.md` | — | No |

### Phase 1 — MIT-BIH preprocessing

| # | Step | Files expected | Tests expected | Gate |
|---|------|----------------|----------------|------|
| 1.1 | Constants and paths | `src/ecg/config.py` | covered indirectly | No |
| 1.2 | Test fixtures: synthetic ECG + synthetic WFDB record | `tests/conftest.py` | — | No |
| 1.3 | Record/annotation loading, lead-by-name | `src/ecg/io.py` | `test_io.py` (F1–F5) | No |
| 1.4 | Bandpass + optional notch | `src/ecg/filters.py` | `test_filters.py` (A1–A5) | No |
| 1.5 | Z-score fit/apply | `src/ecg/normalize.py` | `test_normalize.py` (B1–B4) | No |
| 1.6 | AAMI symbol mapping | `src/ecg/labels.py` | `test_labels.py` (C1–C4) | No |
| 1.7 | Beat and window segmentation | `src/ecg/segment.py` | `test_segment.py` (D1–D6) | No |
| 1.8 | Record-wise + DS1/DS2 splits | `src/ecg/splits.py` | `test_splits.py` (E1–E6) | No |
| 1.9 | Verify DS1/DS2 lists against the published reference | `src/ecg/splits.py` | `test_splits.py` (E4) | **Yes** (needs network) |
| 1.10 | Pipeline orchestration + manifest | `src/ecg/pipeline.py` | `test_pipeline.py` (G1–G5) | No |
| 1.11 | CLI entry point | `scripts/preprocess_mitbih.py` | covered by G1 | No |
| 1.12 | Full suite run | — | all | No |
| 1.13 | Real 48-record preprocessing pass | processed outputs | `requires_data` marked | **Yes** (long-running) |
| 1.14 | Final review of the complete diff | — | — | No |

### Later phases
Phases 2–4 (MIMIC-III, waveform/signal quality, fusion + agentic layer) are **not planned
in detail and not approved.** Each gets its own planning pass before any code.

---

## Progress

Legend: `Not started` / `In progress` / `Completed` / `Blocked`

| Step | Status | Note |
|------|--------|------|
| Repository inspection | Completed | Only `Database use.docx.pdf` present; not a git repo |
| Requirements read from PDF | Completed | All 10 pages read |
| Environment survey | Completed | See "Confirmed facts" below |
| `PLAN.md` | Completed | — |
| `PERMISSIONS.md` | Completed | — |
| `IMPLEMENTATION.md` | Completed | This file |
| 0.1 Scope approval | **Blocked** | Waiting on your answers to PLAN.md's open questions |
| 0.2 – 1.14 | Not started | Blocked behind 0.1 |
| Phase 2 – 4 | Not started | Not planned in detail; out of current scope |

---

## Changes Made

**No code, configuration, dependency, or data changes have been made.**

Files created so far — planning documents only, all in the project root:

| File | What it is |
|------|-----------|
| `PLAN.md` | Goal, current/expected behavior, approach, edge cases, risks, full test plan |
| `PERMISSIONS.md` | Three-tier permission policy for this task |
| `IMPLEMENTATION.md` | This tracker |

Files read, not modified: `Database use.docx.pdf`.

Commands run so far were **read-only environment inspection**: `uname`, `sw_vers`, `which`,
`python3 -V`, `pip --version`, `df -h`, `ls`, `readlink`, `brctl status`, `mdls`, and
`strings` on the PDF for its page count. No file outside this folder was created or
modified.

---

## Deviations

None yet — implementation has not started.

Deviations from `PLAN.md` will be recorded here with: what changed, why, and whether it
adds risk. Per `PERMISSIONS.md`, a deviation that materially changes scope or needs new
permissions stops work and comes to you first.

Two deviations are **anticipated** and flagged now:

1. **DS1/DS2 record lists (step 1.9).** `PLAN.md` names these as the standard
   inter-patient split, but I have not verified the exact membership against the published
   source. If verification is not possible, I will ship only the seeded record-wise split
   and drop test E4 rather than commit an unverified list. Recording this now so it cannot
   quietly become an unchecked assumption.
2. **Python version.** If a required package turns out to lack a Python 3.12 wheel on
   arm64, the pin changes — and that is a dependency decision that comes back to you.

---

## Problems Discovered

Found during inspection, before any implementation:

1. **`~/Desktop` is iCloud-synced.** `~/Library/Mobile Documents/com~apple~CloudDocs/Desktop`
   exists, so "Desktop & Documents" sync is on and this project sits inside the synced
   tree. Storing datasets here would mean iCloud uploads, possible eviction of files to
   cloud-only stubs, and access stalls mid-run. *Resolution in plan:* data root at
   `~/physionet-data`, outside the synced tree; `.gitignore` and README will say why.

2. **Default `python3` is 3.14.2.** Newer than the scientific stack's guaranteed wheel
   coverage; `scipy`/`wfdb` may need source builds or may not build at all on arm64.
   *Resolution in plan:* pinned Python 3.12 conda environment.

3. **No version control.** Not a git repository, so there is no diff to review and no way
   to undo a bad change. *Resolution:* `git init` proposed as step 0.2, gated on your
   approval.

4. **No data present.** Neither MIT-BIH nor MIMIC-III is on this machine. *Resolution:*
   Phase 0 acquisition, with MIMIC-III's credentialing cost documented up front so it is
   not discovered late.

5. **`wget` is absent**, `curl` is present. Minor: the download path will use the `wfdb`
   library or `curl`, never `wget`.

---

## Test Results

**No tests have been written or executed.**

This section will record, verbatim, the actual `pytest` output — tests run, passed,
failed, skipped — plus any lint/type/build results, and any test that could not run and
why. Nothing will be reported as passing that was not observed passing.

| Run | Date | Command | Result |
|-----|------|---------|--------|
| — | — | — | Not yet run |

---

## Confirmed facts vs. assumptions

Kept explicit so nothing unverified is treated as established.

### Confirmed (observed directly, 2026-08-22)
- Project folder contains only `Database use.docx.pdf`; not a git repo.
- macOS 26.6.2, arm64.
- `python3` → 3.14.2; conda present with a 3.13 base and env `fairml312`; `brew`, `curl`,
  `git` present; `pyenv`, `uv`, `wget` absent.
- ~187 GB free on the data volume.
- iCloud CloudDocs Desktop path exists.
- The PDF's full contents (all 10 pages).

### Assumed / to verify before relying on
- MIT-BIH download size (~100 MB) and exact PhysioNet URL — verify at download time.
- MIMIC-III full-database size and current credentialing steps — verify on PhysioNet.
- `wfdb` package availability and API on Python 3.12 / arm64 — verify at install.
- DS1/DS2 record membership — verify against the published reference (step 1.9).
- That all 48 MIT-BIH records are 360 Hz two-channel — the pipeline asserts rather than
  assumes this.
