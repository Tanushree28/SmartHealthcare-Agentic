# PLAN.md — Smart Healthcare: Agentic Cardiac AI

Status: **Phase 0 + Phase 1 IMPLEMENTED and verified (2026-08-26).**
Phases 2–4 remain plan-only and unapproved.
See [IMPLEMENTATION.md](IMPLEMENTATION.md) for what was actually done, the
deviations from this document, and the real test output.

Source of requirements: `Database use.docx.pdf` (10 pages, in the repository root).
Date: 2026-08-22 (plan) / 2026-08-26 (Phase 0+1 executed)

---

## Goal

### What needs to be built

Build the data foundation for the agentic cardiac-AI system described in the PDF: a
reproducible, tested Python preprocessing pipeline that turns raw PhysioNet data into
model-ready arrays, plus the documented procedure for obtaining that data on this Mac.

The PDF describes a large system (ECG signal preprocessing, MIMIC-III clinical
preprocessing, and a fusion/agentic decision layer). That is far too much for one
implementation pass. This plan therefore **phases** the work and asks for approval on
Phase 0 + Phase 1 only.

| Phase | Content | Status in this plan |
|-------|---------|---------------------|
| **0** | Environment + data acquisition on macOS | **Done** (2026-08-26) |
| **1** | MIT-BIH preprocessing pipeline (PDF §2) | **Done** (2026-08-26) |
| 2 | MIMIC-III preprocessing (PDF §3) | Planned, **not** approved |
| 3 | Waveform/signal-quality layer (PDF §4) | Planned, **not** approved |
| 4 | Fusion + agentic decision layer (PDF §5) | Planned, **not** approved |

**Why start here:** every later phase consumes the output of Phase 1, the data is openly
licensed (no credentialing wait), and it is small enough to be fully tested. Phase 2 has a
hard external dependency — credentialed MIMIC-III access — that can take days to clear, so
starting there would block on paperwork.

### What successful completion looks like (Phase 0 + 1)

1. A documented, repeatable environment setup on this machine that does not disturb the
   existing `base` / `fairml312` conda environments.
2. MIT-BIH Arrhythmia Database v1.0.0 present on local disk at a data root **outside the
   project tree**, with an integrity check.
3. `python scripts/preprocess_mitbih.py` produces, from the 48 raw records:
   - per-record segmented beat arrays + labels on disk,
   - a manifest CSV (record, beat count, per-class counts, drop reasons),
   - a split assignment file with **zero record overlap** between train/val/test.
4. `pytest` passes, with tests that run **without network access and without the real
   dataset** (synthetic WFDB fixtures).
5. `IMPLEMENTATION.md` records what was actually done and the real test output.

### Explicit non-goals for this phase

No model training, no neural networks, no agent framework, no MIMIC-III access, no web
service, no deployment. Those are later phases and would be separately planned.

---

## Current Behavior

**There is no current behavior. This is a greenfield project.**

Confirmed by inspection of `/Users/tanushreenepal/Desktop/Smart Healthcare`:

- Contents: exactly one file, `Database use.docx.pdf`. No source code, no tests, no
  dependency manifest, no data.
- **Not a git repository.** No version control history exists.

### Confirmed environment facts (verified by command, 2026-08-22)

| Fact | Value |
|------|-------|
| Platform | macOS 26.6.2, `arm64` (Apple Silicon) |
| Default `python3` | 3.14.2 (python.org framework build) |
| Other interpreters | `/usr/bin/python3` (system), `/opt/anaconda3/bin/python3` (3.13) |
| Package managers | `brew`, `conda` present; `pyenv`, `uv`, `wget` **absent**; `curl`, `git` present |
| Existing conda envs | `fairml312` |
| Free disk | ~187 GB on the data volume |
| **iCloud sync** | **OFF for this project.** Verified 2026-08-24: `~/Desktop` (inode 3001238) and `~/Library/Mobile Documents/com~apple~CloudDocs/Desktop` (inode 3010667) are *different* directories; the CloudDocs stub holds only `.DS_Store`/`.localized`; no `fileprovider`/`ubiquity` xattrs on any project file. iCloud Drive is enabled account-wide, but Desktop & Documents folder sync is not. |
| Existing venv | `venv/`, Python **3.13.9**, created from `/opt/anaconda3/bin/python` |

### Root cause / core problem to solve

Two environment problems must be solved before any preprocessing code can run reliably:

1. **Dataset storage needs a home outside the repository.** *(Corrected 2026-08-24: an
   earlier draft of this plan claimed `~/Desktop` was iCloud-synced. That was inferred from
   the mere existence of a CloudDocs path and was wrong — see the verification in the table
   above. The project is on plain local disk and nothing here syncs anywhere.)* The reason
   to keep data out of the project tree therefore is **not** iCloud: it is that ~100 MB of
   raw MIT-BIH (and potentially multi-GB MIMIC extracts) should never enter git history,
   should survive a re-clone, and should be shareable across projects. A separate data root
   accomplishes all three.
2. **Python 3.14 is newer than the scientific stack's guaranteed wheel coverage.** Building
   `scipy`/`numpy` from source on 3.14 is slow and failure-prone. A pinned 3.12 environment
   avoids this entirely.

Neither is a code bug; both are setup decisions that are cheap now and expensive later.

---

## Expected Behavior

After Phase 0 + 1:

### Environment
- The existing `venv/` on **Python 3.13.9** (already created by you at the project root;
  see Deviation 3 in `IMPLEMENTATION.md`). No new conda environment is created, and `base`
  and `fairml312` are left untouched.
- `requirements.txt` pins the direct dependencies actually used.

### Data location
- Raw data at `~/physionet-data/mitdb/` — on local disk, outside the project tree.
- The repo references it through the `SMART_HEALTH_DATA` environment variable, defaulting
  to `~/physionet-data`. No absolute paths hardcoded in source.
- `data/` inside the repo, if used at all, holds only a symlink and is git-ignored.

### Pipeline behavior
Running:

```
python scripts/download_mitbih.py          # one-time, ~100 MB, network
python scripts/preprocess_mitbih.py
```

produces under `$SMART_HEALTH_DATA/processed/mitbih/`:

- `beats/<record>.npz` — `X` of shape `(n_beats, 300)` float32, `y` of shape `(n_beats,)`
  int8 (AAMI class index), `r_peaks` of shape `(n_beats,)` int64.
- `manifest.csv` — one row per record: record id, lead used, sampling rate, total
  annotations, beats kept, beats dropped and why, per-AAMI-class counts.
- `splits.json` — record ids assigned to `train` / `val` / `test`, plus the list of
  excluded paced records.
- `normalization.json` — z-score statistics **computed from training records only**.

### Guaranteed properties (these are what the tests assert)
1. **No record appears in more than one split.** Beat-level random splitting is never used.
2. Filtering is **zero-phase**; R-peak sample indices are not shifted by preprocessing.
3. Beat segments are R-peak aligned: the annotated R-peak sits at a fixed offset within
   every segment.
4. Beats whose window would extend past a record boundary are **dropped and counted**, not
   silently zero-padded.
5. Non-beat annotations (rhythm markers, signal-quality markers) are **excluded**, never
   mapped to a beat class.
6. Normalization statistics never see validation or test data.
7. Re-running the pipeline on the same input produces byte-identical output.

---

## Affected Areas

Everything is new; nothing existing is modified.

### Files to be created

```
Smart Healthcare/
├── PLAN.md                       (this file)
├── PERMISSIONS.md
├── IMPLEMENTATION.md
├── README.md                     setup + run instructions
├── requirements.txt
├── .gitignore                    excludes data/, *.npz, .env, __pycache__
├── src/
│   └── ecg/
│       ├── __init__.py
│       ├── config.py             constants: fs, band edges, window sizes, paths
│       ├── io.py                 WFDB record + annotation loading, lead selection
│       ├── filters.py            bandpass, optional notch
│       ├── normalize.py          z-score, stat fit/apply
│       ├── labels.py             MIT-BIH symbol → AAMI 5-class mapping
│       ├── segment.py            beat-level and window-level segmentation
│       ├── splits.py             record-wise / inter-patient splits
│       └── pipeline.py           orchestration, manifest generation
├── scripts/
│   ├── download_mitbih.py        data acquisition + integrity check
│   └── preprocess_mitbih.py      CLI entry point
└── tests/
    ├── conftest.py               synthetic ECG + synthetic WFDB record fixtures
    ├── test_io.py
    ├── test_filters.py
    ├── test_normalize.py
    ├── test_labels.py
    ├── test_segment.py
    ├── test_splits.py
    └── test_pipeline.py
```

### Dependencies to be introduced (each requires approval — see PERMISSIONS.md)

| Package | Why it is genuinely needed |
|---------|---------------------------|
| `wfdb` | Reference reader for PhysioNet WFDB format. Writing a parser by hand for the binary signal + annotation format would be a large, bug-prone detour. |
| `numpy` | Array representation. Unavoidable. |
| `scipy` | `butter`/`filtfilt`/`iirnotch`. Hand-rolling IIR filter design is not reasonable. |
| `pandas` | Manifest CSV assembly. Could be dropped for `csv` if you prefer fewer deps. |
| `pytest` | Test runner (you chose it). |

Deliberately **not** added now: `scikit-learn`, `matplotlib`, `torch`, `imbalanced-learn`.
They belong to modelling phases, not preprocessing.

### External systems affected
- **PhysioNet** (`physionet.org`) — read-only HTTP download of an openly licensed dataset.
  No account, no remote state change.
- Local disk: ~100 MB raw + roughly a few hundred MB processed (to be measured, not
  assumed).
- **Nothing is pushed anywhere.** No git remote, no deployment, no external API.

---

## Proposed Approach

### Phase 0 — Environment and data acquisition

**Step 0.1 — Use the existing `venv/` (Python 3.13.9).**
You already created `venv/` at the project root from the anaconda base interpreter. It is
used as-is; `pip install -r requirements.txt` runs inside it. Rationale: Python 3.13 has
full wheel coverage on arm64 for numpy/scipy/pandas/wfdb, so the 3.12 pin this plan
originally proposed buys nothing. `venv/` is git-ignored. *Requires approval (dependency
install).*

**Step 0.2 — Create the data root outside the project tree.**
`~/physionet-data/`, read from `$SMART_HEALTH_DATA` with that default. Rationale: keeps
large binaries out of any future git history, lets the data survive a re-clone or a move of
the project folder, and keeps the code machine-independent (no absolute paths in source).
An in-project `data/` directory would work too and is git-ignored either way — the separate
root is preferred only so the dataset is not tied to this one checkout.

**Step 0.3 — Download MIT-BIH Arrhythmia Database v1.0.0.**
`scripts/download_mitbih.py` uses `wfdb.dl_database('mitdb', dl_dir=...)`, which is the
maintained path and handles the `.dat`/`.hea`/`.atr` triplets per record. Fallback if that
API misbehaves: `curl` the published archive from PhysioNet. The script is **idempotent** —
it skips records already present and verifies each record opens and reports the expected
sampling rate before declaring success. Licensing: MIT-BIH is openly available on
PhysioNet under an open licence; no credentialing, no DUA. *Approximate size ~100 MB — to
be confirmed at download time, not asserted here.* *Requires approval (network + bulk
download).*

**Step 0.4 — Document the MIMIC-III path without acting on it.**
`README.md` will record the three MIMIC-III options and their real costs, so the Phase 2
decision is informed:

| Option | Access | Rough size | Practical note |
|--------|--------|-----------|----------------|
| MIMIC-III Clinical Database **Demo** v1.4 | Open, no credentialing | tens of MB | ~100 patients. Correct choice for building and testing pipeline code. |
| MIMIC-III Clinical Database v1.4 (full) | **Credentialed**: PhysioNet account + CITI human-subjects training + signed data use agreement | several GB gzipped | Approval is a human process taking days to weeks. Start the application early if Phase 2 is wanted. |
| MIMIC-III Waveform Matched Subset | Open | **terabyte scale** | Not viable on 187 GB free. Out of scope. |

I will not create accounts, submit applications, accept data use agreements, or handle your
credentials. Those are yours to do.

### Phase 1 — MIT-BIH preprocessing pipeline

Built in dependency order, each module tested as it lands (PDF §2.1–§2.9).

**Step 1.1 — `config.py`.** Central constants: `FS = 360`, band `0.5–40 Hz`, filter order 3,
`SAMPLES_BEFORE = 100`, `SAMPLES_AFTER = 200` (→ 300-sample beats, matching the PDF),
AAMI class order, paced-record list, data paths. Rationale: the PDF's numbers appear in
one place, so changing a window size is a one-line edit rather than a search-and-replace.

**Step 1.2 — `io.py` — load records and annotations (PDF §2.1).**
Wraps `wfdb.rdrecord` / `wfdb.rdann`. **Selects the lead by signal name, not by column
index.** This matters: most MIT-BIH records are MLII in channel 0, but not all are, and
indexing blindly would silently mix different leads into one training set — a real and
easy-to-miss data-quality bug. Configurable preferred-lead order with an explicit error if
none is present.

**Step 1.3 — `filters.py` — denoising (PDF §2.2).**
Butterworth bandpass 0.5–40 Hz applied with `scipy.signal.filtfilt`.
- *Why `filtfilt` and not `lfilter`:* `filtfilt` is zero-phase. A causal IIR filter
  introduces frequency-dependent group delay, which shifts the R-peak away from its
  annotated sample index — every beat window would then be misaligned by an unknown amount.
  This single choice protects the alignment guarantee.
- *On the 50/60 Hz notch:* the PDF lists a notch filter. With a 40 Hz low-pass already in
  place, powerline interference is **already attenuated**, so a notch at 60 Hz is a no-op.
  I will implement `notch()` but leave it **disabled by default**, documenting that it
  becomes relevant only if the passband is widened past 50 Hz. Implementing a filter that
  does nothing and pretending it helps would be worse than saying so.
- Short-signal guard: `filtfilt` raises when `padlen` exceeds signal length; handled with a
  clear error rather than a stack trace.

**Step 1.4 — `normalize.py` (PDF §2.3, §3.9).**
Per-record z-score by default, with a `fit`/`apply` split so statistics can be computed on
training records only and reused elsewhere. Zero-variance guard via epsilon so a flatline
segment yields zeros, not `NaN`.

**Step 1.5 — `labels.py` — AAMI mapping (PDF §2.6).**
Explicit dict from MIT-BIH beat symbols to the five AAMI classes (N, S, V, F, Q).
Annotation symbols that are **not beats** (rhythm changes, signal-quality markers, and
similar) are routed to an explicit `EXCLUDED` outcome and counted — never folded into `Q`.
Rationale: mislabelling a rhythm marker as an unclassifiable beat injects phantom beats
into the dataset. Any symbol not in the table raises or is counted as unknown depending on
a strict-mode flag, so a silent mapping gap cannot pass unnoticed.

**Step 1.6 — `segment.py` (PDF §2.4, §2.5).**
Beat-level: slice `[r - 100, r + 200)` around each annotated R-peak. Window-level: fixed
5/10-second windows for later rhythm-context models. Out-of-bounds beats at record edges
are dropped with a counted reason. The annotated R-peaks are used directly (PDF §2.4
sanctions this); an independent detector is explicitly deferred, not silently skipped.

**Step 1.7 — `splits.py` (PDF §2.8 — "This is very important").**
Record-wise splitting only. The API accepts no beat-level split mode at all — the safest
way to prevent leakage is to make it unexpressible. Two strategies: a seeded record-wise
split, and the standard inter-patient DS1/DS2 assignment used in the arrhythmia literature
for comparability. The four paced records (102, 104, 107, 217) are excluded by default per
AAMI convention. **The exact DS1/DS2 record lists will be verified against the published
reference before being committed — I will not ship a list I recalled from memory as if it
were checked.**

**Step 1.8 — `pipeline.py` + `scripts/preprocess_mitbih.py`.**
Iterate records → load → filter → normalize → segment → label → write `.npz` per record →
accumulate manifest rows → write `manifest.csv`, `splits.json`, `normalization.json`.
Per-record output rather than one large array: it keeps peak memory bounded, makes
record-wise splitting a file-selection operation, and lets a failed record be reprocessed
alone.

**Step 1.9 — Class imbalance (PDF §2.7): report, do not resample.**
The manifest reports exact per-class counts and the pipeline emits suggested class weights.
No SMOTE, no undersampling at this stage. Rationale: resampling is a *training* decision
that belongs next to the loss function, and baking it into preprocessing would destroy the
true class distribution and make honest evaluation impossible. The PDF's own note that
class-weighted or focal loss usually beats aggressive oversampling supports this.

### Design decisions summarized

| Decision | Chosen | Rejected alternative and why |
|----------|--------|------------------------------|
| Filter application | `filtfilt` (zero-phase) | `lfilter` — group delay misaligns R-peaks |
| Lead selection | by signal name | by index — silently mixes leads across records |
| Notch filter | implemented, off by default | on by default — misleading no-op under a 40 Hz low-pass |
| Split unit | record | beat — the leakage the PDF explicitly warns about |
| Imbalance handling | measured and reported | resampled in preprocessing — corrupts evaluation |
| Output layout | one file per record | one monolithic array — unbounded memory, awkward splits |
| Data location | separate root outside the repo | inside project — couples the dataset to one checkout, risks it entering git history |
| Python | existing `venv/` on 3.13.9 | system 3.14 — uncertain scientific wheel coverage; a new 3.12 env — no benefit over 3.13 |

---

## Edge Cases

Identified before implementation; each maps to a test below.

1. **Beat near a record boundary** — R-peak at sample 50 with a 100-sample left context.
   Drop and count; never zero-pad silently.
2. **Record lacking the preferred lead** — a small number of MIT-BIH records do not use
   MLII. Fall back through a configured preference order; error clearly if none match.
3. **Non-beat annotation symbols** — rhythm and quality markers must be excluded, not
   labelled.
4. **Unknown/unmapped symbol** — must surface, not silently vanish.
5. **Flatline or zero-variance segment** — z-score must not divide by zero.
6. **`NaN` / `inf` in the raw signal** — detect and fail loudly rather than propagate.
7. **Signal shorter than the filter's `padlen`** — clear error.
8. **Unexpected sampling rate** — assert 360 Hz for MIT-BIH; a differing rate must not be
   processed with hardcoded assumptions.
9. **Non-monotonic or duplicated annotation samples** — must not produce overlapping or
   negative-length segments.
10. **A class absent from a split** — F and Q beats are very rare; a split can legitimately
    contain none. Report it; do not crash.
11. **Empty record / zero annotations** — produce an empty result with a manifest row, not
    an exception.
12. **Re-running the download** — idempotent, must not re-fetch or corrupt existing files.
13. **Partial/interrupted download** — integrity check must catch a truncated record.
14. **Paced records** — excluded by default, and the exclusion must be visible in output.
15. **`SMART_HEALTH_DATA` unset or pointing nowhere** — actionable error message naming the
    variable, not a bare `FileNotFoundError`.

---

## Risks / Regressions

Greenfield, so there is no existing behavior to regress. The real risks are correctness
risks that would silently produce a plausible-looking but invalid dataset:

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Data leakage via beat-level splitting** | Inflated accuracy; the whole result is invalid | Record-wise split is the only implemented mode; a dedicated test asserts zero record overlap |
| **Leakage via normalization statistics** | Subtler version of the same | `fit` on train records only; test asserts stats are unchanged by test-set content |
| Phase distortion misaligning beats | Every segment off by an unknown shift | `filtfilt`; test asserts R-peak index is preserved |
| Inconsistent lead across records | Model learns lead artifacts | Name-based selection; manifest records the lead per record |
| Non-beat annotations counted as beats | Phantom beats, corrupted class counts | Explicit exclusion set; test per symbol category |
| DS1/DS2 lists recalled incorrectly | Results not comparable to published work | Verify against the reference before committing; otherwise ship only the seeded split |
| Tests depending on downloaded data | Suite unrunnable offline / on a fresh clone | All tests use synthetic WFDB fixtures written to `tmp_path` |
| Python 3.14 wheel gaps | Long source builds or hard failure | Pinned 3.12 environment |
| Dataset committed to git | Bloated history, patient data in version control | `.gitignore` excludes all WFDB/array/data patterns; data root sits outside the repo |
| Silent dependency drift | Irreproducible runs | Pinned `requirements.txt`; recorded in `IMPLEMENTATION.md` |

**Must remain unchanged:** the `base` and `fairml312` conda environments, the system Python,
and `Database use.docx.pdf` (the source document — never edited).

---

# Test Plan

Testing is part of the implementation, written alongside each module.

**Framework:** `pytest`.
**Hard constraint:** the entire suite runs **offline** with **no real dataset present**.
Fixtures synthesize ECG-like signals and write real WFDB records into `tmp_path`, so the
WFDB reading path is genuinely exercised without a download. Tests that would require the
real database are marked `@pytest.mark.requires_data` and skipped by default.

**Fixtures (`conftest.py`)**
- `synthetic_ecg` — a deterministic signal with QRS-like spikes at known sample indices,
  plus optional injected baseline wander (0.05 Hz) and a 60 Hz tone.
- `synthetic_record` — a two-channel WFDB record with named leads and an annotation file
  containing a known mix of beat and non-beat symbols, written to `tmp_path`.
- `tiny_dataset` — three synthetic records, enough to exercise splitting and the manifest.

---

### Group A — Filtering (`test_filters.py`)

**A1. Bandpass removes baseline wander while preserving QRS amplitude**
- *Testing:* the denoising step actually does its job.
- *Setup:* `synthetic_ecg` + a 0.05 Hz sinusoid at 3× QRS amplitude.
- *Action:* `bandpass(sig, fs=360, low=0.5, high=40)`.
- *Expected:* residual power below 0.2 Hz drops by a large, asserted margin; QRS peak
  amplitude retained within tolerance.
- *Why it matters:* a filter that removes drift but also flattens the QRS complex destroys
  the very morphology the classifier needs.

**A2. Filtering is zero-phase — R-peak indices are preserved**
- *Testing:* the alignment guarantee.
- *Setup:* signal with spikes at known indices.
- *Action:* filter, then locate maxima.
- *Expected:* every recovered peak index within ±1 sample of the original.
- *Why it matters:* this is the single assumption every downstream beat window depends on.
  It is also exactly what breaks if someone later "optimizes" `filtfilt` into `lfilter`.

**A3. Notch attenuates 60 Hz when the passband admits it**
- *Setup:* signal + 60 Hz tone, bandpass widened to 100 Hz, notch enabled.
- *Expected:* 60 Hz spectral component strongly attenuated; neighbouring bands intact.
- *Why it matters:* verifies the notch is real and correctly centred, so the
  off-by-default choice is documented rather than a cover for broken code.

**A4. Signal shorter than `padlen` raises a clear error**
- *Input:* 10-sample array.
- *Expected:* a specific exception naming the length problem — not a raw scipy error.
- *Why it matters:* short trailing fragments are a realistic input; the failure must be
  diagnosable.

**A5. `NaN` in the input is rejected**
- *Expected:* explicit error.
- *Why it matters:* `filtfilt` propagates `NaN` across the whole array, so one bad sample
  would silently void an entire record.

---

### Group B — Normalization (`test_normalize.py`)

**B1. Z-score produces zero mean, unit variance** — standard correctness check within
floating-point tolerance.

**B2. Constant signal does not produce `NaN`**
- *Input:* all-ones array.
- *Expected:* finite output (zeros), no warning.
- *Why it matters:* flatline segments occur in real leads-off intervals; a `NaN` here
  poisons every downstream batch.

**B3. Statistics fitted on train only, applied to val/test**
- *Setup:* train records with mean 0, a test record offset by +50.
- *Action:* `fit(train)`, then `apply` to both.
- *Expected:* the fitted mean reflects only the training data; the test record retains its
  offset after transformation.
- *Why it matters:* **direct leakage regression test** for PDF §3.9.

**B4. `apply` is deterministic and idempotent given fixed statistics.**

---

### Group C — Label mapping (`test_labels.py`)

**C1. Every documented beat symbol maps to its correct AAMI class**
- *Action:* parametrized over the full symbol table.
- *Expected:* exact expected class per symbol.
- *Why it matters:* a single mis-keyed entry silently mislabels a whole beat category, and
  nothing downstream would reveal it.

**C2. Non-beat annotation symbols are excluded, not classified**
- *Input:* rhythm-change and signal-quality markers.
- *Expected:* `EXCLUDED`, and they appear in the manifest's excluded count.
- *Why it matters:* prevents phantom beats entering the dataset.

**C3. Unknown symbol behaviour is explicit**
- *Input:* a symbol absent from the table.
- *Expected:* raises in strict mode; counted as unknown otherwise. Never silently dropped.
- *Why it matters:* makes a mapping gap visible instead of quietly shrinking the dataset.

**C4. The five AAMI classes have a stable, documented index order**
- *Why it matters:* a reordering between runs would silently invalidate every saved label
  array and any trained model.

---

### Group D — Segmentation (`test_segment.py`)

**D1. Beat segment shape and R-peak alignment**
- *Expected:* shape `(n_beats, 300)`; the R-peak sits at index 100 in every row.
- *Why it matters:* the core contract of beat-level segmentation.

**D2. Boundary beat at the start of a record is dropped and counted**
- *Input:* R-peak at sample 50, left context 100.
- *Expected:* excluded from `X`; drop reason recorded as an edge case; no exception; no
  zero-padding.
- *Why it matters:* the tempting fix — pad with zeros — invents a flat pre-QRS segment that
  a model will happily learn from.

**D3. Boundary beat at the end of a record is dropped and counted** — mirror of D2.

**D4. Non-monotonic / duplicate annotation samples do not corrupt output**
- *Expected:* no negative-length or overlapping-index errors; deterministic handling.

**D5. Window-level segmentation yields the right count and length**
- *Input:* 30 s at 360 Hz, 10 s windows.
- *Expected:* 3 windows of 3600 samples; a documented rule for the trailing partial window.

**D6. Zero annotations yields an empty array of correct dtype and shape**
- *Why it matters:* the empty case must flow through the pipeline into a manifest row
  rather than raising.

---

### Group E — Splits (`test_splits.py`)

**E1. No record appears in more than one split** — *the* leakage test.
- *Expected:* pairwise intersections of train/val/test record sets are empty; their union
  equals the eligible record set.
- *Why it matters:* PDF §2.8 calls this out as critical, and it is the failure mode that
  produces impressive, meaningless accuracy.

**E2. Beat-level splitting is not offered**
- *Action:* attempt to request one.
- *Expected:* `TypeError`/`ValueError` — the mode does not exist.
- *Why it matters:* encodes the constraint in the API rather than in a comment.

**E3. Paced records are excluded by default and listed as excluded**
- *Expected:* 102, 104, 107, 217 in none of the splits, present in the excluded list.

**E4. DS1/DS2 assignment matches the verified reference lists**
- *Expected:* exact set equality against the reference.
- *Why it matters:* the point of DS1/DS2 is comparability with published results; an
  approximate list is worthless. **This test will only be committed once the lists are
  verified against the source.**

**E5. Seeded splits are reproducible; different seeds differ**
- *Why it matters:* reproducibility of every downstream number.

**E6. Every split is non-empty for the standard 44-record set** — guards a config change
that would starve validation.

---

### Group F — I/O (`test_io.py`)

**F1. Lead is selected by name, not index**
- *Setup:* synthetic record whose channel 0 is V5 and channel 1 is MLII.
- *Expected:* the MLII channel is returned.
- *Why it matters:* the exact bug that name-based selection exists to prevent; indexing
  would return V5 and no test would notice without this one.

**F2. Missing preferred lead falls back through the configured order.**

**F3. No configured lead present raises a clear, named error.**

**F4. Sampling rate mismatch is rejected**
- *Input:* record at 250 Hz.
- *Expected:* explicit error; not silently filtered with 360 Hz coefficients.
- *Why it matters:* wrong-`fs` filter coefficients produce a plausible-looking but wrongly
  filtered signal — a silent corruption.

**F5. Missing record path produces an actionable message naming `SMART_HEALTH_DATA`.**

---

### Group G — Pipeline integration (`test_pipeline.py`)

**G1. End-to-end run over `tiny_dataset`**
- *Action:* run the pipeline against three synthetic records in `tmp_path`.
- *Expected:* one `.npz` per record; `X`/`y`/`r_peaks` lengths agree; manifest row counts
  reconcile exactly with array lengths; `splits.json` and `normalization.json` written.
- *Why it matters:* verifies the modules compose, which unit tests cannot show.

**G2. Manifest arithmetic reconciles**
- *Expected:* `kept + dropped_boundary + excluded_nonbeat + unknown == total annotations`,
  per record.
- *Why it matters:* an accounting identity that catches any silently discarded beat.

**G3. Re-running produces identical output**
- *Expected:* identical file hashes.
- *Why it matters:* determinism is a precondition for trusting any later comparison.

**G4. A single corrupt record does not abort the whole run**
- *Setup:* one unreadable record among three.
- *Expected:* the other two complete; the failure is recorded in the manifest and the exit
  status reflects partial failure.
- *Why it matters:* a 48-record job should not lose 47 records' work to one bad file.

**G5. Normalization statistics in the output derive only from training records**
- *Why it matters:* end-to-end confirmation of B3 at the artifact level, where it actually
  counts.

---

### Group H — Data acquisition (`test_download.py`)

**H1. Download script is idempotent (no network)**
- *Setup:* fake an already-complete data directory.
- *Expected:* no download attempted; exit success.

**H2. Truncated record is detected**
- *Setup:* truncate a `.dat` file.
- *Expected:* integrity check fails with a message naming the record.
- *Why it matters:* a partial download otherwise surfaces much later as bizarre signal
  values.

**H3. Network calls are never made during the default test run**
- *Action:* run the suite with sockets patched to raise.
- *Expected:* full suite passes.
- *Why it matters:* keeps the suite fast, offline, and CI-safe.

---

### Not tested, deliberately

- The real 48-record MIT-BIH download (network + ~100 MB; behind `requires_data`).
- Filter numerical output against a reference implementation — `scipy` is trusted; what is
  tested is the *behaviour* of the chosen configuration.
- Anything MIMIC-III (Phase 2).
- No test exists purely to raise a coverage number.

---

## Open Questions — all resolved

1. ~~**Phase 0 + 1 scope**~~ — **Resolved 2026-08-26:** approved as described and
   implemented in full.
2. ~~Conda env on Python 3.12~~ — **Resolved 2026-08-24:** using your existing `venv/`
   on Python 3.13.9. Confirmed harmless: every dependency installed from a wheel.
3. ~~**Data root `~/physionet-data`**~~ — **Resolved 2026-08-26:** used as proposed;
   93.9 MB of raw data now lives there.
4. ~~**Git**~~ — **Resolved:** the folder *is* a repository (remote
   `github.com/Tanushree28/SmartHealthcare-Agentic`, branch `master`). No `git init`
   was needed and nothing was committed or pushed.
5. **MIMIC-III** — **still open, and still yours to decide.** Credentialing takes days
   to weeks, so if Phase 2 is wanted against the full database the application should
   start soon. The alternative is the open demo subset. No MIMIC-III work has begun and
   none will without a separate planning pass.
