# SmartHealthcare-Agentic

Data foundation for an agentic cardiac-AI system: a reproducible, tested
preprocessing pipeline that turns raw PhysioNet ECG recordings into model-ready
arrays.

**Current scope: Phase 0 (environment + data acquisition) and Phase 1 (MIT-BIH
preprocessing).** No model training, no neural networks, no agent framework, no
MIMIC-III. Those are later phases — see [PlanDoc/PLAN.md](PlanDoc/PLAN.md).

> **Not a medical device.** Everything here produces research artifacts. Nothing
> in this repository is clinically validated, diagnostic, or fit for patient
> care.

**[View the results page &rarr;](https://tanushree28.github.io/SmartHealthcare-Agentic/)**
&nbsp;&mdash;&nbsp; a visual walkthrough of the pipeline and the baseline results.
It is generated from the processed data by `python scripts/build_report.py`, so
its figures always match the pipeline that produced them.

---

## Setup

Requires Python 3.13 (the checked-in `venv/` was built on 3.13.9, arm64 macOS).

```bash
python3 -m venv venv                     # if venv/ is absent
./venv/bin/pip install -r requirements.txt
```

### Where the data lives

Raw and processed data live **outside the repository**, under the path named by
`SMART_HEALTH_DATA` (default `~/physionet-data`):

```
~/physionet-data/
├── mitdb/                     raw WFDB records (48 records, ~94 MB)
└── processed/mitbih/
    ├── beats/<record>.npz     X (n, 300) float32, y (n,) int8, r_peaks (n,) int64
    ├── manifest.csv           per-record accounting
    ├── splits.json            record-wise train/val/test assignment
    └── normalization.json     z-score stats fitted on training records only
```

Keeping it outside the tree stops ~94 MB of binaries entering git history, lets
the dataset survive a re-clone, and keeps every path in `src/` machine-independent.

```bash
export SMART_HEALTH_DATA=/somewhere/else   # optional
```

---

## Running

```bash
./venv/bin/python scripts/download_mitbih.py     # one-time, ~94 MB, network
./venv/bin/python scripts/preprocess_mitbih.py   # ~2 s for all 48 records
```

The download is idempotent — records already present and passing their integrity
check are skipped, so re-running costs nothing. `--verify` checks what is on disk
and makes no network request.

Useful preprocessing flags:

| Flag | Effect |
|------|--------|
| `--strategy ds1_ds2` | Use the inter-patient DS1/DS2 split from de Chazal et al. (2004) instead of the seeded record-wise split |
| `--records 100 101` | Process a subset |
| `--include-paced` | Keep records 102, 104, 107, 217 (excluded by default per AAMI) |
| `--strict-symbols` | Fail on any annotation symbol missing from the AAMI table |
| `--notch` | Enable the 60 Hz notch (a no-op under the default 40 Hz low-pass) |

Exit status is non-zero if any record fails, even when others succeed.

### Tests

```bash
./venv/bin/python -m pytest              # 123 tests, offline, no dataset needed
./venv/bin/python -m pytest -m requires_data   # 9 more, needs the real database
```

The default suite synthesizes real WFDB records in `tmp_path`, so the reading
path is genuinely exercised without a download. It passes on a fresh clone with
`SMART_HEALTH_DATA` pointing at nothing.

---

## What the pipeline guarantees

These are asserted by tests, not just intended:

1. **No record appears in more than one split.** Beat-level splitting is not
   implemented and no function accepts a split-unit argument — the leakage PDF
   §2.8 warns about is unexpressible rather than merely discouraged.
2. **Normalization statistics never see validation or test data**, and
   `normalization.json` names the records they came from so the claim is auditable.
3. **Filtering is zero-phase** (`filtfilt`), so R-peak sample indices are not
   shifted; the annotated R-peak sits at index 100 of every 300-sample beat.
4. **Leads are selected by name, not column index.** Records 102 and 104 carry no
   MLII; index-based selection would have silently mixed V5 into the dataset.
5. **Beats whose window crosses a record boundary are dropped and counted**, never
   zero-padded. 48 beats across the database are dropped this way.
6. **Non-beat annotations are excluded, never mapped to a beat class.** 3,153
   rhythm, quality and landmark markers are counted separately.
7. **Manifest arithmetic reconciles**: `kept + dropped + excluded + unknown ==
   total annotations`, per record and overall.
8. **Re-running produces identical output** (array contents and all text artifacts).

### Class imbalance is measured, not corrected

`normalization.json` reports per-class counts and suggested inverse-frequency
weights **for the training split only**. No SMOTE, no undersampling. Resampling is
a training decision that belongs next to the loss function; doing it here would
destroy the true class distribution and make honest evaluation impossible.

---

## Results on the real database

48 records, 112,647 annotations, all at 360 Hz. Matches the published figures for
MIT-BIH Arrhythmia Database v1.0.0.

| | Count |
|---|---|
| Beats kept | 109,446 |
| Dropped at record boundary | 48 |
| Excluded non-beat annotations | 3,153 |
| Unknown symbols | 0 |
| **Total annotations** | **112,647** |

Beat annotations = 109,446 + 48 = **109,494**, the documented total.

Per AAMI class (default seeded record-wise split, 31/7/6 records):

| Class | All processed | Train split |
|-------|--------------:|------------:|
| N | 90,589 | 63,184 |
| S | 2,781 | 2,462 |
| V | 7,235 | 6,660 |
| F | 802 | 797 |
| Q | 8,039 | 13 |

Q collapses from 8,039 to 13 between the two columns because almost every Q beat
is a paced beat inside records 102/104/107/217, which AAMI excludes from the
splits. They are still processed and written to disk — excluded is not discarded.

---

## Baseline classifier

A classical baseline, to confirm the preprocessing output is learnable and to
give any future deep model a floor to beat. See [COMMANDS.md](COMMANDS.md) to run it.

Features follow PDF §2.9: six RR-interval features (derived from the saved
`r_peaks`, so the raw recordings are never re-read) plus the beat mean-pooled
from 300 samples to 30. Classifiers are Random Forest and LDA, each wrapped in an
sklearn `Pipeline` so the scaler cannot see test data even by accident.

**Results on DS2 (49,690 beats), Random Forest on RR + morphology:**

| Class | Support | Recall | Precision |
|-------|--------:|-------:|----------:|
| N | 44,196 | 0.904 | 0.959 |
| S | 1,836 | 0.271 | 0.200 |
| V | 3,219 | 0.844 | 0.594 |
| F | 388 | 0.273 | 0.120 |
| Q | 7 | 0.000 | 0.000 |

Macro recall 0.459. Accuracy 0.872, reported but uninformative.

### On the comparison with published work — read this before quoting it

de Chazal et al. (2004) report V sensitivity 0.803 and S sensitivity 0.760 on
this same DS1/DS2 split. **That is a reference range, not a like-for-like
benchmark, and our numbers are not a reproduction of their method.**

No code from that paper exists publicly — it predates code release as a norm —
so nothing of theirs is integrated here. What we borrowed is the DS1/DS2 record
lists in [`src/ecg/splits.py`](src/ecg/splits.py), i.e. which 44 patients go on
which side of the split. Everything else is our own.

The methods differ in ways that matter:

| | de Chazal 2004 | This project |
|---|---|---|
| ECG leads | two | **one** |
| Morphology features | fixed sampling windows around the beat | 300 samples mean-pooled to 30 |
| RR features | 4 | 6 (two ratio features added) |
| Classifier | LDA, their configuration | Random Forest |

So our V recall of 0.844 against their 0.803 is **not** evidence of a better
method. The defensible reading is narrower and is the one this baseline was
built for: V detection lands in the published range, which indicates the Phase 1
output carries the signal it should. Anything stronger than that is unsupported.

**Ventricular detection is at parity with published work. Supraventricular is
not** — and the reason is worth stating plainly, because it is *not* a data
problem. In the training split, S beats arrive at 0.501 s against 0.845 s for
normal beats, exactly the premature timing that defines the class, and **a single
threshold on `pre_rr_ratio` reaches S recall of 0.972 on DS2.** The signal is
there; this classifier is not exploiting it. The likely cause — 30 morphology
features drowning out 6 RR features under `max_features='sqrt'` — is recorded in
[PlanDoc/IMPLEMENTATION.md](PlanDoc/IMPLEMENTATION.md) as the next experiment.

### The leakage tripwire

Phase 1 z-scores beats using statistics fitted on *its* training records. Training
on a different split than the data was preprocessed for means the test beats were
scaled with statistics that already saw them. `train_baseline.py` refuses to run
in that case and names the command that fixes it, rather than trusting anyone to
remember.

---

## Later phases (planned, not implemented)

Phase 2 is MIMIC-III clinical preprocessing. It has a hard external dependency
worth knowing about early:

| Option | Access | Rough size | Note |
|--------|--------|-----------|------|
| MIMIC-III Clinical Database **Demo** v1.4 | Open, no credentialing | tens of MB | ~100 patients. The right choice for building and testing pipeline code. |
| MIMIC-III Clinical Database v1.4 (full) | **Credentialed**: PhysioNet account + CITI human-subjects training + signed data use agreement | several GB gzipped | Approval is a human process taking days to weeks. Apply early if Phase 2 is wanted. |
| MIMIC-III Waveform Matched Subset | Open | terabyte scale | Not viable on this machine. Out of scope. |

Account creation, CITI training and the data use agreement are yours to complete —
they are legal commitments that only you can make.

---

## Layout

```
src/ecg/
├── config.py      constants: fs, band edges, window sizes, class order, paths
├── errors.py      typed exceptions, so failures are diagnosable
├── io.py          WFDB loading, lead selection by name
├── filters.py     zero-phase bandpass, optional notch
├── normalize.py   z-score with separate fit/apply
├── labels.py      MIT-BIH symbol -> AAMI 5-class mapping
├── segment.py     beat-level and window-level segmentation
├── splits.py      record-wise and DS1/DS2 splits
├── acquire.py     download + integrity checking
└── pipeline.py    orchestration, manifest generation
scripts/
├── download_mitbih.py
└── preprocess_mitbih.py
PlanDoc/
├── PLAN.md              the approved plan
└── IMPLEMENTATION.md    what was actually done, deviations, test output
PERMISSIONS.md           permission policy for this project
```

## Data licence

MIT-BIH Arrhythmia Database v1.0.0 is openly available on PhysioNet — no account
or data use agreement required. It is de-identified clinical data, which is not
the same as public: it stays on this machine.
