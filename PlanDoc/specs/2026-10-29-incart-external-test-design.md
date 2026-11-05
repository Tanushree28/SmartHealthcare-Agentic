# External test on INCART — does the MIT-BIH beat model work on new patients?

**Status:** designed and approved 2026-10-29; built and run 2026-10-31.
Result: INCART headline macro-F1 0.570 (95% CI 0.482-0.646) against DS2 0.555,
so the model holds up; resampling does not matter (+0.0001). See
`results/2026-10-31-incart-external-test.md`.

**Where it fits:** step 2 of the open-data plan discussed on 2026-10-28/29
(tentative). That plan avoids credentialed data entirely:

| Step | Dataset | Question |
|---|---|---|
| 1 | MIT-BIH (done) | Classify each beat N/S/V/F/Q |
| **2** | **INCART (this document)** | **Does the model work on new patients and equipment?** |
| 3 | MIMIC-IV demo + MIMIC-IV-ECG | Real cases where the ECG and the chart belong to one person |
| 4 | CODE-15% | Does adding the ECG to age and sex predict death better? |

Steps 3 and 4 get their own designs. Full MIMIC-III / MIMIC-IV are dropped
from the plan because credentialing is not available.

---

## 1. Why

The user's view: a heartbeat is a heartbeat. A V beat from 1977 looks like one
from 2017, so a model trained on MIT-BIH should work on other patients. That is
largely right about the biology, but two things can still break the model:

- **New people.** Beat shapes differ from person to person and MIT-BIH has 47
  people. Even on MIT-BIH's own held-out patients (DS2), the selected ensemble
  scores macro-F1 0.555 (N/S/V/F), with S at F1 0.386 and F at 0.002.
- **New equipment.** Different recorder, lead, sampling rate and noise.

This step tests the claim on a dataset the model has never seen, with the same
kind of labels (cardiologist beat-by-beat annotations), so the numbers are
directly comparable with DS2.

## 2. The dataset

St Petersburg INCART 12-lead Arrhythmia Database v1.0.0 (PhysioNet `incartdb`):
75 recordings of 30 minutes from 32 Holter patients, 12 standard leads, 257 Hz,
over 175,000 reference beat annotations using the standard PhysioBank beat
codes. Open access, ODC Attribution License v1.0. About 800 MB uncompressed.

## 3. What gets built

### Steps

1. **Download and check.** All 75 records into `~/physionet-data/incartdb/`
   (the data root, next to `mitdb/`). A check confirms each record opens, is
   257 Hz, carries lead II, and has an `.atr` file. Idempotent, like
   `download_mitbih.py`.
2. **Load.** Lead II of each record, and its beat annotations. Reuse
   `mit_bih.io.load_record(..., expected_fs=257, preference=("II",))` and
   `mit_bih.io.load_annotations`; do not write a second WFDB reader.
3. **Map labels.** `mit_bih.labels.map_symbols` with `strict=True`. An
   unknown symbol stops the run; it is never guessed. Non-beat annotations are
   excluded exactly as in MIT-BIH. The symbol counts (beat / excluded) are
   written to the results. A record containing paced beats (`/` or `f`) is
   left out and listed, as MIT-BIH's paced records are left out of DS2.
4. **Resample 257 → 360 Hz.** The networks reject any other rate, so the raw
   lead II signal is resampled (`scipy.signal.resample_poly(x, 360, 257)`)
   before it reaches `mit_bih.inference`, which then band-passes it at 360 Hz
   as usual. Annotation positions move with it: `round(sample * 360 / 257)`.
   Positions that land outside the resampled signal are dropped and counted.
5. **Score.** The unchanged `D_all_four` ensemble (20 checkpoints) through
   `mit_bih.inference.EcgClassifier.from_bundle()` and `prepare_beats` with the
   annotated positions. Predicted class = argmax, exactly as the DS2 evaluation
   (`scripts/mit_bih/evaluate_bundle.py`). Beat counts must reconcile:
   classified + dropped at the edges = annotated beats.
6. **Report.** Per-class recall / precision / F1, confusion matrix, headline
   macro-F1 (N/S/V/F, Q excluded as everywhere else), and the same numbers per
   record. Side by side with DS2.
7. **Resampling check.** The 22 DS2 records (paced records excluded, as in the
   published evaluation), read from the raw WFDB files and prepared through
   `mit_bih.inference` as `evaluate_bundle.py --from-raw` does, scored twice:
   - **original**: 360 Hz as recorded — must reproduce macro-F1 0.5554;
   - **round trip**: signal 360 → 257 → 360 Hz, and annotation positions
     rounded onto the 257 Hz grid and back, mirroring what INCART goes through.

### Fixed before the run (so the result cannot steer the rule)

- **"The model holds up on INCART"** means the DS2 headline macro-F1 (original,
  re-run in step 7) falls inside INCART's 95% interval (§3, intervals).
  Otherwise the write-up says it dropped (or rose), by how much, and in which
  classes.
- **"Resampling matters"** means the round-trip DS2 headline macro-F1 differs
  from the original by more than 0.01. Expectation, stated now: it will not
  matter much, because the band-pass keeps only 0.5–40 Hz and 257 Hz carries
  everything up to 128 Hz.
- INCART is scored **once**. Nothing is tuned on it: no thresholds, no member
  choice, no preprocessing change chosen after seeing INCART numbers.

### Beat-position check (runs before scoring, no model needed)

The networks were trained on beats centred on MIT-BIH's annotated positions.
If INCART's marks sit at a different point of the QRS, every window is
off-centre and the score drops for a reason unrelated to the model.

Measure, for each annotated beat, the signed offset from the mark to the largest
absolute band-passed value within ±50 ms — the same refinement
`inference.detect_r_peaks` uses. Do it for INCART (after resampling) and for
DS2 (at 360 Hz as recorded). Report the median and interquartile range for
both, in milliseconds.

Report alongside it the median band-passed amplitude at the mark, in mV, for
both datasets. The networks use MIT-BIH's global z-score statistics, so a
recorder or lead that records larger or smaller beats shifts every input. This
is a diagnostic only; nothing is rescaled.

**Rule:** if the two medians differ by more than 3 samples at 360 Hz (8.3 ms),
also score a second INCART variant with every mark moved to that local maximum,
and report both. The as-marked result stays the headline.

### Intervals

95% percentile bootstrap, 1,000 draws, seed `config.PHASE_1D_SEED`,
resampling **patients** (all of a patient's beats together), because beats
from one patient are not independent. INCART's record-to-patient mapping is
taken from the downloaded files if present (headers or a documentation file).
If no mapping exists, resample records and say in the results that this
understates the uncertainty, since some records share a patient. DS2 is
resampled by record (22 records, 22 patients).

## 4. Where things go

```
src/incart/                    new module, INCART only
├── __init__.py
├── config.py                  record names, 257 Hz, lead II, data paths
├── io.py                      lead II + beat labels per record (wraps mit_bih.io)
├── resample.py                257 -> 360 Hz; positions move with the signal
└── evaluate.py                scoring, per-record table, patient bootstrap, position check
scripts/incart/
├── download_incart.py         download + check all 75 records
├── evaluate_on_incart.py      steps 2-6 and the position check
└── resampling_check.py        step 7
tests/incart/                  synthetic INCART-style records; one real-data test
results/
├── YYYY-MM-DD-incart-external-test.json     INCART numbers (aggregates only)
├── YYYY-MM-DD-incart-resampling-check.json  DS2 original vs round trip
└── YYYY-MM-DD-incart-external-test.md       the write-up, both runs
~/physionet-data/incartdb/                   raw records (outside the repo)
~/physionet-data/processed/incartdb/         per-beat predictions (.npz) + run JSONs
```

- **Dependencies run one way.** `incart` imports `mit_bih` and `common`;
  `mit_bih` never imports `incart`. Nothing in `src/mit_bih/` changes, and the
  inference bundle is not re-exported.
- **Results naming.** Dates are run dates, per `results/README.md`. The
  dataset name leads (`incart-…`) so the files sort together.
- **Per-beat outputs stay in the data root.** The repository gets aggregate
  numbers only, as with every other result.

## 5. Tests

Written first, each failing before its code exists (project convention):

- resampling keeps a synthetic spike at `round(pos * 360 / 257)` ± 1 sample,
  and keeps signal length at `round(n * 360 / 257)` ± 1;
- annotation positions move with the signal; out-of-range ones are dropped and
  counted;
- loading picks lead II by name from a synthetic 12-lead WFDB record, and
  rejects a record at another rate;
- an unknown annotation symbol stops the run;
- beat counts reconcile end to end on a synthetic record;
- patient bootstrap resamples whole patients (a fixture where one patient is
  all-wrong widens the interval as expected);
- the position check returns 0 ms for marks placed exactly on the peaks of a
  synthetic signal, and the shift for marks placed off them;
- each new script starts (`--help`), added to `tests/test_scripts.py`;
- `requires_data` test: one real INCART record loads at 257 Hz with lead II
  and maps with no unknown symbols.

## 6. Constraints

- **PERMISSIONS.md.** Download (network) and the long runs need the user's OK;
  both are asked for together with this document. No new dependencies:
  `scipy.signal.resample_poly` (scipy 1.16.2) and `wfdb` 4.3.0 are already
  pinned. No commits without the user's OK.
- **Patient data stays local.** Scripts print only aggregate numbers (counts,
  metrics, offsets in summary). No signal values, per-beat rows or header
  contents are printed or read into an AI conversation.
- **Run time.** DS2 took 7 min 24 s for 49,646 beats on this Mac (2026-10-14).
  INCART's ~175k beats: about 26 min. Resampling check (DS2 twice): about
  15 min. A second INCART variant, if the position rule fires: another ~26 min.
- **Not clinical validation.** The write-up keeps the project's standing
  caveat: research results, not a medical device. Scores are uncalibrated
  (Phase 1f) and the outputs say so.

## 7. Non-goals and anti-goals

- **Not doing:** retraining or fine-tuning on INCART; leads other than II;
  showing INCART records in the agents or web page; calibration.
- **Must not happen:** INCART beats leaking into any training or selection; an
  INCART result changing any MIT-BIH choice after the fact; a drop explained
  away without the resampling check and the position check to back it.

## 8. Left to the builder

Module internals and function names; how progress is printed; batching for
memory; the exact JSON layout (aggregates only, with a provenance block via
`common.provenance`, like the existing results); figures, if any.

## 9. Calls made by the author, not by the user

- Lead II only (closest to MIT-BIH's MLII).
- The two decision rules in §3 and the 8.3 ms position threshold.
- Patient-level bootstrap with 1,000 draws; record-level fallback.
- Three results files rather than one, because two scripts produce them.
- The DS2 "original" run is repeated (not read from the published file) so the
  intervals for DS2 and INCART are computed the same way, and so the pipeline
  is shown to still reproduce 0.5554.
- Records with paced beats are left out, mirroring the DS2 protocol.
- Amplitude is reported, not corrected: correcting it would be a
  preprocessing change chosen after looking at INCART.
