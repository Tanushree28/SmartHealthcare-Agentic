"""MIMIC-III clinical data: ingestion, cohort, features, preprocessing, training
and evaluation (Phase 2).

- ``config``: the prediction task (cohort, index time, windows, target,
  horizon), the clinical variables and their ITEMIDs, data locations.
- ``ingestion``: schema-checked, case-insensitive table readers; chunked
  reading of CHARTEVENTS/LABEVENTS for the full database.
- ``cohort``: cohort selection with validated joins and a counted flow.
- ``features``: time-window vitals and labs, unit handling, missingness flags.
- ``dataset``: patient-level (SUBJECT_ID) train/validation/test splits.
- ``preprocessing``: imputation, scaling and encoding inside one pipeline, so
  they are fitted on training patients only.
- ``train`` / ``evaluate``: model selection on validation, a single test
  evaluation, saved model and report.

**Data.** Built and tested on synthetic tables (``tests/mimic_iii``) and run
on the open MIMIC-III Demo. The Demo (100 patients, all of whom had died by the
time PhysioNet selected them) verifies the pipeline; it cannot estimate
clinical performance. Full MIMIC-III requires PhysioNet credentialing, which
is pending.

``mimic_iii`` never imports ``mit_bih``.
"""
