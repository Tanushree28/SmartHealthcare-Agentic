# Master Implementation Prompt: MIT-BIH + MIMIC-III Healthcare AI

You are a senior machine learning engineer, clinical AI researcher, and software architect working inside my **existing repository**. Your assignment is to organize and complete the implementation of the MIT-BIH ECG pipeline, MIMIC-III clinical pipeline, and their integration. **Implement code and run tests; do not stop after producing a plan.**

## 1. Inspect before changing anything

- Read the repository tree, README, `COMMANDS.md`, `PlanDoc/specs/`, configs, tests, notebooks, and existing source code where present.
- Identify completed, partial, broken, and missing work. Preserve existing working functionality.
- Check which datasets, dependencies, compute resources, and permissions are actually available. Do not claim access to missing data.
- Create a short implementation checklist and begin executing it.

## 2. Organize the existing repository

Keep this as **one project**, with three separate source modules. Adapt to the existing layout rather than forcing a disruptive rewrite:

```text
src/
  mit_bih/                  # ECG processing and arrhythmia classification
    preprocessing.py
    dataset.py
    model.py
    train.py
    evaluate.py
  mimic_iii/                # Clinical data processing and prediction
    ingestion.py
    features.py
    preprocessing.py
    dataset.py
    train.py
    evaluate.py
  multimodal/               # Shared inference and fusion architecture
    ecg_encoder.py
    clinical_encoder.py
    fusion.py
    inference.py
  common/                   # Shared configuration, metrics, utilities
tests/
configs/
scripts/
PlanDoc/
```

- Create only the files that are actually needed; reuse existing implementations.
- Move code only when safe, and update imports, paths, commands, and tests.
- Keep restricted datasets and large artifacts out of version control.
- Explain each module and how they connect in the README.

## 3. Finish the MIMIC-III pipeline

Implement and test:

1. Memory-safe, chunked ingestion of relevant tables, including `PATIENTS`, `ADMISSIONS`, `ICUSTAYS`, `CHARTEVENTS`, `LABEVENTS`, and diagnosis tables as appropriate.
2. A clearly specified prediction target, index time, observation window, and prediction horizon.
3. Validated joins, unit handling, time-window features, vital signs, labs, demographics, and clinically justified categories.
4. Missing-value imputation and missingness flags.
5. Patient-level train/validation/test splits by `SUBJECT_ID`.
6. Imputation, scaling, and encoding fitted **only on training data**.
7. Tests against patient overlap, temporal leakage, wrong joins, incorrect windows, and malformed inputs.
8. Reusable feature tables, preprocessing artifacts, model checkpoints, and evaluation reports.
9. Memory-safe processing of full MIMIC-III data when authorized data and resources are available.

Use small synthetic MIMIC-like fixtures for automated tests. Never present synthetic test metrics as real clinical performance.

## 4. Finish the MIT-BIH ECG pipeline

Implement or verify:

1. WFDB signal and annotation loading.
2. Signal preprocessing, beat segmentation, and label mapping (document the chosen AAMI-style mapping if used).
3. Patient/record-grouped train/validation/test splitting without beat-level leakage.
4. A configurable ECG classifier (reuse an existing working model where possible).
5. Training, early stopping, checkpoints, reproducible seeds, and class imbalance handling.
6. Evaluation using confusion matrix, per-class recall, macro-F1, precision, and other appropriate metrics.
7. Saved inference preprocessing and model artifacts.
8. Unit tests and an ECG inference smoke test.

Train and evaluate on real data only when it is available and authorized.

## 5. Integrate MIT-BIH and MIMIC-III responsibly

These datasets do **not** inherently contain the same patients. **Never fabricate patient matches or pair unrelated records as if they belong to one patient.**

Implement in stages:

- **Independent baselines:** Working ECG-only and clinical-only models with their own valid evaluation.
- **Shared inference interface:** Accept ECG data, clinical data, or both, validate inputs, run the appropriate encoders/models, and return clearly labeled outputs.
- **Fusion architecture:** Implement and unit-test feature concatenation or another simple fusion module with synthetic paired fixtures.
- **Real multimodal training:** Perform this only if a genuinely paired, authorized ECG + clinical dataset with a consistent prediction target and valid temporal alignment is available. Otherwise mark training and clinical validation of fusion as **BLOCKED**, not complete.

Do not claim that independently trained probabilities are a clinically validated combined risk score.

## 6. Build an executable end-to-end workflow

Provide working commands/scripts to:

1. Validate available datasets and configuration.
2. Prepare MIMIC-III features.
3. Prepare MIT-BIH ECG samples.
4. Create leakage-safe splits.
5. Train and evaluate each independent model.
6. Build and smoke-test the shared inference interface.
7. Train and evaluate fusion only if valid paired data exists.
8. Save artifacts and produce a results report.

Keep resource-intensive training configurable; do not silently launch unbounded jobs. Run feasible smoke tests and actual experiments within the available environment.

## 7. Testing and documentation

- Add unit and integration tests for ingestion, preprocessing, feature extraction, splitting, leakage, model forward passes, serialization, fusion shapes, and missing modalities.
- Run tests, diagnose failures, and fix them.
- Update the README with setup, folder responsibilities, datasets, execution commands, architecture, target definitions, limitations, and reproducibility instructions.
- Keep `PlanDoc/` updated with decisions, progress, and actual experiment results.
- Do not fabricate test outcomes, training results, files, or model checkpoints.

## 8. Execution rules

- **Do not stop at planning or folder creation. Continue through implementation and validation.**
- Preserve working code and avoid unnecessary refactoring.
- Make reasonable decisions independently, but clearly document important assumptions.
- Respect data access restrictions and avoid committing patient data or secrets.
- If a task is blocked by missing datasets, permissions, dependencies, or hardware, complete all verifiable work around it and report the blocker precisely.
- Run relevant tests after each major change.
- Keep going until all feasible work is implemented and verified.

## 9. Final completion report

Produce a table with **Component | Status | Tests executed | Evidence/artifact path | Remaining blocker** for:

- Repository organization
- MIMIC ingestion and features
- MIMIC clinical model
- MIT-BIH preprocessing
- MIT-BIH ECG model
- Multimodal architecture
- Shared inference
- Automated tests
- Documentation

Use only **COMPLETE**, **PARTIAL**, **BLOCKED**, or **NOT STARTED**. Include actual commands, observed outputs, and verified artifact locations. Clearly separate *software integration complete* from *multimodal model clinically trained and evaluated*.

**Start now by inspecting the repository. Then implement, test, fix, and document everything feasible without waiting for repeated approval.**
