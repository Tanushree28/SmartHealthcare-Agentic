# End-to-end run (verification evidence)

Command: `./venv/bin/python scripts/run_end_to_end.py`  
Run at 2026-10-15T04:43:02+00:00 UTC, git commit 156d81a with uncommitted changes.

| Step | Status | Exit code | Seconds |
|---|---|---:|---:|
| 1. validate setup | OK | 0 | 1.4 |
| 2. MIMIC-III features | OK | 0 | 1.6 |
| 3. MIMIC-III patient split | OK | 0 | 0.2 |
| 4. MIMIC-III train + evaluate | OK | 0 | 2.5 |
| 5. MIT-BIH beats + DS1/DS2 split | OK | 0 | 2.0 |
| 6. MIT-BIH ECG model evaluation | OK | 0 | 58.7 |
| 7. shared inference smoke test | OK | 0 | 2.6 |
| 8. fusion training | BLOCKED | 3 | 1.4 |

Total: 70 s. Step 8 is expected to report BLOCKED (exit code 3): no paired ECG + clinical data exists.

Step 6 here scores 3 DS2 records (100, 208, 232) as a quick check. The full-DS2 evaluation is in
`results/2026-10-14-bundle-eval-ds1_ds2.json` and `results/2026-10-14-bundle-eval-raw-ds1_ds2.json`.

Identifiers of MIMIC-III Demo ICU stays are deliberately left out of this file.
