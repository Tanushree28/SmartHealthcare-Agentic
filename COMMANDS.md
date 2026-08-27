# COMMANDS.md — copy-paste reference

Every command you need, in the order you'd normally use them.
All of these are safe to re-run.

---

## Start here, every time

```bash
cd "/Users/tanushreenepal/Desktop/Smart Healthcare"
source venv/bin/activate
```

Your prompt shows `(venv)` once it works. Run `deactivate` when you're done.

> Skipping this gives `ModuleNotFoundError` — plain `python` is your system
> Python, which doesn't have the packages. If you'd rather not activate, put
> `./venv/bin/` in front of every `python` below.

---

## 1. Check everything still works

```bash
python -m pytest
```
Expect `171 passed, 9 deselected` in a few seconds. Needs no data — it makes up
its own recordings.

```bash
python -m pytest -m requires_data
```
Expect `9 passed`. These check facts about the real database on disk.

```bash
python -m pytest -q                 # quieter
python -m pytest tests/test_io.py   # just one file
python -m pytest -k lead            # just tests matching "lead"
```

---

## 2. Get / check the data

```bash
python scripts/download_mitbih.py --verify
```
No internet needed. Expect `48/48 records verify.`

```bash
python scripts/download_mitbih.py
```
Downloads anything missing (~94 MB). Already-complete records are skipped, so
re-running costs nothing.

---

## 3. Preprocess

```bash
python scripts/preprocess_mitbih.py
```
Default seeded split. Writes to `~/physionet-data/processed/mitdb/`. Takes ~2 s.

```bash
python scripts/preprocess_mitbih.py \
  --strategy ds1_ds2 \
  --out-dir ~/physionet-data/processed/mitdb-ds12
```
The split used in published papers. **Required before training the baseline.**

```bash
python scripts/preprocess_mitbih.py --records 100 101 200   # a few records only
python scripts/preprocess_mitbih.py --include-paced         # keep records 102/104/107/217
python scripts/preprocess_mitbih.py --help                  # all options
```

---

## 4. Train the baseline classifier

```bash
python scripts/train_baseline.py \
  --processed-dir ~/physionet-data/processed/mitdb-ds12
```
Takes ~1 minute. Prints per-class recall, confusion matrices, and a summary
table; writes `baseline_metrics.json` next to the data.

```bash
# Just the main model, skip the ablations
python scripts/train_baseline.py \
  --processed-dir ~/physionet-data/processed/mitdb-ds12 \
  --classifiers random_forest --feature-sets rr+morphology
```

> If it refuses to run with a `NormalizationMismatchError`, that is the leakage
> tripwire doing its job: the beats on disk were scaled for a different split.
> The error message tells you which command to run. Follow it.

---

## 5. Rebuild the results page

The page published at GitHub Pages is generated from your processed data, so
its numbers can never drift from the pipeline.

```bash
python scripts/build_report.py
```

Writes `docs/index.html`. Preview it before pushing:

```bash
python -m http.server -d docs 8000
# then open http://localhost:8000
```

Re-run it after any change to the pipeline, the model, or the page templates
in `site/`. It refuses to run if the data it needs is missing, and names the
command that produces it.

---

## 6. Look at the results

```bash
open ~/physionet-data/processed/mitdb/manifest.csv          # per-record spreadsheet
cat  ~/physionet-data/processed/mitdb/splits.json           # who's in train/val/test
cat  ~/physionet-data/processed/mitdb-ds12/baseline_metrics.json
du -sh ~/physionet-data/*                                   # how much space it uses
```

---

## 7. Git

Nothing is committed yet. To see what changed:

```bash
git status
git diff
```

---

## Publishing to GitHub Pages

One-time setup, after the first push:

1. Push the repo (see Git below).
2. On GitHub: **Settings** &rarr; **Pages**.
3. Under *Build and deployment*, set **Source** to `Deploy from a branch`.
4. Set **Branch** to `master` and the folder to **`/docs`**. Save.
5. Wait about a minute. The page appears at:

   `https://tanushree28.github.io/SmartHealthcare-Agentic/`

After that, every push that includes a changed `docs/index.html` republishes
automatically. To update the page:

```bash
python scripts/build_report.py
git add docs/index.html
git commit -m "Rebuild results page"
git push
```

> The `docs/.nojekyll` file tells GitHub to serve the HTML as-is instead of
> running Jekyll over it. Leave it in place.

---

## If something breaks

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` | You skipped `source venv/bin/activate` |
| `DataRootError` | Data missing — run `python scripts/download_mitbih.py` |
| `NormalizationMismatchError` | Read the message; it names the exact command to run |
| Tests fail after you edit something | `git diff` to see what changed; `git checkout <file>` to undo |
| `build_report.py` says data is missing | It names the exact command to run first |
| Pages shows a 404 | Check Settings &rarr; Pages points at `master` + `/docs`; give it a minute |

Rebuild the environment from scratch if needed:

```bash
rm -rf venv
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Use a different data location:

```bash
export SMART_HEALTH_DATA=/Volumes/MyDrive/physionet-data
```
