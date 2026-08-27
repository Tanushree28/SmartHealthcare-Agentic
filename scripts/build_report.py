#!/usr/bin/env python3
"""Regenerate the published results page from the data actually on disk.

    python scripts/build_report.py

Writes ``docs/index.html``, which GitHub Pages serves. Every figure on that
page is computed here from the processed output, so the page cannot drift away
from the pipeline that produced it -- re-run the pipeline, re-run this, and the
numbers move together.

The templates live in ``site/``. This script only supplies data and assembles
the document.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg import config, features, filters, io  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DOCS = ROOT / "docs"

#: The page is written as fragments and wrapped here, so non-ASCII must be
#: escaped: keeping the output 7-bit means no encoding surprise can survive a
#: mis-served charset. HTML fragments take entities; the script takes \uXXXX.
ENTITIES = {
    "·": "&#183;", "–": "&#8211;", "—": "&#8212;",
    "²": "&#178;", "³": "&#179;", "÷": "&#247;",
    "⁴": "&#8308;", "⁵": "&#8309;", "→": "&#8594;",
    "…": "&#8230;", "’": "&#8217;", "“": "&#8220;",
    "”": "&#8221;", "×": "&#215;", "≥": "&#8805;",
    "±": "&#177;",
}

DOC_TITLE = "MIT-BIH Beat Pipeline"
DOC_DESC = ("A visual walkthrough of the MIT-BIH ECG preprocessing pipeline and the "
            "baseline classifier's results, computed from the real data.")
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
         "&family=IBM+Plex+Sans:wght@400;500;600"
         "&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap")


def escape_html(text: str) -> str:
    for raw, entity in ENTITIES.items():
        text = text.replace(raw, entity)
    return text


def escape_js(text: str) -> str:
    return "".join(c if ord(c) < 128 else "\\u%04x" % ord(c) for c in text)


def collect(processed_ds12: Path, processed_default: Path) -> dict:
    """Pull every number the page displays out of the pipeline's own output."""
    import pandas as pd

    for path, what in ((processed_ds12, "--ds12-dir"), (processed_default, "--processed-dir")):
        if not (path / "splits.json").exists():
            raise SystemExit(
                f"no pipeline output at {path}.\n"
                f"Run:  python scripts/preprocess_mitbih.py "
                f"{'--strategy ds1_ds2 --out-dir ' + str(path) if what == '--ds12-dir' else ''}"
            )

    metrics_path = processed_ds12 / "baseline_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(
            f"no baseline_metrics.json at {processed_ds12}.\n"
            f"Run:  python scripts/train_baseline.py --processed-dir {processed_ds12}"
        )

    out: dict = {}
    metrics = json.loads(metrics_path.read_text())
    out["results"] = [
        {"label": r["label"], "macro": r["macro_recall"], "acc": r["accuracy"],
         "per_class": r["per_class"], "cm": r["confusion_matrix"]}
        for r in metrics["results"]
    ]

    split = json.loads((processed_ds12 / "splits.json").read_text())
    out["split_sizes"] = {k: len(split[k]) for k in ("train", "val", "test", "excluded")}
    out["split_records"] = {k: split[k] for k in ("train", "val", "test", "excluded")}

    # Mean beat shape per class, over the training records only.
    sums = {i: np.zeros(config.BEAT_LENGTH) for i in range(5)}
    counts = {i: 0 for i in range(5)}
    for record in split["train"]:
        with np.load(processed_ds12 / "beats" / f"{record}.npz") as data:
            X, y = data["X"], data["y"]
        for cls in range(5):
            mask = y == cls
            if mask.any():
                sums[cls] += X[mask].sum(axis=0)
                counts[cls] += int(mask.sum())
    out["mean_beats"] = {config.AAMI_CLASSES[c]: (sums[c] / counts[c]).round(3).tolist()
                         for c in range(5) if counts[c]}
    out["beat_counts_train"] = {config.AAMI_CLASSES[c]: counts[c] for c in range(5)}

    # The RR-ratio distribution -- the page's central finding.
    feats = features.extract_records(processed_ds12 / "beats", split["train"])
    column = feats.X[:, list(feats.feature_names).index("pre_rr_ratio")]
    bins = np.linspace(0.2, 1.8, 41)
    out["rr_hist"] = {"bins": bins.round(3).tolist()}
    for cls, name in enumerate(config.AAMI_CLASSES):
        values = column[feats.y == cls]
        if values.size:
            out["rr_hist"][name] = np.histogram(values, bins=bins)[0].tolist()

    # Raw vs filtered, five seconds of one record.
    record = io.load_record(config.mitdb_dir(), "100")
    window = slice(0, config.FS * 5)
    out["denoise"] = {
        "raw": record.signal[window].round(3).tolist(),
        "filtered": filters.denoise(record.signal)[window].round(3).tolist(),
        "fs": config.FS,
    }

    manifest = pd.read_csv(processed_default / "manifest.csv", dtype={"record": str})
    out["pipeline_totals"] = {
        "annotations": int(manifest.n_annotations.sum()),
        "kept": int(manifest.n_beats_kept.sum()),
        "boundary": int(manifest.n_dropped_boundary.sum()),
        "nonbeat": int(manifest.n_excluded_nonbeat.sum()),
        "unknown": int(manifest.n_unknown.sum()),
        "records": len(manifest),
    }
    out["class_all"] = {c: int(manifest[f"n_class_{c}"].sum()) for c in config.AAMI_CLASSES}
    return out


def assemble(payload: dict, head: str, body: str, script: str) -> str:
    """Wrap the fragments in a complete, self-contained HTML document."""
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{DOC_TITLE}</title>
<meta name="description" content="{DOC_DESC}">
<meta name="color-scheme" content="light dark">
<meta property="og:title" content="{DOC_TITLE}">
<meta property="og:description" content="{DOC_DESC}">
<meta property="og:type" content="article">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='13'>&#129728;</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>
{escape_html(head)}
</head>
<body>
{escape_html(body)}
<script type="application/json" id="viz-data">{blob}</script>
{escape_js(script)}
</body>
</html>
"""
    leftover = sorted({c for c in doc if ord(c) > 127})
    if leftover:
        raise SystemExit(f"non-ASCII survived escaping: {leftover!r}")
    return doc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds12-dir", type=Path,
                        default=config.data_root() / "processed" / "mitdb-ds12",
                        help="output of the ds1_ds2 preprocessing run")
    parser.add_argument("--processed-dir", type=Path, default=None,
                        help="output of the default preprocessing run")
    parser.add_argument("--out", type=Path, default=DOCS / "index.html")
    args = parser.parse_args(argv)

    processed_default = (args.processed_dir or config.processed_dir()).expanduser()
    payload = collect(args.ds12_dir.expanduser(), processed_default)

    doc = assemble(
        payload,
        (SITE / "head.html").read_text(encoding="utf-8"),
        (SITE / "body.html").read_text(encoding="utf-8"),
        (SITE / "script.html").read_text(encoding="utf-8"),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(doc, encoding="utf-8")
    # Tell GitHub Pages not to run Jekyll over this directory.
    (args.out.parent / ".nojekyll").touch()

    print(f"Wrote {args.out}  ({args.out.stat().st_size/1024:.0f} KB)")
    print(f"  {payload['pipeline_totals']['kept']:,} beats, "
          f"{len(payload['results'])} model results")
    print(f"\nPreview locally:  python -m http.server -d {args.out.parent} 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
