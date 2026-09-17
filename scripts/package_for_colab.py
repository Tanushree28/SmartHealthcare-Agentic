#!/usr/bin/env python
"""Bundle the code -- and only the code -- for a Colab session.

The data is deliberately **not** included. MIT-BIH is public and
``download_mitbih.py`` already fetches and verifies it, so Colab rebuilds the
126 MB of processed beats from source in about a minute. That keeps this archive
around 100 KB, makes each session a genuine from-scratch reproduction, and
removes any chance of training against a stale copy of the beats.

Usage:
    ./venv/bin/python scripts/package_for_colab.py
    ./venv/bin/python scripts/package_for_colab.py --out ~/Desktop/phase1d.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Everything Colab needs to run the study and prove the code is sound first.
#:
#: ``site`` holds the report templates. It is here because ``tests/test_report.py``
#: reads them: without it the suite reports 10 errors in Colab, and a test step
#: that always shows errors teaches you to ignore the test step.
INCLUDE_DIRS = ("src", "scripts", "tests", "site")
INCLUDE_FILES = ("pytest.ini", "requirements.txt")

#: torch is excluded on purpose. Colab ships a CUDA build of torch already;
#: installing our pinned CPU-oriented version over it risks replacing a working
#: GPU build with one that cannot see the GPU. The version actually used is
#: recorded in the run provenance instead.
COLAB_REQUIREMENTS = """\
# Phase 1d dependencies for Colab.
#
# torch is NOT pinned here: Colab provides a CUDA build, and overwriting it
# would risk losing GPU support. train_nn.py records whatever version is
# present in its provenance block, so the run stays auditable.
numpy>=2.0
scipy>=1.13
pandas>=2.0
wfdb==4.3.0
scikit-learn==1.7.2
pytest==8.4.2

# Phase 1f: src/ecg/plots.py imports matplotlib directly, and tests/test_plots.py
# runs in the notebook's test gate. Colab already ships a usable matplotlib, so
# this is normally a no-op -- it is listed so the gate cannot fail on a runtime
# that does not.
matplotlib>=3.8
"""

SKIP_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store"}


def _wanted(path: Path) -> bool:
    return not any(part in SKIP_PARTS for part in path.parts) and path.is_file()


def build(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in INCLUDE_DIRS:
            for path in sorted((ROOT / directory).rglob("*")):
                if _wanted(path):
                    archive.write(path, path.relative_to(ROOT))
                    written += 1

        for name in INCLUDE_FILES:
            path = ROOT / name
            if path.exists():
                archive.write(path, name)
                written += 1

        archive.writestr("requirements-colab.txt", COLAB_REQUIREMENTS)
        written += 1

    print(f"wrote {out_path}  ({written} files, {out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(ROOT / "phase1d-colab.zip"),
        help="where to write the archive",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out).expanduser()
    build(out_path)

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    latest = notebooks[-1].name if notebooks else "the phase notebook"

    print(
        "\nNext:\n"
        f"  1. open notebooks/{latest} in Google Colab\n"
        "     (or whichever phase you are running -- there are "
        f"{len(notebooks)} in notebooks/)\n"
        "  2. Runtime -> Change runtime type -> GPU\n"
        f"  3. upload {out_path.name} when the notebook asks\n"
        "  4. run all cells\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
