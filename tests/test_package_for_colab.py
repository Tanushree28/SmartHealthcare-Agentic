"""The Colab archive must be able to run the suite it ships.

A packaging bug is invisible locally -- everything works here -- and only shows
up as errors in a Colab session, where the natural reaction is to ignore them.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import package_for_colab  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archive(tmp_path):
    path = package_for_colab.build(tmp_path / "pkg.zip")
    with zipfile.ZipFile(path) as handle:
        return set(handle.namelist())


def test_every_test_module_is_included(archive):
    """If a test ships, whatever it reads must ship too."""
    local = {f"tests/{p.name}" for p in (ROOT / "tests").glob("test_*.py")}

    assert local <= archive


def test_the_report_templates_are_included(archive):
    """tests/test_report.py reads these; without them the suite errors."""
    assert {"site/head.html", "site/body.html", "site/script.html"} <= archive


def test_conftest_is_included(archive):
    assert "tests/conftest.py" in archive


def test_pytest_config_is_included(archive):
    """pythonpath=src lives here; without it nothing imports."""
    assert "pytest.ini" in archive


def test_the_scripts_colab_runs_are_included(archive):
    assert {
        "scripts/download_mitbih.py",
        "scripts/preprocess_mitbih.py",
        "scripts/train_nn.py",
    } <= archive


def test_colab_requirements_omit_torch(archive, tmp_path):
    """Colab ships a CUDA torch; overwriting it can cost GPU support.

    Only real requirement lines are checked -- the file's comments explain the
    exclusion and naturally contain the word "torch".
    """
    assert "requirements-colab.txt" in archive

    path = package_for_colab.build(tmp_path / "again.zip")
    with zipfile.ZipFile(path) as handle:
        text = handle.read("requirements-colab.txt").decode()

    requirements = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert requirements, "requirements-colab.txt has no actual requirements"
    assert not [r for r in requirements if r.lower().startswith("torch")]
    assert any(r.startswith("wfdb") for r in requirements)


def test_no_data_or_caches_are_shipped(archive):
    assert not [n for n in archive if "__pycache__" in n]
    assert not [n for n in archive if n.endswith(".npz")]
