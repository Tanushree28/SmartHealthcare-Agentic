"""Provenance for every artifact: what code, what libraries, what inputs.

An artifact without provenance cannot be audited. Every model, report and
bundle this project writes records the git commit, whether the working tree was
dirty, the library versions, and the SHA-256 of the files it was built from.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .paths import PROJECT_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path | str, chunk_bytes: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks so large files fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root: Path = PROJECT_ROOT) -> dict:
    """Current commit and whether uncommitted changes exist.

    Returns ``None`` values rather than raising when git is unavailable: a
    missing commit hash should be visible in the artifact, not fatal.
    """

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True,
                timeout=10, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def environment() -> dict:
    """Python and the library versions that affect numbers."""
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("pandas", "sklearn", "scipy", "torch"):
        module = sys.modules.get(name)
        if module is None:
            try:
                module = __import__(name)
            except ImportError:
                continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def provenance(inputs: dict[str, Path | str] | None = None) -> dict:
    """The standard provenance block: time, code, environment, input hashes."""
    block = {
        "generated_utc": utc_now(),
        "git": git_state(),
        "environment": environment(),
    }
    if inputs:
        block["inputs"] = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        }
    return block


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        # JSON has no NaN. An undefined metric is written as null, never as a
        # number that looks real.
        return None if not math.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = sorted(value) if isinstance(value, set) else value
        return [_jsonable(v) for v in items]
    return value


def write_json(path: Path | str, payload: dict) -> Path:
    """Write ``payload`` as stable, sorted JSON. NaN and inf become null."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    return path
