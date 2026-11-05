"""Where data and project files live.

Raw and processed data stay **outside** the repository, under the directory
named by ``SMART_HEALTH_DATA`` (default ``~/physionet-data``). Nothing in
``src/`` hard-codes a machine path.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV_VAR = "SMART_HEALTH_DATA"
DEFAULT_DATA_ROOT = Path.home() / "physionet-data"

#: The repository root: ``src/common/paths.py`` -> two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Root directory for raw and processed data, outside the project tree."""
    raw = os.environ.get(DATA_ROOT_ENV_VAR)
    return Path(raw).expanduser() if raw else DEFAULT_DATA_ROOT


def processed_root() -> Path:
    """Parent of every module's processed output."""
    return data_root() / "processed"


def relative_to_project(path: Path | str) -> str:
    """``path`` relative to the project root when inside it, else absolute.

    Used when recording artifact locations, so committed metadata does not
    carry one machine's home directory.
    """
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def resolve_project_path(path: Path | str) -> Path:
    """Inverse of :func:`relative_to_project`."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
