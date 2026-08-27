"""Central configuration for the MIT-BIH preprocessing pipeline.

Every numeric constant the PDF specifies lives here and nowhere else, so
changing a window size is a one-line edit rather than a search-and-replace
across modules.

Paths are resolved from the ``SMART_HEALTH_DATA`` environment variable so no
absolute path is baked into source. See PlanDoc/PLAN.md step 0.2.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Signal acquisition (PDF 2.1)
# ---------------------------------------------------------------------------

#: MIT-BIH Arrhythmia Database sampling rate. Records at any other rate are
#: rejected rather than filtered with coefficients designed for this one.
FS: int = 360

#: Lead preference order. Most MIT-BIH records use MLII on channel 0, but not
#: all do -- selecting by index would silently mix leads across the dataset.
LEAD_PREFERENCE: tuple[str, ...] = ("MLII", "II", "V5", "V1", "V2", "V4")

# ---------------------------------------------------------------------------
# Denoising (PDF 2.2)
# ---------------------------------------------------------------------------

BANDPASS_LOW_HZ: float = 0.5
BANDPASS_HIGH_HZ: float = 40.0
BANDPASS_ORDER: int = 3

#: Powerline frequency for the optional notch. Disabled by default: with a
#: 40 Hz low-pass already in place a 60 Hz notch is a no-op. It becomes
#: meaningful only if the passband is widened past 50 Hz.
NOTCH_FREQ_HZ: float = 60.0
NOTCH_QUALITY: float = 30.0
NOTCH_ENABLED_BY_DEFAULT: bool = False

# ---------------------------------------------------------------------------
# Normalization (PDF 2.3, 3.9)
# ---------------------------------------------------------------------------

#: Added to the standard deviation so a flatline segment yields zeros, not NaN.
NORMALIZE_EPS: float = 1e-8

# ---------------------------------------------------------------------------
# Segmentation (PDF 2.4, 2.5)
# ---------------------------------------------------------------------------

SAMPLES_BEFORE: int = 100
SAMPLES_AFTER: int = 200
BEAT_LENGTH: int = SAMPLES_BEFORE + SAMPLES_AFTER  # 300 samples ~ 0.83 s @ 360 Hz

#: Fixed-duration windows for later rhythm-context models.
WINDOW_SECONDS: tuple[int, ...] = (5, 10)

# ---------------------------------------------------------------------------
# Labels (PDF 2.6)
# ---------------------------------------------------------------------------

#: AAMI class order. This order is a contract: reordering it would silently
#: invalidate every saved label array and any model trained against them.
AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")

# ---------------------------------------------------------------------------
# Splits (PDF 2.8)
# ---------------------------------------------------------------------------

#: Records with paced beats. Excluded by default per AAMI recommendation --
#: paced morphology is not comparable to the other classes.
PACED_RECORDS: tuple[str, ...] = ("102", "104", "107", "217")

#: All 48 records in MIT-BIH Arrhythmia Database v1.0.0.
ALL_RECORDS: tuple[str, ...] = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
)

DEFAULT_SPLIT_SEED: int = 20260826
DEFAULT_SPLIT_FRACTIONS: tuple[float, float, float] = (0.7, 0.15, 0.15)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT_ENV_VAR = "SMART_HEALTH_DATA"
DEFAULT_DATA_ROOT = Path.home() / "physionet-data"

MITDB_DIRNAME = "mitdb"
PROCESSED_DIRNAME = "processed"


def data_root() -> Path:
    """Root directory for raw and processed data, outside the project tree."""
    raw = os.environ.get(DATA_ROOT_ENV_VAR)
    return Path(raw).expanduser() if raw else DEFAULT_DATA_ROOT


def mitdb_dir() -> Path:
    """Directory holding the raw MIT-BIH WFDB records."""
    return data_root() / MITDB_DIRNAME


def processed_dir() -> Path:
    """Directory holding pipeline outputs."""
    return data_root() / PROCESSED_DIRNAME / MITDB_DIRNAME
