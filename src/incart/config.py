"""INCART constants and the thresholds fixed before the run.

The decision thresholds are here, in code, so the rules applied to the results
are the rules written in the design before any INCART number existed.
"""

from __future__ import annotations

from pathlib import Path

from common import paths
from mit_bih import config as mit_config

#: INCART's sampling rate. The networks were trained at ``MODEL_FS``.
FS: int = 257
MODEL_FS: int = mit_config.FS

#: The one lead scored: the closest of the 12 to MIT-BIH's MLII.
LEAD: str = "II"

#: PhysioNet database slug and the directory under the data root.
DATABASE: str = "incartdb"
DIRNAME: str = "incartdb"

#: All 75 records of INCART v1.0.0.
RECORDS: tuple[str, ...] = tuple(f"I{i:02d}" for i in range(1, 76))

#: Documentation file that may give each record's patient. Its name and layout
#: are not verified; the grouping is used only if it validates (see io.py).
PATIENT_FILE: str = "files-patients-diagnoses.txt"

#: Patients in INCART, from the database description.
N_PATIENTS: int = 32

#: Paced beats. A record containing them is left out, as MIT-BIH's paced
#: records are left out of DS2.
PACED_SYMBOLS: tuple[str, ...] = ("/", "f")

#: Bootstrap: 95% percentile interval, 1,000 draws, the Phase 1d seed.
N_BOOT: int = 1000
ALPHA: float = 0.05
SEED: int = mit_config.PHASE_1D_SEED

#: "Resampling matters" if the round-trip DS2 headline macro-F1 differs from
#: the original by more than this.
RESAMPLING_THRESHOLD: float = 0.01

#: The position rule fires if the INCART and DS2 median offsets differ by more
#: than this many samples at MODEL_FS (3 / 360 s = 8.3 ms).
POSITION_THRESHOLD_SAMPLES: int = 3

NOT_CLINICAL: str = (
    "Research result on a public dataset; not clinical validation and not a "
    "medical device."
)


def raw_dir() -> Path:
    """The raw INCART WFDB records, next to ``mitdb/`` under the data root."""
    return paths.data_root() / DIRNAME


def processed_dir() -> Path:
    """Per-beat predictions and the run JSONs for INCART."""
    return paths.processed_root() / DIRNAME
