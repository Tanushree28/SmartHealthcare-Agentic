"""MIT-BIH acquisition and integrity checking (PLAN.md step 0.3).

The download logic lives here rather than in ``scripts/download_mitbih.py`` so
it can be tested without a network connection and without importing a script by
path. The script is a thin CLI over these functions.

Idempotence is the point: re-running must not re-fetch, and a partially
downloaded record must be caught here rather than surfacing much later as
bizarre signal values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import wfdb

from . import config

DATABASE = "mitdb"
REQUIRED_EXTENSIONS = (".hea", ".dat", ".atr")

#: Bits per stored sample, by WFDB format code. Used to predict the exact
#: ``.dat`` size from the header so truncation is detectable without reading
#: the whole file.
FORMAT_BITS: dict[str, int] = {
    "8": 8, "80": 8,
    "16": 16, "61": 16, "160": 16,
    "212": 12,
    "310": 10, "311": 10,
    "24": 24,
    "32": 32,
}


@dataclass(frozen=True)
class RecordCheck:
    """Outcome of verifying one record on disk."""

    record: str
    ok: bool
    reason: str = ""

    def __str__(self) -> str:
        return f"{self.record}: {'ok' if self.ok else self.reason}"


def expected_dat_bytes(header) -> int | None:
    """Predicted ``.dat`` size in bytes, or ``None`` for a mixed-format record."""
    formats = {str(f) for f in (header.fmt or [])}
    if len(formats) != 1:
        return None
    bits = FORMAT_BITS.get(formats.pop())
    if bits is None:
        return None
    total_bits = int(header.sig_len) * int(header.n_sig) * bits
    return math.ceil(total_bits / 8)


def check_record(directory: Path, name: str, expected_fs: int | None = config.FS) -> RecordCheck:
    """Verify one record: files present, header sane, ``.dat`` not truncated."""
    directory = Path(directory)

    for ext in REQUIRED_EXTENSIONS:
        if not (directory / f"{name}{ext}").exists():
            return RecordCheck(name, False, f"missing {name}{ext}")

    try:
        header = wfdb.rdheader(str(directory / name))
    except Exception as exc:  # noqa: BLE001 -- any header problem is a failed check
        return RecordCheck(name, False, f"unreadable header ({type(exc).__name__}: {exc})")

    fs = int(round(float(header.fs)))
    if expected_fs is not None and fs != expected_fs:
        return RecordCheck(name, False, f"sampling rate {fs} Hz, expected {expected_fs} Hz")

    expected = expected_dat_bytes(header)
    if expected is not None:
        actual = (directory / f"{name}.dat").stat().st_size
        if actual < expected:
            return RecordCheck(
                name,
                False,
                f"truncated .dat: {actual} bytes on disk, header implies {expected} "
                f"({header.sig_len} samples x {header.n_sig} channels)",
            )

    return RecordCheck(name, True)


def verify(
    directory: Path | str,
    records=config.ALL_RECORDS,
    expected_fs: int | None = config.FS,
) -> list[RecordCheck]:
    """Verify every expected record. Never touches the network."""
    directory = Path(directory).expanduser()
    return [check_record(directory, str(r), expected_fs) for r in records]


def missing_records(directory: Path | str, records=config.ALL_RECORDS) -> list[str]:
    """Records that are absent or fail their integrity check."""
    return [check.record for check in verify(directory, records) if not check.ok]


def download(
    directory: Path | str = None,
    records=config.ALL_RECORDS,
    force: bool = False,
) -> list[RecordCheck]:
    """Download MIT-BIH into ``directory``, skipping records already valid.

    Returns the post-download verification results. Raises if anything is still
    invalid afterwards -- a half-complete dataset must not look like success.
    """
    directory = Path(directory or config.mitdb_dir()).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    wanted = [str(r) for r in records]
    todo = wanted if force else missing_records(directory, wanted)

    if todo:
        # wfdb fetches the .hea/.dat/.atr triplet per record.
        wfdb.dl_database(DATABASE, dl_dir=str(directory), records=todo)

    checks = verify(directory, wanted)
    bad = [c for c in checks if not c.ok]
    if bad:
        raise RuntimeError(
            "download completed but these records failed verification:\n  "
            + "\n  ".join(str(c) for c in bad)
        )
    return checks


def directory_size_bytes(directory: Path | str) -> int:
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
