"""Real INCART on disk (skipped by default; run with ``-m requires_data``).

Asserts only shapes and counts; prints nothing from the record.
"""

from __future__ import annotations

import pytest

from incart import config, io


@pytest.mark.requires_data
def test_a_real_record_loads_at_257_hz_with_lead_ii_and_known_symbols():
    if not (config.raw_dir() / "I01.hea").exists():
        pytest.skip(f"INCART not downloaded to {config.raw_dir()}")

    rec = io.load_labelled_record(config.raw_dir(), "I01")  # strict mapping: raises on unknown

    assert rec.fs == 257
    assert rec.lead == "II"
    assert rec.n_beats > 1000  # 30 minutes of heartbeats
    assert rec.beat_labels.min() >= 0 and rec.beat_labels.max() <= 4
