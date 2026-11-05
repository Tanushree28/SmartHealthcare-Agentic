"""External test of the MIT-BIH beat model on the St Petersburg INCART database.

INCART (PhysioNet ``incartdb`` v1.0.0) has 75 half-hour 12-lead recordings
from 32 patients, sampled at 257 Hz, with cardiologist beat annotations in the
same codes as MIT-BIH. This package scores the unchanged ``D_all_four``
ensemble on lead II, once, after resampling to 360 Hz, and compares the
result with MIT-BIH's held-out patients (DS2). Design:
``PlanDoc/specs/2026-10-29-incart-external-test-design.md``.

- ``config``: record names, rates, the lead, the fixed decision thresholds.
- ``resample``: 257 -> 360 Hz; annotation positions move with the signal.
- ``io``: one record's lead and beat labels; download and checks; patient
  grouping.
- ``evaluate``: beats at the model rate, the beat-position check, metrics,
  the patient bootstrap and the decision rules.

Dependencies run one way: ``incart`` imports ``mit_bih`` and ``common``;
``mit_bih`` never imports ``incart``.

Research results on a public dataset; not clinical validation and not a
medical device. The model's scores are uncalibrated.
"""

from __future__ import annotations
