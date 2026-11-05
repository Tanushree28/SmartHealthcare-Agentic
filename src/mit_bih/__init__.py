"""MIT-BIH Arrhythmia Database: ECG beat preprocessing, classification and evaluation.

Phases 0-1g of the Smart Healthcare agentic cardiac-AI project:

- Preprocessing (Phase 1): ``io``, ``filters``, ``normalize``, ``labels``,
  ``segment``, ``splits``, ``pipeline``, ``acquire``.
- Classical baseline (Phase 1b): ``features``, ``baseline``.
- Neural classifier (Phases 1d-1g): ``dataset``, ``folds``, ``sequence``,
  ``nn/``, ``thresholds``, ``ensemble``.
- Evaluation: ``evaluate``, ``stats``, ``ranking``, ``calibration``, ``plots``.

Renamed from ``ecg`` on 2026-10-14 when the project was split into
``mit_bih``, ``mimic_iii`` and ``multimodal``. See PlanDoc/PLAN.md for later
phases.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
