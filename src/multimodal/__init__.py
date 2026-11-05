"""Shared inference over the ECG (``mit_bih``) and clinical (``mimic_iii``) models,
and a fusion architecture for when paired data exists (Phase 4).

- ``ecg_encoder``: an ECG signal -> per-beat class scores and a fixed-length summary.
- ``clinical_encoder``: one ICU stay's features -> preprocessed vector and the
  clinical model's mortality score.
- ``inference``: one entry point for ECG, clinical, or both. Returns each
  model's output separately, clearly labelled; never merges them.
- ``fusion``: a late-fusion network, its training loop, and the checks a
  paired dataset must pass.

**Status.** Software integration is complete and tested. The fusion network
is tested on synthetic paired fixtures only. **Training and clinical
validation of fusion are BLOCKED**: no genuinely paired ECG + clinical dataset
is available (MIT-BIH and MIMIC-III contain different patients).

Dependencies run one way: this package imports ``mit_bih`` and ``mimic_iii``;
neither imports it.
"""
