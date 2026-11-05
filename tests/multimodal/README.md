# tests/multimodal/

Tests for `src/multimodal/`: the ECG and clinical encoders, the shared
inference interface, and the fusion network.

Fusion is tested on synthetic paired fixtures only. No genuinely paired ECG +
clinical dataset is available, so no test here measures clinical performance.
