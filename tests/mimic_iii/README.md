# tests/mimic_iii/

Tests for `src/mimic_iii/`.

They must run offline on synthetic tables, like `tests/mit_bih/`, so the suite
never needs real MIMIC data (and never sees credentialed data).
`synthetic.py` builds the synthetic MIMIC-III-shaped database the tests use;
`test_real_demo.py` (marker `requires_data`) checks facts about the real Demo.
