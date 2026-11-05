"""Code shared by ``mit_bih``, ``mimic_iii`` and ``multimodal``.

Only what more than one module actually uses lives here:

- ``paths``: the data root and project root.
- ``provenance``: git commit, library versions, file hashes, JSON writing.
- ``metrics``: binary-classification metrics (MIMIC-III and fusion).

``mit_bih`` predates this package and keeps its own ``config.data_root``; a
test asserts the two agree, so they cannot drift apart.
"""
