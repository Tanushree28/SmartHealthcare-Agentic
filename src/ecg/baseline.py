"""Classical baseline classifiers for AAMI beat classification.

The point of this module is **not** to be the final model. It is to establish an
honest floor: a number a later deep model must beat before anyone can claim it
learned something. It also confirms that the Phase 1 output is learnable at all.

Everything is wrapped in a :class:`sklearn.pipeline.Pipeline` so the scaler is
fitted inside cross-validation and inside ``fit`` only. Leakage through feature
scaling then becomes structurally impossible rather than a rule someone has to
remember.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

CLASSIFIERS = ("random_forest", "lda")

#: Fixed so a run is reproducible. Chosen on the validation records only --
#: never on the test split.
#:
#: ``min_samples_leaf=50`` is the one parameter that matters here, and it is not
#: arbitrary. With the sklearn default of 1-2, fully-grown trees memorise
#: patient-specific beat morphology from the training records. That works within
#: a patient and collapses across patients, which is exactly what an
#: inter-patient split measures: S recall on held-out records was 0.099 with
#: leaf=2 and 0.519 with leaf=50. Large leaves force each split to be supported
#: by enough beats to reflect a general rule -- above all "this beat arrived
#: early", the timing signal that actually defines the S class -- rather than
#: one patient's waveform quirk.
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 50,
    "n_jobs": -1,
}

FEATURE_SETS = {
    "rr+morphology": {"use_rr": True, "use_morphology": True},
    "rr_only": {"use_rr": True, "use_morphology": False},
    "morphology_only": {"use_rr": False, "use_morphology": True},
}


def build_pipeline(
    kind: str = "random_forest",
    seed: int = config.DEFAULT_SPLIT_SEED,
    balanced: bool = True,
    classes: np.ndarray | None = None,
) -> Pipeline:
    """Build a scaler + classifier pipeline.

    ``balanced`` counteracts the 86%-normal class distribution. This is where
    Phase 1's deliberately-deferred class weights get applied: at training time,
    next to the loss, rather than by resampling the dataset and destroying the
    true distribution.
    """
    if kind not in CLASSIFIERS:
        raise ValueError(f"unknown classifier {kind!r}; expected one of {list(CLASSIFIERS)}")

    if kind == "random_forest":
        model = RandomForestClassifier(
            random_state=seed,
            class_weight="balanced" if balanced else None,
            **RANDOM_FOREST_PARAMS,
        )
    else:
        # LinearDiscriminantAnalysis has no class_weight. Uniform priors are the
        # equivalent lever: they remove the prior advantage the N class would
        # otherwise get purely from being 86% of the data.
        priors = None
        if balanced:
            n = len(classes) if classes is not None else len(config.AAMI_CLASSES)
            priors = np.full(n, 1.0 / n)
        model = LinearDiscriminantAnalysis(priors=priors)

    # StandardScaler matters here: RR features are in seconds (~1.0) while the
    # morphology features are z-scored amplitudes. LDA in particular is sensitive
    # to that scale mismatch.
    return Pipeline([("scaler", StandardScaler()), ("classifier", model)])


def feature_importances(pipeline: Pipeline, feature_names) -> dict[str, float]:
    """Per-feature importance, when the classifier exposes one.

    This is the part that answers *which half of the Phase 1 output carries the
    signal* -- the question the ablations exist to settle.
    """
    model = pipeline.named_steps["classifier"]
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    return dict(zip(feature_names, (float(v) for v in values)))


def grouped_importance(importances: dict[str, float]) -> dict[str, float]:
    """Collapse per-feature importance into the two feature families."""
    if not importances:
        return {}
    groups = {"rr": 0.0, "morphology": 0.0}
    for name, value in importances.items():
        groups["morphology" if name.startswith("morph_") else "rr"] += value
    return groups


# ---------------------------------------------------------------------------
# Leakage tripwire
# ---------------------------------------------------------------------------


class NormalizationMismatchError(Exception):
    """The stored normalization does not belong to the split being trained on."""


def load_split_or_refuse(processed_dir, required_strategy: str | None = None) -> dict:
    """Load a processed run's split, refusing if its normalization does not match.

    Why this exists: Phase 1 z-scores the beats using statistics fitted on *its*
    training records. If you preprocess with one split and then train with a
    different one, the test beats were scaled using statistics that already saw
    them. That is quiet, plausible-looking leakage, and nothing downstream would
    reveal it -- so it is refused here rather than trusted to discipline.
    """
    import json
    from pathlib import Path

    processed_dir = Path(processed_dir).expanduser()
    splits_path = processed_dir / "splits.json"
    norm_path = processed_dir / "normalization.json"

    for path in (splits_path, norm_path):
        if not path.exists():
            raise FileNotFoundError(
                f"no {path.name} in {processed_dir}; "
                f"run scripts/preprocess_mitbih.py first"
            )

    split = json.loads(splits_path.read_text())
    norm = json.loads(norm_path.read_text())

    split_train = sorted(split.get("train", []))
    norm_train = sorted(norm.get("train_records", []))

    if split_train != norm_train:
        raise NormalizationMismatchError(
            f"normalization.json was fitted on {len(norm_train)} record(s) but "
            f"splits.json lists {len(split_train)} training record(s). The beats "
            f"on disk were scaled for a different split, so the test set is "
            f"contaminated. Re-run scripts/preprocess_mitbih.py for this split."
        )

    if required_strategy and split.get("strategy") != required_strategy:
        raise NormalizationMismatchError(
            f"{processed_dir} was preprocessed with strategy "
            f"{split.get('strategy')!r}, but {required_strategy!r} was requested. "
            f"The stored beats were normalized using {split.get('strategy')!r} "
            f"training records, which overlap the {required_strategy!r} test set. "
            f"Re-run: python scripts/preprocess_mitbih.py "
            f"--strategy {required_strategy} --out-dir <new dir>"
        )

    return split
