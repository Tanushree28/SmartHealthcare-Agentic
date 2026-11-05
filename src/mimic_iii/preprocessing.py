"""Imputation, scaling and encoding, fitted on the training split only.

Everything that learns from data -- the imputation medians, the scaling means
and standard deviations, the category vocabulary -- lives inside one sklearn
``Pipeline`` together with the classifier. Fitting that pipeline on the
training rows is then the *only* way to fit the preprocessing, so validation
and test statistics cannot reach it even by accident. The same pattern
``mit_bih.baseline`` uses for its scaler.

- Numeric summaries: median imputation, then standardisation. A variable that
  is entirely missing in training is kept (imputed as 0) so the output width
  does not depend on the data.
- Missingness flags: passed through. They are computed before imputation, so
  "never measured" stays visible to the model after the value is filled in.
- Categories: missing becomes its own level; levels unseen in training are
  encoded as all zeros rather than raising at prediction time.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import features


def build_preprocessor() -> ColumnTransformer:
    cols = features.feature_columns()
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing",
                                 keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("numeric", numeric, cols["numeric"]),
            ("flags", "passthrough", cols["flags"]),
            ("categorical", categorical, cols["categorical"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(
    C: float = 1.0,
    class_weight: str | None = "balanced",
    max_iter: int = 5000,
    seed: int = 0,
) -> Pipeline:
    """Preprocessing plus L2 logistic regression, as one fit-on-train unit."""
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter,
                                   random_state=seed)),
    ])


def output_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns the classifier sees, after one-hot encoding."""
    return [str(n) for n in pipeline.named_steps["pre"].get_feature_names_out()]
