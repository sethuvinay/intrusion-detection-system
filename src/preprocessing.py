"""Train-fit preprocessing for IDS feature tables.

The governing rule: every statistic (category levels, means, variances) is
learned on the training split only, then applied unchanged to
validation / test data — no leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TabularPreprocessor:
    """Encode categoricals and scale numerics, fit on train only.

    - Categorical columns are one-hot encoded against the category levels
      seen during ``fit``; unseen levels at transform time map to all-zeros.
    - Numeric columns are standardised with ``StandardScaler``.
    - Column order of the output is fixed at fit time, so train and test
      matrices always align.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
        scale_numeric: bool = True,
        random_state: int = 42,
    ) -> None:
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.scale_numeric = scale_numeric
        self.random_state = random_state

    # -- fitting --------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        df = df.copy()
        if self.categorical_features is None:
            self.categorical_features_ = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
        else:
            self.categorical_features_ = list(self.categorical_features)
        if self.numeric_features is None:
            self.numeric_features_ = [
                c for c in df.columns if c not in self.categorical_features_
            ]
        else:
            self.numeric_features_ = list(self.numeric_features)

        self.categories_: Dict[str, List[str]] = {
            c: pd.Categorical(df[c].astype(str)).categories.tolist()
            for c in self.categorical_features_
        }
        self.scaler_ = None
        if self.scale_numeric and self.numeric_features_:
            self.scaler_ = StandardScaler().fit(
                df[self.numeric_features_].astype(float)
            )
        self.feature_names_ = list(self.numeric_features_) + [
            f"{c}__{v}"
            for c in self.categorical_features_
            for v in self.categories_[c]
        ]
        return self

    # -- transforming ----------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()
        parts: List[pd.DataFrame] = []

        if self.numeric_features_:
            num = df[self.numeric_features_].astype(float)
            if self.scaler_ is not None:
                num = pd.DataFrame(
                    self.scaler_.transform(num),
                    columns=self.numeric_features_,
                    index=df.index,
                )
            parts.append(num)

        for col in self.categorical_features_:
            # Build from a Series (not a bare Categorical) so the dummy
            # frame keeps df's index — otherwise concat() would outer-join
            # on mismatched indexes and introduce NaNs.
            codes = pd.Series(
                pd.Categorical(
                    df[col].astype(str), categories=self.categories_[col]
                ),
                index=df.index,
                name=col,
            )
            dummies = pd.get_dummies(codes, prefix=col, prefix_sep="__")
            expected = [f"{col}__{v}" for v in self.categories_[col]]
            dummies = dummies.reindex(columns=expected, fill_value=0)
            parts.append(dummies.astype(float))

        out = pd.concat(parts, axis=1)
        return out[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def handle_imbalance(
    X: pd.DataFrame,
    y,
    strategy: str = "none",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """Address class imbalance in IDS training data.

    Attack traffic is the minority class in both benchmarks. Strategies:

    - ``"none"`` — leave the data as-is.
    - ``"undersample"`` — randomly downsample the majority (normal) class to
      the minority class size. Simple, fast, and lossless for the attack
      samples that matter most.
    - ``"class_weight"`` — keep all rows and return per-sample weights
      (inverse class frequency) for estimators that accept ``sample_weight``.

    Returns ``(X_out, y_out, sample_weight)`` where ``sample_weight`` is
    ``None`` unless ``strategy="class_weight"``.
    """
    y_arr = np.asarray(y)
    if strategy == "none":
        return X, y_arr, None

    if strategy == "undersample":
        rng = np.random.default_rng(random_state)
        classes, counts = np.unique(y_arr, return_counts=True)
        target = int(counts.min())
        idx_parts = []
        for cls in classes:
            cls_idx = np.where(y_arr == cls)[0]
            if len(cls_idx) > target:
                cls_idx = rng.choice(cls_idx, size=target, replace=False)
            idx_parts.append(cls_idx)
        idx = np.concatenate(idx_parts)
        rng.shuffle(idx)
        X_out = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        return X_out, y_arr[idx], None

    if strategy == "class_weight":
        classes, counts = np.unique(y_arr, return_counts=True)
        weights = {
            int(c): len(y_arr) / (len(classes) * n)
            for c, n in zip(classes, counts)
        }
        sample_weight = np.array([weights[int(v)] for v in y_arr], dtype=float)
        return X, y_arr, sample_weight

    raise ValueError(
        f"Unknown imbalance strategy {strategy!r}; "
        "choose 'none', 'undersample' or 'class_weight'."
    )
