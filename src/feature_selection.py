"""Systematic feature-selection methods.

In the original research, isolating the traffic signals that separate
attacks from normal flows — before benchmarking models — was the step
that lifted accuracy substantially over the all-features baseline.
Each function below returns a :class:`SelectionResult` with the selected
feature names, per-feature scores (where the method provides them) and
the reduced feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFECV,
    SelectKBest,
    chi2,
    mutual_info_classif,
)


@dataclass
class SelectionResult:
    """Outcome of a feature-selection run."""

    method: str
    selected_features: List[str]
    scores: Dict[str, float] = field(default_factory=dict)
    X_selected: pd.DataFrame = field(default=None)  # type: ignore[assignment]


def _pack(
    method: str,
    X: pd.DataFrame,
    mask: np.ndarray,
    scores: np.ndarray | None,
) -> SelectionResult:
    selected = X.columns[mask].tolist()
    score_map = (
        {c: float(s) for c, s in zip(X.columns, scores)} if scores is not None else {}
    )
    X_sel = pd.DataFrame(
        X.values[:, mask], columns=selected, index=X.index
    )
    return SelectionResult(
        method=method, selected_features=selected, scores=score_map, X_selected=X_sel
    )


def select_k_best_mutual_info(
    X: pd.DataFrame, y, k: int = 20
) -> SelectionResult:
    """Top-k features by mutual information with the target.

    Works with any real-valued features; the default choice for IDS tables
    that mix scaled numerics and one-hot categoricals.
    """
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)
    return _pack("mutual_info", X, selector.get_support(), selector.scores_)


def select_k_best_chi2(X: pd.DataFrame, y, k: int = 20) -> SelectionResult:
    """Top-k features by chi-squared statistic (requires non-negative X)."""
    if (X < 0).any().any():
        raise ValueError(
            "chi2 selection requires non-negative features. Scale to [0, 1] "
            "first (e.g. MinMaxScaler) or use mutual_info instead."
        )
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)
    return _pack("chi2", X, selector.get_support(), selector.scores_)


def select_rfecv(
    X: pd.DataFrame,
    y,
    cv: int = 3,
    scoring: str = "f1",
    n_estimators: int = 100,
    random_state: int = 42,
) -> SelectionResult:
    """Recursive feature elimination with cross-validation (RFECV).

    Lets cross-validated F1 decide how many features to keep rather than
    fixing k up front. Can be slow on very wide tables — run on a sample
    first if needed.
    """
    estimator = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=-1, random_state=random_state
    )
    selector = RFECV(estimator=estimator, cv=cv, scoring=scoring, n_jobs=-1)
    selector.fit(X, y)
    # ranking_ == 1 marks selected features; store the rank as the score.
    return _pack(
        "rfecv", X, selector.support_, selector.ranking_.astype(float)
    )


def select_by_tree_importance(
    X: pd.DataFrame, y, k: int = 20, random_state: int = 42
) -> SelectionResult:
    """Top-k features by RandomForest impurity-based importance."""
    k = min(k, X.shape[1])
    forest = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=random_state
    )
    forest.fit(X, y)
    importances = forest.feature_importances_
    top_idx = np.argsort(importances)[::-1][:k]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_idx] = True
    return _pack("tree_importance", X, mask, importances)


#: Registry used by the ``--feature-selection`` CLI flag and the dispatcher.
SELECTION_METHODS = {
    "mutual_info": select_k_best_mutual_info,
    "chi2": select_k_best_chi2,
    "rfecv": select_rfecv,
    "tree_importance": select_by_tree_importance,
}


def select_features(
    X: pd.DataFrame, y, method: str = "mutual_info", k: int = 20, **kwargs
) -> SelectionResult:
    """Dispatch to a feature-selection method by name.

    ``k`` is honoured by the top-k methods and ignored by ``rfecv``.
    """
    try:
        func = SELECTION_METHODS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown selection method {method!r}. "
            f"Choose from {sorted(SELECTION_METHODS)}."
        ) from exc
    if method == "rfecv":
        return func(X, y, **kwargs)
    return func(X, y, k=k, **kwargs)
