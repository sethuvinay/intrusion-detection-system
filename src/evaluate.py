"""Evaluation utilities with emphasis on the false-positive rate.

In intrusion detection, accuracy alone is misleading: the false-positive
rate — the fraction of *benign* traffic flagged as malicious — drives
analyst workload and alert fatigue. Cutting it while holding detection
sensitivity (recall) was a primary objective of this research, so every
helper here reports FPR alongside the standard metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

__all__ = [
    "binary_metrics",
    "summarize_results",
    "improvement_over_baseline",
    "plot_confusion_matrix",
    "plot_training_history",
    "save_metrics",
    "load_metrics",
]


def binary_metrics(y_true, y_pred) -> Dict[str, float]:
    """Full binary-classification report for IDS predictions.

    ``y_true`` / ``y_pred`` use 0 = normal, 1 = attack.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp

    def _safe(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    accuracy = _safe(tp + tn, total)
    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)  # sensitivity / detection rate
    specificity = _safe(tn, tn + fp)
    fpr = _safe(fp, fp + tn)  # false-positive rate — the key IDS metric
    fnr = _safe(fn, fn + tp)
    f1 = _safe(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,  # == sensitivity
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize_results(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Model-name -> metrics dict, rendered as a comparison table."""
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fpr": m["fpr"],
                "fnr": m["fnr"],
            }
        )
    return pd.DataFrame(rows).set_index("model")


def improvement_over_baseline(
    metrics: Dict[str, float], baseline: Dict[str, float]
) -> Dict[str, float]:
    """Relative gains of ``metrics`` versus a ``baseline`` metrics dict.

    Returns the accuracy gain (%) and the false-positive-rate reduction (%)
    — the two headline quantities reported in the original research.
    """
    acc_gain = (
        (metrics["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100
        if baseline["accuracy"]
        else 0.0
    )
    fpr_reduction = (
        (baseline["fpr"] - metrics["fpr"]) / baseline["fpr"] * 100
        if baseline["fpr"]
        else 0.0
    )
    return {
        "accuracy_gain_pct": float(acc_gain),
        "fpr_reduction_pct": float(fpr_reduction),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: Tuple[str, str] = ("Normal", "Attack"),
    save_path: str | Path | None = None,
):
    """Plot a labelled confusion matrix; saves to ``save_path`` if given."""
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_training_history(
    history: Dict[str, list], save_path: str | Path | None = None
):
    """Plot training loss (and validation accuracy when available)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    if history.get("val_acc"):
        axes[1].plot(history["val_acc"], marker="o", color="green")
        axes[1].set_title("Validation accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_metrics(metrics: Dict, path: str | Path) -> None:
    """Write a metrics dict to JSON (numpy-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_builtin(metrics), f, indent=2)


def load_metrics(path: str | Path) -> Dict:
    """Read a metrics JSON file written by :func:`save_metrics`."""
    with open(path) as f:
        return json.load(f)
