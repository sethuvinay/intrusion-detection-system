"""End-to-end training entry point.

Pipeline: load -> preprocess (fit on train only) -> feature selection ->
train -> evaluate (accuracy, FPR, sensitivity, ...).

Examples::

    # Synthetic smoke-test (no download needed)
    python src/train.py --dataset synthetic --model all

    # NSL-KDD with mutual-information feature selection
    python src/train.py --dataset nsl-kdd --data-dir data --model all \\
        --feature-selection mutual_info --k 25

    # UNSW-NB15, deep model only
    python src/train.py --dataset unsw-nb15 --data-dir data --model mlp \\
        --epochs 30

On headless servers, set ``MPLBACKEND=Agg`` so plots save without a display.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data_loader import (
    SUPPORTED_DATASETS,
    generate_synthetic_ids_data,
    load_dataset,
    split_features_target,
)
from preprocessing import TabularPreprocessor, handle_imbalance
from feature_selection import SELECTION_METHODS, select_features
from models import (
    IntrusionCNN1D,
    IntrusionMLP,
    evaluate_torch,
    get_baseline_models,
    make_dataloaders,
    train_torch_model,
)
from evaluate import (
    binary_metrics,
    improvement_over_baseline,
    plot_confusion_matrix,
    plot_training_history,
    save_metrics,
    summarize_results,
)

TORCH_MODELS = {"mlp": IntrusionMLP, "cnn": IntrusionCNN1D}
CLASSICAL_MODELS = ["random_forest", "xgboost", "svm"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate IDS models on UNSW-NB15 / NSL-KDD."
    )
    p.add_argument(
        "--dataset",
        choices=["synthetic", *sorted(SUPPORTED_DATASETS)],
        default="synthetic",
        help="Dataset to use (default: synthetic smoke-test).",
    )
    p.add_argument(
        "--data-dir", default="data", help="Directory holding the raw files."
    )
    p.add_argument(
        "--model",
        choices=["all", *CLASSICAL_MODELS, *sorted(TORCH_MODELS)],
        default="all",
        help="Which model(s) to train.",
    )
    p.add_argument(
        "--feature-selection",
        choices=["none", *sorted(SELECTION_METHODS)],
        default="mutual_info",
        help="Feature-selection method applied before training.",
    )
    p.add_argument(
        "--k", type=int, default=25, help="Top-k for top-k selection methods."
    )
    p.add_argument(
        "--imbalance",
        choices=["none", "undersample", "class_weight"],
        default="none",
        help="How to handle class imbalance.",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=4000,
                   help="Rows for --dataset synthetic.")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--hidden-dims",
        default="128,64",
        help="Comma-separated MLP hidden sizes.",
    )
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument(
        "--output-dir", default="outputs", help="Where metrics/plots go."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_splits(args: argparse.Namespace):
    """Return (X_train, y_train, X_test, y_test) as raw frames."""
    if args.dataset == "synthetic":
        df = generate_synthetic_ids_data(
            n_samples=args.n_samples, random_state=args.seed
        )
        X, y = split_features_target(df)
        return train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

    train_df = load_dataset(args.dataset, args.data_dir, split="train")
    try:
        test_df = load_dataset(args.dataset, args.data_dir, split="test")
    except FileNotFoundError:
        print("Test file not found — splitting the training file instead.")
        X, y = split_features_target(train_df)
        return train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    return X_train, X_test, y_train, y_test


def train_classical(name: str, X_train, y_train, sample_weight, seed: int):
    clf = get_baseline_models(random_state=seed)[name]
    fit_kwargs = {}
    if sample_weight is not None and name in ("random_forest", "xgboost"):
        fit_kwargs["sample_weight"] = sample_weight
    clf.fit(X_train, y_train, **fit_kwargs)
    return clf


def train_deep(name: str, X_train, y_train, X_test, args, out_dir: Path):
    hidden = tuple(int(h) for h in args.hidden_dims.split(","))
    model_cls = TORCH_MODELS[name]
    if name == "mlp":
        model = model_cls(
            input_dim=X_train.shape[1], hidden_dims=hidden, dropout=args.dropout
        )
    else:
        model = model_cls(input_dim=X_train.shape[1], dropout=args.dropout)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=args.seed, stratify=y_train
    )
    train_loader, val_loader = make_dataloaders(
        X_tr, y_tr, X_val, y_val, batch_size=args.batch_size
    )
    history = train_torch_model(
        model, train_loader, val_loader, epochs=args.epochs, lr=args.lr
    )
    plot_training_history(history, save_path=out_dir / f"training_{name}.png")
    torch.save(model.state_dict(), out_dir / f"{name}.pt")

    test_loader, _ = make_dataloaders(
        X_test, np.zeros(len(X_test)), batch_size=args.batch_size,
        shuffle_train=False,
    )
    result = evaluate_torch(model, test_loader)
    return result["y_pred"], history


def run_experiment(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- load -----------------------------------------------------------
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_splits(args)
    y_train = np.asarray(y_train_raw)
    y_test = np.asarray(y_test_raw)
    print(f"train={len(y_train)} test={len(y_test)} "
          f"attack-rate={y_train.mean():.3f}")

    # -- preprocess (fit on train only) ---------------------------------
    pre = TabularPreprocessor(random_state=args.seed)
    X_train_p = pre.fit_transform(X_train_raw)
    X_test_p = pre.transform(X_test_raw)
    print(f"encoded features: {X_train_p.shape[1]}")
    joblib.dump(pre, out_dir / "preprocessor.joblib")

    # -- imbalance -------------------------------------------------------
    X_train_p, y_train, sample_weight = handle_imbalance(
        X_train_p, y_train, strategy=args.imbalance, random_state=args.seed
    )

    # -- feature selection (fit on train only) ---------------------------
    if args.feature_selection == "none":
        selected = list(X_train_p.columns)
        X_train_s, X_test_s = X_train_p, X_test_p
    else:
        sel = select_features(
            X_train_p, y_train, method=args.feature_selection, k=args.k,
            random_state=args.seed,
        )
        selected = sel.selected_features
        X_train_s, X_test_s = sel.X_selected, X_test_p[selected]
    print(f"selected {len(selected)} features via {args.feature_selection}")
    (out_dir / "selected_features.txt").write_text("\n".join(selected))

    # -- train -----------------------------------------------------------
    if args.model == "all":
        wanted = CLASSICAL_MODELS + sorted(TORCH_MODELS)
    elif args.model in CLASSICAL_MODELS:
        wanted = [args.model]
    else:
        wanted = [args.model]

    results, histories = {}, {}
    for name in wanted:
        print(f"\n=== {name} ===")
        if name in TORCH_MODELS:
            y_pred, history = train_deep(
                name, X_train_s.values, y_train, X_test_s.values, args, out_dir
            )
            histories[name] = history
        else:
            clf = train_classical(
                name, X_train_s.values, y_train, sample_weight, args.seed
            )
            joblib.dump(clf, out_dir / f"{name}.joblib")
            y_pred = clf.predict(X_test_s.values)
        results[name] = binary_metrics(y_test, y_pred)
        plot_confusion_matrix(
            y_test, y_pred, save_path=out_dir / f"confusion_matrix_{name}.png"
        )

    # -- report ----------------------------------------------------------
    table = summarize_results(results)
    print("\n" + table.round(4).to_string())

    classical = {n: m for n, m in results.items() if n in CLASSICAL_MODELS}
    if classical and any(n in TORCH_MODELS for n in results):
        baseline_name = min(classical, key=lambda n: classical[n]["accuracy"])
        best_deep = max(
            [n for n in results if n in TORCH_MODELS],
            key=lambda n: results[n]["accuracy"],
        )
        gains = improvement_over_baseline(
            results[best_deep], classical[baseline_name]
        )
        print(
            f"\nBest deep model ({best_deep}) vs weakest baseline "
            f"({baseline_name}): "
            f"+{gains['accuracy_gain_pct']:.1f}% accuracy, "
            f"{gains['fpr_reduction_pct']:.1f}% FPR reduction."
        )
        results["_gains_vs_baseline"] = gains

    save_metrics(
        {"config": vars(args), "selected_features": selected,
         "results": results},
        out_dir / "metrics.json",
    )
    print(f"\nArtifacts written to {out_dir.resolve()}")
    return results


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
