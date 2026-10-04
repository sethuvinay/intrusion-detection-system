"""Model zoo: classical baselines and PyTorch deep models for IDS.

Classical baselines (scikit-learn / XGBoost) establish the reference point;
the PyTorch models (MLP, 1-D CNN) are the deep-learning stage that, in the
original research, cut the false-positive rate while holding sensitivity.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------


def get_baseline_models(random_state: int = 42) -> Dict[str, object]:
    """Return the classical baseline classifiers used for benchmarking."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# Deep models (PyTorch)
# ---------------------------------------------------------------------------


class IntrusionMLP(nn.Module):
    """Feed-forward network for tabular flow features.

    Batch normalisation + dropout keep it regularised on the relatively
    small feature vectors typical of IDS benchmarks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntrusionCNN1D(nn.Module):
    """1-D CNN that treats the ordered feature vector as a short signal."""

    def __init__(
        self,
        input_dim: int,
        channels: Tuple[int, int] = (32, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[1] * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.unsqueeze(1)))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def make_dataloaders(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    batch_size: int = 256,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build PyTorch dataloaders from numpy/pandas arrays."""

    def _t(a, dtype):
        return torch.tensor(np.asarray(a), dtype=dtype)

    train_ds = TensorDataset(
        _t(X_train, torch.float32), _t(y_train, torch.long)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train
    )
    val_loader = None
    if X_val is not None and y_val is not None:
        val_ds = TensorDataset(_t(X_val, torch.float32), _t(y_val, torch.long))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate_torch(
    model: nn.Module, loader: DataLoader, device: Optional[str] = None
) -> Dict[str, object]:
    """Score a trained torch model; returns accuracy and raw predictions."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def predict_torch(
    model: nn.Module,
    X,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> np.ndarray:
    """Predict class labels for a feature matrix with a torch model."""
    loader, _ = make_dataloaders(
        X, np.zeros(len(X)), batch_size=batch_size, shuffle_train=False
    )
    out = evaluate_torch(model, loader, device=device)
    return out["y_pred"]  # type: ignore[return-value]


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """Train a torch classifier; keeps the best-validation-accuracy weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history: Dict[str, list] = {"train_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

        history["train_loss"].append(running_loss / total)
        msg = (
            f"epoch {epoch + 1:>3}/{epochs} "
            f"loss={running_loss / total:.4f} "
            f"train_acc={correct / total:.4f}"
        )
        if val_loader is not None:
            val_acc = evaluate_torch(model, val_loader, device=device)[
                "accuracy"
            ]
            history["val_acc"].append(val_acc)
            msg += f" val_acc={val_acc:.4f}"
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
