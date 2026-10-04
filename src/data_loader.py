"""Loaders for the UNSW-NB15 and NSL-KDD intrusion-detection benchmarks.

Expected layout — datasets are *not* committed to the repository
(see the README for download instructions)::

    data/
        UNSW_NB15_training-set.csv
        UNSW_NB15_testing-set.csv
        KDDTrain+.txt
        KDDTest+.txt

The loaders return tidy ``pandas`` DataFrames with a standardised schema:
the traffic features, plus ``label`` (0 = normal, 1 = attack),
``attack_category`` (coarse family) and ``attack_type`` (original
fine-grained label, where available).

A small synthetic generator is also provided so the pipeline and the demo
notebook run without downloading the real (large) datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# NSL-KDD schema
# ---------------------------------------------------------------------------

#: The 41 NSL-KDD traffic features, in the order they appear in the raw files.
NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

#: Coarse attack families used throughout the NSL-KDD literature.
NSL_KDD_ATTACK_FAMILIES = {
    "normal": "Normal",
    # Denial of service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS",
    # Probing / surveillance
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe",
    # Remote-to-local
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    # User-to-root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R",
}

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _find_nsl_kdd_label_column(df: pd.DataFrame) -> int:
    """Locate the NSL-KDD label column by content (mirrors vary in layout)."""
    for col in df.columns:
        values = set(df[col].astype(str).str.strip().unique())
        if "normal" in values:
            return col
    raise ValueError(
        "Could not locate the label column (no column containing 'normal'). "
        "Is this an NSL-KDD KDDTrain+ / KDDTest+ file?"
    )


def load_nsl_kdd(data_dir: str | Path, split: str = "train") -> pd.DataFrame:
    """Load an NSL-KDD split into a tidy DataFrame.

    Handles mirrors that include or omit the trailing difficulty column.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    path = Path(data_dir) / ("KDDTrain+.txt" if split == "train" else "KDDTest+.txt")
    if not path.exists():
        raise FileNotFoundError(
            f"NSL-KDD file not found: {path}. Download KDDTrain+.txt / "
            "KDDTest+.txt (see README) and place them in --data-dir."
        )

    raw = pd.read_csv(path, header=None)
    label_col = _find_nsl_kdd_label_column(raw)
    attack_type = raw[label_col].astype(str).str.strip()

    feature_cols = [c for c in raw.columns if c != label_col]
    if len(feature_cols) > len(NSL_KDD_FEATURES):
        # Some mirrors append a difficulty column after the label.
        feature_cols = feature_cols[: len(NSL_KDD_FEATURES)]
    if len(feature_cols) != len(NSL_KDD_FEATURES):
        raise ValueError(
            f"Expected {len(NSL_KDD_FEATURES)} feature columns, "
            f"found {len(feature_cols)}."
        )

    df = raw[feature_cols].copy()
    df.columns = NSL_KDD_FEATURES
    df["attack_type"] = attack_type.values
    df["attack_category"] = attack_type.map(
        lambda t: NSL_KDD_ATTACK_FAMILIES.get(t, "Other")
    ).values
    df["label"] = (attack_type != "normal").astype(int).values
    return df.reset_index(drop=True)


def load_unsw_nb15(data_dir: str | Path, split: str = "train") -> pd.DataFrame:
    """Load a UNSW-NB15 train/test CSV into a tidy DataFrame."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    filename = (
        "UNSW_NB15_training-set.csv"
        if split == "train"
        else "UNSW_NB15_testing-set.csv"
    )
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 file not found: {path}. Download {filename} from the "
            "official dataset page (see README) and place it in --data-dir."
        )

    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(
            f"'label' column not found in {filename}; expected the official "
            "UNSW-NB15 training/testing-set CSV."
        )
    df = df.drop(columns=[c for c in ("id",) if c in df.columns])
    df["label"] = df["label"].astype(int)
    if "attack_cat" in df.columns:
        df["attack_category"] = df["attack_cat"].astype(str).str.strip()
    return df.reset_index(drop=True)


#: Registry of supported benchmark datasets.
SUPPORTED_DATASETS: Dict[str, Callable[..., pd.DataFrame]] = {
    "unsw-nb15": load_unsw_nb15,
    "nsl-kdd": load_nsl_kdd,
}


def load_dataset(name: str, data_dir: str | Path, split: str = "train") -> pd.DataFrame:
    """Dispatch to the loader for ``name`` (``'unsw-nb15'`` or ``'nsl-kdd'``)."""
    try:
        loader = SUPPORTED_DATASETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose from {sorted(SUPPORTED_DATASETS)}."
        ) from exc
    return loader(data_dir, split=split)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a tidy frame into ``(features, binary target)``."""
    drop = [
        c
        for c in ("label", "attack_category", "attack_cat", "attack_type")
        if c in df.columns
    ]
    return df.drop(columns=drop), df["label"].astype(int)


def generate_synthetic_ids_data(
    n_samples: int = 4000,
    n_numeric: int = 16,
    attack_ratio: float = 0.35,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a small synthetic IDS-like dataset for smoke-testing.

    The frame mimics the tidy schema (numeric flow statistics, a couple of
    categorical protocol/service fields, ``label`` / ``attack_category``)
    with attacks that shift a subset of feature means — enough signal for
    the pipeline to learn from, but **not** a substitute for the real
    benchmarks. Numbers produced from this data are illustrative only.
    """
    rng = np.random.default_rng(random_state)
    n_attack = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attack

    normal = rng.normal(0.0, 1.0, size=(n_normal, n_numeric))
    attack = rng.normal(0.0, 1.0, size=(n_attack, n_numeric))
    attack[:, :6] += rng.uniform(1.5, 3.0, size=6)  # attack signature

    df = pd.DataFrame(
        np.vstack([normal, attack]),
        columns=[f"feat_{i:02d}" for i in range(n_numeric)],
    )
    df["proto"] = rng.choice(
        ["tcp", "udp", "icmp"], size=n_samples, p=[0.6, 0.3, 0.1]
    )
    df["service"] = rng.choice(
        ["http", "dns", "ftp", "smtp", "-"],
        size=n_samples,
        p=[0.4, 0.25, 0.15, 0.1, 0.1],
    )

    label = np.array([0] * n_normal + [1] * n_attack)
    attack_families = pd.Series(
        rng.choice(["DoS", "Probe", "Exploits"], size=n_samples)
    )
    df["attack_category"] = np.where(label == 1, attack_families, "Normal")
    df["attack_type"] = np.where(label == 1, attack_families.str.lower(), "normal")
    df["label"] = label
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
