from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd
from datasets import Dataset, DatasetDict

@dataclass
class DataConfig:
    local_path: Optional[str]
    text_col: str
    context_col: Optional[str]
    label_col: str
    test_size: float
    val_size: float
    max_samples: Optional[int]

def load_local_dataset(cfg: DataConfig) -> Dataset:
    if cfg.local_path is None:
        raise ValueError("data.local_path is null. Provide a local CSV/JSONL with columns text/label(/context).")

    if cfg.local_path.endswith(".csv"):
        df = pd.read_csv(cfg.local_path)
    elif cfg.local_path.endswith(".jsonl"):
        df = pd.read_json(cfg.local_path, lines=True)
    else:
        raise ValueError("Unsupported file type. Use .csv or .jsonl")

    required = {cfg.text_col, cfg.label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if cfg.context_col is not None and cfg.context_col not in df.columns:
        raise ValueError(f"context_col='{cfg.context_col}' not found in data")

    df = df.dropna(subset=[cfg.text_col, cfg.label_col]).copy()
    df[cfg.text_col] = df[cfg.text_col].astype(str)
    df[cfg.label_col] = df[cfg.label_col].astype(str)

    if cfg.context_col is not None:
        df[cfg.context_col] = df[cfg.context_col].fillna("").astype(str)

    if cfg.max_samples:
        df = df.sample(n=min(cfg.max_samples, len(df)), random_state=42).reset_index(drop=True)

    return Dataset.from_pandas(df, preserve_index=False)

def make_splits(ds: Dataset, test_size: float, val_size: float, seed: int) -> DatasetDict:
    tmp = ds.train_test_split(test_size=test_size, seed=seed)
    train = tmp["train"]
    test = tmp["test"]

    val_rel = val_size / (1.0 - test_size)
    tmp2 = train.train_test_split(test_size=val_rel, seed=seed)
    return DatasetDict(train=tmp2["train"], validation=tmp2["test"], test=test)

def build_label_maps(ds: Dataset, label_col: str) -> Dict[str, Any]:
    labels = sorted(set(ds[label_col]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return {"label2id": label2id, "id2label": id2label}
