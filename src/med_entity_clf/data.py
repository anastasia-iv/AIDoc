from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

class DataValidationError(ValueError):
    """Raised when input dataset does not match expected schema."""
    
def validate_dataframe(
    df: pd.DataFrame,
    *,
    text_col: str,
    label_col: str,
    context_col: Optional[str] = None,
    allow_empty_context: bool = True,
    dropna: bool = True,
    min_rows: int = 1,
) -> pd.DataFrame:
    """
    Validate and (optionally) clean a dataframe for text classification.

    Returns a cleaned dataframe (copy) if validation passes, otherwise raises DataValidationError.
    """

    required = [text_col, label_col]
    if context_col:
        required.append(context_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    out = df.copy()

    if dropna:
        out = out.dropna(subset=[text_col, label_col] + ([context_col] if context_col else []))

    out[text_col] = out[text_col].astype(str).str.strip()
    out[label_col] = out[label_col].astype(str).str.strip()

    if context_col:
        out[context_col] = out[context_col].astype(str)
        if allow_empty_context:
            out[context_col] = out[context_col].fillna("").astype(str)
        out[context_col] = out[context_col].str.strip()

    out = out[(out[text_col] != "") & (out[label_col] != "")]

    if len(out) < min_rows:
        raise DataValidationError(f"Dataset has too few rows after cleaning: {len(out)} < {min_rows}")

    return out

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
        raise ValueError(
            "data.local_path is null. Provide a local CSV/JSONL with columns text/label(/context)."
        )

    if cfg.local_path.endswith(".csv"):
        df = pd.read_csv(cfg.local_path)
    elif cfg.local_path.endswith(".jsonl"):
        df = pd.read_json(cfg.local_path, lines=True)
    else:
        raise ValueError("Unsupported file type. Use .csv or .jsonl")

    df = validate_dataframe(
        df,
        text_col=cfg.text_col,
        label_col=cfg.label_col,
        context_col=cfg.context_col,
        allow_empty_context=True,
        dropna=True,
        min_rows=10,
    )

    if cfg.max_samples:
        df = df.sample(
            n=min(cfg.max_samples, len(df)),
            random_state=42
        ).reset_index(drop=True)

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