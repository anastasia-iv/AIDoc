from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from med_entity_clf.utils import setup_logging, set_seed, ensure_dir
from med_entity_clf.data import DataConfig, load_local_dataset, make_splits
from med_entity_clf.modeling import format_input


logger = logging.getLogger("evaluate_cli")


def _normalize_label2id(label2id: Dict[Any, Any]) -> Dict[str, int]:
    """
    HF config can store label2id with str keys (JSON) or int keys.
    We normalize to {label_str: id_int}.
    """
    out: Dict[str, int] = {}
    for k, v in label2id.items():
        out[str(k)] = int(v)
    return out


def tokenize_for_eval(
    ds,
    tokenizer,
    *,
    data_cfg: DataConfig,
    use_context: bool,
    max_length: int,
    label2id: Dict[str, int],
):
    text_col = data_cfg.text_col
    ctx_col = data_cfg.context_col
    label_col = data_cfg.label_col

    def _map(ex):
        text = ex[text_col]
        context = ex[ctx_col] if (ctx_col is not None and ctx_col in ex) else ""
        inp = format_input(text, context, use_context=use_context)
        tok = tokenizer(inp, truncation=True, max_length=max_length)
        lbl = str(ex[label_col])
        if lbl not in label2id:
            raise ValueError(
                f"Label '{lbl}' not found in model label2id. "
                f"Known labels sample: {list(label2id.keys())[:10]}"
            )
        tok["labels"] = label2id[lbl]
        return tok

    return ds.map(_map, remove_columns=ds.column_names)


@torch.no_grad()
def run_eval(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    return {"accuracy": acc, "macro_f1": macro_f1, "num_samples": int(y_true.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (same as train)")
    ap.add_argument("--model_dir", required=True, help="Directory with save_pretrained() artifacts")
    ap.add_argument("--out", required=True, help="Where to save metrics JSON")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Which split to evaluate")
    args = ap.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log_level = cfg.get("logging", {}).get("level", "INFO")
    setup_logging(log_level)

    seed = int(cfg.get("training", {}).get("random_seed", 42))
    set_seed(seed)

    device_cfg = str(cfg.get("hardware", {}).get("device", "cpu"))
    if device_cfg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device(device_cfg)
        logger.info("CUDA available: %s", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")

    data_cfg = DataConfig(
        local_path=cfg["data"]["local_path"],
        text_col=cfg["data"].get("text_col", "text"),
        context_col=cfg["data"].get("context_col", "context"),
        label_col=cfg["data"].get("label_col", "label"),
        test_size=float(cfg["data"].get("test_size", 0.1)),
        val_size=float(cfg["data"].get("val_size", 0.1)),
        max_samples=cfg["data"].get("max_samples", None),
    )

    use_context = bool(cfg.get("hf", {}).get("use_context", True))
    max_length = int(cfg.get("training", {}).get("max_length", 256))

    model_dir = Path(args.model_dir)
    out_path = Path(args.out)
    ensure_dir(str(out_path.parent))

    logger.info("Config: %s", args.config)
    logger.info("Model dir: %s", model_dir)
    logger.info("Dataset: %s", data_cfg.local_path)
    logger.info("Split: %s", args.split)
    logger.info("Device: %s", device)

    ds = load_local_dataset(data_cfg)
    dd = make_splits(ds, data_cfg.test_size, data_cfg.val_size, seed)

    split_ds = dd[args.split]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    if not hasattr(model.config, "label2id") or not model.config.label2id:
        raise RuntimeError("Model config has no label2id; cannot map string labels to ids for evaluation.")
    label2id = _normalize_label2id(model.config.label2id)

    tok_ds = tokenize_for_eval(
        split_ds,
        tokenizer,
        data_cfg=data_cfg,
        use_context=use_context,
        max_length=max_length,
        label2id=label2id,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dl = DataLoader(tok_ds, batch_size=64, shuffle=False, collate_fn=collator)

    metrics = run_eval(model, dl, device)

    payload = {
        "split": args.split,
        "model_dir": str(model_dir),
        "data_path": data_cfg.local_path,
        "seed": seed,
        "use_context": use_context,
        "max_length": max_length,
        "metrics": metrics,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved metrics to: %s", out_path)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
