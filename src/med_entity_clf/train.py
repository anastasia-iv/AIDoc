from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

from .data import DataConfig, load_local_dataset, make_splits, build_label_maps
from .modeling import HFConfig, get_tokenizer, get_model, format_input

logger = logging.getLogger("med_entity_clf")

@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    warmup_ratio: float
    fp16: bool
    eval_strategy: str
    save_strategy: str
    metric_for_best_model: str
    greater_is_better: bool

def tokenize_dataset(dd, tokenizer, hf_cfg: HFConfig, data_cfg: DataConfig, label2id: Dict[str, int]):
    text_col = data_cfg.text_col
    ctx_col = data_cfg.context_col
    label_col = data_cfg.label_col

    def _map(ex):
        context = ex[ctx_col] if (ctx_col is not None and ctx_col in ex) else None
        x = format_input(ex[text_col], context, hf_cfg.use_context)
        tok = tokenizer(x, truncation=True, max_length=hf_cfg.max_length)
        tok["labels"] = label2id[ex[label_col]]
        return tok

    return dd.map(_map, remove_columns=dd["train"].column_names)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    out = {}
    out["accuracy"] = acc.compute(predictions=preds, references=labels)["accuracy"]
    out["macro_f1"] = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    return out

def train_pipeline(
    seed: int,
    output_dir: str,
    data_cfg: DataConfig,
    hf_cfg: HFConfig,
    tr_cfg: TrainConfig,
):
    ds = load_local_dataset(data_cfg)
    dd = make_splits(ds, data_cfg.test_size, data_cfg.val_size, seed)

    maps = build_label_maps(ds, data_cfg.label_col)
    label2id, id2label = maps["label2id"], maps["id2label"]
    logger.info("Num labels: %d", len(label2id))

    tokenizer = get_tokenizer(hf_cfg)
    dd_tok = tokenize_dataset(dd, tokenizer, hf_cfg, data_cfg, label2id)

    model = get_model(hf_cfg, num_labels=len(label2id), id2label=id2label, label2id=label2id)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=tr_cfg.lr,
        weight_decay=tr_cfg.weight_decay,
        num_train_epochs=tr_cfg.num_train_epochs,
        per_device_train_batch_size=tr_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=tr_cfg.per_device_eval_batch_size,
        warmup_ratio=tr_cfg.warmup_ratio,
        fp16=tr_cfg.fp16,
        evaluation_strategy=tr_cfg.eval_strategy,
        save_strategy=tr_cfg.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model=tr_cfg.metric_for_best_model.replace("eval_", ""),
        greater_is_better=tr_cfg.greater_is_better,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dd_tok["train"],
        eval_dataset=dd_tok["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Start training...")
    trainer.train()

    logger.info("Final eval on test...")
    test_metrics = trainer.evaluate(dd_tok["test"])
    logger.info("TEST metrics: %s", test_metrics)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {"label2id": label2id, "id2label": id2label, "test_metrics": test_metrics}
