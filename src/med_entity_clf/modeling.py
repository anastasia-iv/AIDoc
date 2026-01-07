from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class HFConfig:
    base_model: str
    use_context: bool
    max_length: int

def get_tokenizer(cfg: HFConfig):
    return AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)

def get_model(cfg: HFConfig, num_labels: int, id2label: dict, label2id: dict):
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

def format_input(text: str, context: Optional[str], use_context: bool) -> str:
    if use_context and context is not None and len(context) > 0:
        return f"{text} [SEP] {context}"
    return text
