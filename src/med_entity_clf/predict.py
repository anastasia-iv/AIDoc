from typing import List, Optional, Dict, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .modeling import format_input

def load_infer_artifacts(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tok, model

def predict(
    model_dir: str,
    texts: List[str],
    contexts: Optional[List[str]] = None,
    use_context: bool = True,
    max_length: int = 128,
) -> List[Dict[str, Any]]:
    tok, model = load_infer_artifacts(model_dir)
    if contexts is None:
        contexts = [""] * len(texts)

    inputs = [format_input(t, c, use_context) for t, c in zip(texts, contexts)]
    batch = tok(inputs, truncation=True, max_length=max_length, return_tensors="pt", padding=True)

    with np.errstate(over="ignore"):
        out = model(**batch).logits.detach().cpu().numpy()

    probs = np.exp(out - out.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    id2label = model.config.id2label
    preds = probs.argmax(axis=1)

    res = []
    for i in range(len(texts)):
        res.append({
            "text": texts[i],
            "pred_id": int(preds[i]),
            "pred_label": id2label[str(int(preds[i]))] if isinstance(id2label, dict) else id2label[int(preds[i])],
            "pred_prob": float(probs[i, preds[i]]),
        })
    return res

def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)
