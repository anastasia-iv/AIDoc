from typing import List, Optional, Dict, Any, Union
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .modeling import format_input

def _get_label(id2label, idx: int) -> str:
    if isinstance(id2label, dict):
        if idx in id2label:
            return id2label[idx]
        s = str(idx)
        if s in id2label:
            return id2label[s]
        raise KeyError(f"id2label has no key {idx} or '{idx}'. Available sample keys: {list(id2label)[:5]}")

    return id2label[idx]

def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)

def load_infer_artifacts(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tok, model

def postprocess_logits(
    logits: np.ndarray,
    id2label: Union[Dict[Any, str], List[str]],
) -> List[Dict[str, Any]]:
    """
    Convert raw logits -> probabilities -> (pred_id, pred_label, pred_prob).
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D array (batch, num_classes), got shape={logits.shape}")

    probs = logits_to_probs(logits)
    pred_ids = probs.argmax(axis=1)

    results: List[Dict[str, Any]] = []
    for i in range(probs.shape[0]):
        pred_id = int(pred_ids[i])
        results.append({
            "pred_id": pred_id,
            "pred_label": _get_label(id2label, pred_id),
            "pred_prob": float(probs[i, pred_id]),
        })
    return results

def predict(
    model_dir: str,
    texts: List[str],
    contexts: Optional[List[str]] = None,
    use_context: bool = True,
    max_length: int = 128,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    tok, model = load_infer_artifacts(model_dir)

    # device handling (optional, but nice)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if contexts is None:
        contexts = [""] * len(texts)
    if len(contexts) != len(texts):
        raise ValueError("contexts must be None or have the same length as texts")

    inputs = [format_input(t, c, use_context) for t, c in zip(texts, contexts)]
    batch = tok(inputs, truncation=True, max_length=max_length, return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(**batch).logits.detach().cpu().numpy()

    id2label = model.config.id2label
    preds = postprocess_logits(logits, id2label)

    # attach original text
    res: List[Dict[str, Any]] = []
    for i in range(len(texts)):
        res.append({
            "text": texts[i],
            **preds[i],
        })
    return res