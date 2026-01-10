import pandas as pd

from datasets import Dataset

from med_entity_clf.data import make_splits, build_label_maps, DataConfig
from med_entity_clf.modeling import HFConfig, get_tokenizer
from med_entity_clf.train import tokenize_dataset


def _toy_dataset(n=20):
    df = pd.DataFrame({
        "text": [f"aspirin {i}" for i in range(n)],
        "context": [f"used to treat pain {i}" for i in range(n)],
        "label": ["T1" if i % 3 == 0 else "T2" for i in range(n)],
    })
    return Dataset.from_pandas(df, preserve_index=False)


def test_tokenize_has_required_fields():
    ds = _toy_dataset(20)
    data_cfg = DataConfig(
        local_path="",
        text_col="text",
        context_col="context",
        label_col="label",
        test_size=0.2,
        val_size=0.2,
        max_samples=None,
    )
    hf_cfg = HFConfig(base_model="distilbert-base-uncased", use_context=True, max_length=64)

    dd = make_splits(ds, test_size=0.2, val_size=0.2, seed=42)
    maps = build_label_maps(ds, data_cfg.label_col)
    label2id = maps["label2id"]

    tok = get_tokenizer(hf_cfg)
    dd_tok = tokenize_dataset(dd, tok, hf_cfg, data_cfg, label2id)

    ex = dd_tok["train"][0]
    assert "input_ids" in ex
    assert "attention_mask" in ex
    assert "labels" in ex


def test_labels_are_in_range():
    ds = _toy_dataset(30)
    data_cfg = DataConfig(
        local_path="",
        text_col="text",
        context_col="context",
        label_col="label",
        test_size=0.2,
        val_size=0.2,
        max_samples=None,
    )
    hf_cfg = HFConfig(base_model="distilbert-base-uncased", use_context=True, max_length=64)

    dd = make_splits(ds, test_size=0.2, val_size=0.2, seed=42)
    maps = build_label_maps(ds, data_cfg.label_col)
    label2id = maps["label2id"]
    num_labels = len(label2id)

    tok = get_tokenizer(hf_cfg)
    dd_tok = tokenize_dataset(dd, tok, hf_cfg, data_cfg, label2id)

    labels = dd_tok["train"]["labels"]
    assert all(isinstance(x, int) for x in labels)
    assert min(labels) >= 0
    assert max(labels) < num_labels
