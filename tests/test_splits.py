import pandas as pd

from datasets import Dataset

from med_entity_clf.data import make_splits


def _toy_dataset(n=50):
    df = pd.DataFrame({
        "text": [f"t{i}" for i in range(n)],
        "context": [f"c{i}" for i in range(n)],
        "label": ["A" if i % 2 == 0 else "B" for i in range(n)],
    })
    return Dataset.from_pandas(df, preserve_index=False)


def test_splits_non_empty():
    ds = _toy_dataset(50)
    dd = make_splits(ds, test_size=0.2, val_size=0.2, seed=42)

    assert len(dd["train"]) > 0
    assert len(dd["validation"]) > 0
    assert len(dd["test"]) > 0


def test_splits_total_size_matches():
    ds = _toy_dataset(50)
    dd = make_splits(ds, test_size=0.2, val_size=0.2, seed=42)

    total = len(dd["train"]) + len(dd["validation"]) + len(dd["test"])
    assert total == len(ds)


def test_split_reproducibility_with_fixed_seed():
    ds = _toy_dataset(50)
    dd1 = make_splits(ds, test_size=0.2, val_size=0.2, seed=42)
    dd2 = make_splits(ds, test_size=0.2, val_size=0.2, seed=42)

    train_texts_1 = dd1["train"]["text"]
    train_texts_2 = dd2["train"]["text"]
    assert train_texts_1 == train_texts_2
