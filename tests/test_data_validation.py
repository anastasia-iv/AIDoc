import pandas as pd
import pytest

from med_entity_clf.data import validate_dataframe, DataValidationError


def test_missing_label_column_raises():
    df = pd.DataFrame({"text": ["abc"], "context": ["ctx"]})
    with pytest.raises(DataValidationError):
        validate_dataframe(df, text_col="text", label_col="label", context_col="context")


def test_nan_rows_are_dropped():
    df = pd.DataFrame({
        "text": ["a", None, "c"],
        "context": ["x", "y", None],
        "label": ["L1", "L2", None],
    })
    out = validate_dataframe(
        df,
        text_col="text",
        label_col="label",
        context_col="context",
        dropna=True,
        min_rows=1,
    )
    assert len(out) == 1
    assert out.iloc[0]["text"] == "a"
    assert out.iloc[0]["label"] == "L1"


def test_empty_after_cleaning_raises():
    df = pd.DataFrame({
        "text": [None, ""],
        "context": ["x", "y"],
        "label": [None, ""],
    })
    with pytest.raises(DataValidationError):
        validate_dataframe(
            df,
            text_col="text",
            label_col="label",
            context_col="context",
            dropna=True,
            min_rows=1,
        )
