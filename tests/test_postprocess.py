import numpy as np

from med_entity_clf.predict import logits_to_probs, postprocess_logits


def test_logits_to_probs_valid():
    logits = np.array([
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    probs = logits_to_probs(logits)

    assert probs.shape == logits.shape
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_postprocess_logits_returns_correct_label_str_keys():
    logits = np.array([[0.1, 0.2, 2.0]])  # argmax = 2
    id2label = {"0": "A", "1": "B", "2": "C"}

    out = postprocess_logits(logits, id2label)
    assert len(out) == 1
    assert out[0]["pred_id"] == 2
    assert out[0]["pred_label"] == "C"
    assert 0.0 <= out[0]["pred_prob"] <= 1.0


def test_postprocess_logits_returns_correct_label_int_keys():
    logits = np.array([[3.0, 0.1]])  # argmax = 0
    id2label = {0: "X", 1: "Y"}

    out = postprocess_logits(logits, id2label)
    assert out[0]["pred_id"] == 0
    assert out[0]["pred_label"] == "X"
    assert 0.0 <= out[0]["pred_prob"] <= 1.0
