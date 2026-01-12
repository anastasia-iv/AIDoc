from __future__ import annotations
import torch
import argparse
import logging
from pathlib import Path
import pandas as pd

from med_entity_clf.predict import predict
from med_entity_clf.utils import setup_logging


def main():
    ap = argparse.ArgumentParser(description="Offline inference for medical entity classification")
    ap.add_argument("--model_dir", default="/app/artifacts/medsearch-miniLM", help="Path to saved HF model dir")
    ap.add_argument("--input_path", required=True, help="CSV with columns: text, optional context")
    ap.add_argument("--output_path", required=True, help="Where to write predictions CSV")
    ap.add_argument("--use_context", action="store_true", help="Use context column if present")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("cli_predict")

    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"input_path not found: {in_path}")

    df = pd.read_csv(in_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain column 'text'.")

    texts = df["text"].astype(str).tolist()

    contexts = None
    if args.use_context:
        if "context" in df.columns:
            contexts = df["context"].fillna("").astype(str).tolist()
        else:
            contexts = [""] * len(texts)

    logger.info("Loaded %d rows from %s", len(df), in_path)
    logger.info("Model dir: %s", args.model_dir)

    preds = predict(
        model_dir=args.model_dir,
        texts=texts,
        contexts=contexts,
        use_context=args.use_context,
        max_length=args.max_length,
    )

    out_df = pd.DataFrame(preds)
    cols = ["text", "pred_id", "pred_label", "pred_prob"]
    out_df = out_df[[c for c in cols if c in out_df.columns]]

    out_df.to_csv(out_path, index=False)
    logger.info("Saved predictions to %s", out_path)


if __name__ == "__main__":
    main()
