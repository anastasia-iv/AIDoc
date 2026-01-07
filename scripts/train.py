import argparse
import yaml
import logging
import torch

from med_entity_clf.utils import setup_logging, set_seed, ensure_dir
from med_entity_clf.data import DataConfig
from med_entity_clf.modeling import HFConfig
from med_entity_clf.train import TrainConfig, train_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--output_dir", default=None, help="Override paths.output_dir from config")
    ap.add_argument(
        "--model_config",
        required=True,
        help="Path to model.yaml"
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]

    # logging
    log_level = cfg.get("logging", {}).get("level", "INFO")
    setup_logging(log_level)
    logger = logging.getLogger("train_cli")

    # seed
    seed = int(cfg.get("training", {}).get("random_seed", 42))
    set_seed(seed)

    # output dir
    output_dir = args.output_dir or cfg.get("paths", {}).get("output_dir", "artifacts/med_entity_clf")
    ensure_dir(output_dir)

    # data config
    data_cfg = DataConfig(
        local_path=cfg["data"]["local_path"],
        text_col=cfg["data"].get("text_col", "text"),
        context_col=cfg["data"].get("context_col", "context"),
        label_col=cfg["data"].get("label_col", "label"),
        test_size=float(cfg["data"].get("test_size", 0.1)),
        val_size=float(cfg["data"].get("val_size", 0.1)),
        max_samples=cfg["data"].get("max_samples", None),
    )

    # model config
    hf_cfg = HFConfig(
        base_model=model_cfg["base_name"],
        use_context=True,
        max_length=int(cfg["training"].get("max_length", 256)),
    )


    # training config
    tr_cfg = TrainConfig(
        lr=float(cfg["training"].get("lr", 2e-5)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
        num_train_epochs=int(cfg["training"].get("num_epochs", 2)),
        per_device_train_batch_size=int(cfg["training"].get("batch_size", 16)),
        per_device_eval_batch_size=int(cfg["training"].get("eval_batch_size", 32)),
        warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.06)),
        fp16=bool(cfg["training"].get("fp16", False)),
        eval_strategy=str(cfg["training"].get("eval_strategy", "epoch")),
        save_strategy=str(cfg["training"].get("save_strategy", "epoch")),
        metric_for_best_model=str(cfg["training"].get("metric_for_best_model", "eval_macro_f1")),
        greater_is_better=bool(cfg["training"].get("greater_is_better", True)),
    )

    # device (опционально)
    device_cfg = cfg.get("hardware", {}).get("device", "cpu")
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))

    logger.info("Config loaded from: %s", args.config)
    logger.info("Output dir: %s", output_dir)
    logger.info("Dataset: %s", data_cfg.local_path)
    logger.info("Base model: %s", hf_cfg.base_model)

    train_pipeline(
        seed=seed,
        output_dir=output_dir,
        data_cfg=data_cfg,
        hf_cfg=hf_cfg,
        tr_cfg=tr_cfg,
    )


if __name__ == "__main__":
    main()
