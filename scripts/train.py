import argparse
import yaml
import logging

from med_entity_clf.utils import setup_logging, set_seed, ensure_dir
from med_entity_clf.data import DataConfig
from med_entity_clf.modeling import HFConfig
from med_entity_clf.train import TrainConfig, train_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["project"]["log_level"])
    logger = logging.getLogger("train_cli")

    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    output_dir = args.output_dir or cfg["project"]["output_dir"]
    ensure_dir(output_dir)

    data_cfg = DataConfig(**cfg["data"])
    hf_cfg = HFConfig(**cfg["hf"])
    tr_cfg = TrainConfig(**cfg["train"])

    logger.info("Config loaded. Output: %s", output_dir)
    train_pipeline(seed=seed, output_dir=output_dir, data_cfg=data_cfg, hf_cfg=hf_cfg, tr_cfg=tr_cfg)

if __name__ == "__main__":
    main()
