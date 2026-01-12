import argparse
import logging
from pathlib import Path

import yaml

from med_entity_clf.utils import setup_logging, set_seed, ensure_dir, resolve_path
from med_entity_clf.data import DataConfig
from med_entity_clf.modeling import HFConfig
from med_entity_clf.train import TrainConfig, train_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--output_dir", default=None, help="Override output dir from config")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["logging"]["level"])
    logger = logging.getLogger("train_cli")

    seed = int(cfg["training"]["random_seed"])
    set_seed(seed)

    base_dir = cfg_path.parent.parent 

    output_dir = args.output_dir or cfg["paths"]["output_dir"]
    output_dir = resolve_path(output_dir, base_dir=base_dir)
    ensure_dir(str(output_dir))

    data_cfg_dict = dict(cfg["data"])
    data_cfg_dict["local_path"] = str(resolve_path(data_cfg_dict["local_path"], base_dir=base_dir))

    paths_dict = dict(cfg["paths"])
    if "cache_dir" in paths_dict and paths_dict["cache_dir"]:
        paths_dict["cache_dir"] = str(resolve_path(paths_dict["cache_dir"], base_dir=base_dir))

    data_cfg = DataConfig(**data_cfg_dict)
    hf_cfg = HFConfig(**cfg["hf"])
    tr_cfg = TrainConfig(**cfg["training"])

    logger.info("Config loaded: %s", cfg_path)
    logger.info("Output dir: %s", output_dir)

    train_pipeline(
        seed=seed,
        output_dir=str(output_dir),
        data_cfg=data_cfg,
        hf_cfg=hf_cfg,
        tr_cfg=tr_cfg,
        cache_dir=paths_dict.get("cache_dir"),
        device=cfg.get("hardware", {}).get("device", "cpu"),
    )


if __name__ == "__main__":
    main()
