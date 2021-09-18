import os
import random
import argparse
from shutil import copy

import torch
import numpy as np

from my_package.lightning_wrapper import LightningModel, LightningRunner
from my_package.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train or test a detector.")
    parser.add_argument("config", help="Train config file path.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    return args


def setup_random_seed(self):
    torch.manual_seed(self.random_seed)
    torch.cuda.manual_seed(self.random_seed)
    torch.cuda.manual_seed_all(self.random_seed)
    np.random.seed(self.random_seed)
    random.seed(self.random_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    train_cfg = cfg.train_cfg if hasattr(cfg, "train_cfg") else None
    val_cfg = cfg.val_cfg if hasattr(cfg, "val_cfg") else None
    test_cfg = cfg.test_cfg if hasattr(cfg, "test_cfg") else None

    runner = LightningRunner(
        LightningModel(
            model_cfg=cfg.model,
            data_cfg=cfg.data,
            metrics_cfg=cfg.metrics,
            optimizer_cfg=cfg.optimizer_cfg,
            scheduler_cfg=cfg.scheduler_cfg,
            warm_up_cfg=cfg.warm_up_cfg,
            train_cfg=train_cfg,
            val_cfg=val_cfg,
            test_cfg=test_cfg,
        ),
        hparams=cfg.hparams,
        **cfg.runner_cfg,
    )

    os.makedirs(f"{runner.experiment_dir}/logs", exist_ok=True)
    copy(
        args.config,
        os.path.join(f"{runner.experiment_dir}/logs", args.config.split("/")[-1]),
    )

    runner.train()
    runner.test(mode="all")


if __name__ == "__main__":
    main()
