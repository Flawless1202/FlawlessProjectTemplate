from typing import Optional

from ..utils import Registry, build_from_cfg


DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")


def build_dataset(cfg: dict, default_args: Optional[dict] = None):
    return build_from_cfg(cfg, DATASETS, default_args)


def build_transforms(cfg: dict, default_args: Optional[dict] = None):
    return build_from_cfg(cfg, TRANSFORMS, default_args)
