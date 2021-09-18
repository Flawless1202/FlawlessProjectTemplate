from typing import Optional

from ..utils import Registry, build_from_cfg


METRICS = Registry("metrics")


def build_metrics(cfg: dict, default_args: Optional[dict] = None):
    return build_from_cfg(cfg, METRICS, default_args)
