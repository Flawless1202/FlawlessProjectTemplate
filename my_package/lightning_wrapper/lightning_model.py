from typing import List, Any, Optional, Dict, Callable
import collections

import pytorch_lightning as pl
import torch
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..models import build_model
from ..datasets import build_dataset
from ..metrics import build_metrics
from ..utils.config import ConfigDict


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model_cfg: ConfigDict,
        data_cfg: ConfigDict,
        metrics_cfg: ConfigDict,
        optimizer_cfg: Optional[ConfigDict] = None,
        scheduler_cfg: Optional[ConfigDict] = None,
        warm_up_cfg: Optional[ConfigDict] = None,
        train_cfg: Optional[ConfigDict] = None,
        val_cfg: Optional[ConfigDict] = None,
        test_cfg: Optional[ConfigDict] = None,
    ):
        super().__init__()

        self.model = build_model(model_cfg)
        self.data_cfg = data_cfg
        self.train_dataset = build_dataset(data_cfg.train)
        self.val_dataset = build_dataset(data_cfg.val)
        self.test_dataset = build_dataset(data_cfg.test)
        self.metrics = build_metrics(metrics_cfg)

        self.train_cfg = dict(loss=dict())
        self.val_cfg = dict(loss=dict(), predict=dict(), target=dict())
        self.test_cfg = dict(loss=dict(), predict=dict(), target=dict())

        if train_cfg is not None:
            self.train_cfg.update(train_cfg)
        if val_cfg is not None:
            self.val_cfg.update(val_cfg)
        if test_cfg is not None:
            self.test_cfg.update(test_cfg)

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.warm_up_cfg = warm_up_cfg

    def forward(self, data: Dict[str, Any], **kwargs):
        return self.model(data, **kwargs)

    # def on_train_start(self) -> None:
    #     self.logger.log_hyperparams(self.hparams)

    def training_step(self, data: Dict[str, Any], batch_idx: int):
        loss = self.model.loss(data, **self.train_cfg["loss"])

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, data: Dict[str, Any], batch_idx: int):
        loss = self.model.loss(data, **self.val_cfg["loss"])

        pred = self.model.predict(data, **self.val_cfg["predict"])
        gt, gt_mask = self.model.get_target(data, **self.val_cfg["target"])

        logs = self.metrics.calculate(pred, gt, gt_mask)
        logs.update(val_loss=loss)

        return logs

    def test_step(self, data: Dict[str, Any], batch_idx: int):
        pred = self.model.predict(data, **self.test_cfg["predict"])
        gt, gt_mask = self.model.get_target(data, **self.test_cfg["target"])
        logs = self.metrics.calculate(pred, gt, gt_mask)

        return logs

    def validation_epoch_end(self, outputs: List[Any]):
        logs = self.metrics.summary(outputs)
        self.log_dict(logs, logger=True, prog_bar=True)

    def test_epoch_end(self, outputs: List[Any]):
        logs = self.metrics.summary(outputs)
        self.log_dict(logs, logger=True, prog_bar=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=False,
        )

    def configure_optimizers(self):
        optim_cfg = self.optimizer_cfg.copy()
        optim_class = getattr(optim, optim_cfg.pop("type"))

        if len([p for p in self.parameters()]) > 0:
            optimizer = optim_class(self.parameters(), **optim_cfg)
        else:
            return

        scheduler_cfg = self.scheduler_cfg.copy()
        lr_sheduler_class = getattr(optim.lr_scheduler, scheduler_cfg.pop("type"))
        scheduler = (
            {
                "scheduler": lr_sheduler_class(optimizer, **scheduler_cfg),
                "monitor": "avg_val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
            if optimizer is not None
            else None
        )

        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:

        warm_up_type, warm_up_step = (
            self.warm_up_cfg.type,
            self.warm_up_cfg.step_size,
        )

        if warm_up_type == "Exponential":
            lr_scale = self.model.enc_embed.d_model ** -0.5
            lr_scale *= min(
                (self.trainer.global_step + 1) ** (-0.5),
                (self.trainer.global_step + 1) * warm_up_step ** (-1.5),
            )
        elif warm_up_type == "Linear":
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warm_up_step)
        else:
            raise NotImplementedError

        for pg in optimizer.param_groups:
            pg["lr"] = lr_scale * self.optimizer_cfg.lr

        optimizer.step(closure=optimizer_closure)

    def _collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}"
        )

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem is None:
            return None
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, (str, bytes)):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self._collate([d[key] for d in batch]) for key in elem}

        raise TypeError(default_collate_err_msg_format.format(elem_type))
