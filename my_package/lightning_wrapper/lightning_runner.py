from typing import List, Optional, Dict
from datetime import datetime
from glob import glob
import re
import pickle as pkl

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler


class LightningRunner(object):
    def __init__(
        self,
        model: pl.LightningModule,
        hparams: Dict,
        random_seed: int,
        work_dir: str = "./work_dirs",
        num_gpus: int = 1,
        max_epochs: int = 10,
        gradient_clip_val: float = 0.5,
        precision: int = 16,
        monitor_metrics: List[str] = None,
        checkpoint_monitor_metrics: List[str] = None,
        profiler: Optional[str] = None,
        accumulate_grad_batches: int = 1,
        check_val_every_n_epoch: int = 1,
        resume_from_checkpoint: Optional[str] = None,
        test_checkpoint: Optional[str] = None,
    ):
        self.model = model
        self.hparams = hparams
        self.work_dir = work_dir
        self.random_seed = random_seed
        self.profiler = profiler
        self.monitor_metrics = monitor_metrics
        self.checkpoint_monitor_metrics = checkpoint_monitor_metrics

        self._runner_start_time = datetime.now().strftime("%y-%m-%d-%H:%M:%S")

        print(
            f"The checkpoint file will be saved in the directory with following name:\n"
            f"\t'model_name'/'task_name'/{'-'.join(sorted(self.hparams))}/'start_time'"
        )

        self.pl_trainer = pl.Trainer(
            gpus=num_gpus,
            max_epochs=max_epochs,
            logger=self._logger,
            profiler=self._profiler,
            gradient_clip_val=gradient_clip_val,
            amp_backend="native",
            precision=precision,
            weights_summary="top",
            callbacks=[self._checkpoint_callback, self._lr_logger_callback],
            resume_from_checkpoint=resume_from_checkpoint,
            accumulate_grad_batches=accumulate_grad_batches,
            check_val_every_n_epoch=check_val_every_n_epoch,
        )

    @property
    def experiment_dir(self):
        return (
            f"{self.work_dir}/{self.model.model.__class__.__name__}_{self.model.train_dataset.task}/"
            f"{'-'.join([str(self.hparams[i]) for i in self.hparams])}/{self._runner_start_time}"
        )

    @property
    def _checkpoint_callback(self):
        _checkpoint_dir = f"{self.experiment_dir}/checkpoints"
        _checkpoint_callback = ModelCheckpoint(
            dirpath=_checkpoint_dir,
            filename=(
                f"{{epoch}}_{{avg_val_loss:.3f}}"
                f"{''.join([''.join(['_{', metrics, ':.3f}']) for metrics in self.monitor_metrics])}"
            ),
            save_last=False,
            save_top_k=-1,
            verbose=True,
            monitor=self.checkpoint_monitor_metrics[0],
            mode=self.checkpoint_monitor_metrics[1],
        )

        return _checkpoint_callback

    @property
    def _lr_logger_callback(self):
        return LearningRateMonitor(logging_interval="step")

    @property
    def _logger(self):
        _log_dir = f"{self.experiment_dir}/logs"
        _logger = TensorBoardLogger(
            save_dir=_log_dir,
            name=None,
            version=None,
            default_hp_metric=False,
        )
        return _logger

    @property
    def _profiler(self):
        if self.profiler == "simple":
            return SimpleProfiler(
                dirpath=f"{self.experiment_dir}/logs",
                filename="simple_profiler.txt",
            )
        elif self.profiler == "advance":
            return AdvancedProfiler(
                dirpath=f"{self.experiment_dir}/logs",
                filename="advanced_profiler.txt",
            )
        else:
            return None

    def train(self):
        self.pl_trainer.fit(self.model)

    def test(self, mode: str = "all", checkpoint: Optional[str] = None):
        if checkpoint is not None:
            self.pl_trainer.test(ckpt_path=checkpoint)
        else:
            checkpoint_paths = glob(f"{self.experiment_dir}/checkpoints/*.ckpt")

            if len(checkpoint_paths) == 0:
                print(
                    f"No checkpoint file found in {self.experiment_dir}/checkpoints, use original parameters instead"
                )
                self.pl_trainer.test(self.model)
                return

            checkpoint_paths = sorted(
                checkpoint_paths,
                key=lambda x: int(re.search(r"epoch=(\d+)?", x).group(1)),
            )

            if mode == "best":
                best_checkpoint_path = sorted(
                    checkpoint_paths,
                    key=lambda x: float(
                        re.search(
                            r"{}=(\d+\.\d+)?".format(
                                self.checkpoint_monitor_metrics[0]
                            ),
                            x,
                        ).group(1)
                    ),
                    reverse=self.checkpoint_monitor_metrics[1] == "max",
                )[0]

                print(f"Testing with checkpoint file '{best_checkpoint_path}' ...")

                self.pl_trainer.test(ckpt_path=best_checkpoint_path)

            elif mode == "all":
                results = {}
                for ckpt_idx, checkpoint_path in enumerate(checkpoint_paths):
                    if checkpoint_path.split("/")[-1] == "last.ckpt":
                        continue
                    epoch_idx = int(
                        re.search(r"epoch=(\d+)?", checkpoint_path).group(1)
                    )
                    print(f"Testing with checkpoint file '{checkpoint_path}' ...")

                    result = self.pl_trainer.test(ckpt_path=checkpoint_path)[0]
                    results[checkpoint_path] = result

                    with open(f"{self.experiment_dir}/logs/results.txt", "a") as f:
                        f.write(f"epoch_{epoch_idx:03d}: {checkpoint_path}\n")
                        f.write("\n".join([f"{k}: {v:.3f}" for k, v in result.items()]))
                        f.write("\n\n")

                with open(f"{self.experiment_dir}/logs/results.pkl", "wb") as f:
                    pkl.dump(results, f)

            else:
                raise NotImplementedError
