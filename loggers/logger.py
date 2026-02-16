import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentLogger:
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        use_wandb: bool = False,
        wandb_project: str = "RealPLKSR",
        wandb_entity: Optional[str] = None,
        config: Optional[Dict] = None,
        resume_id: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.start_time = time.time()
        self.step = 0
        self.epoch = 0

        self.metrics_history = []
        self.best_metrics = {}

        if use_wandb:
            try:
                import wandb
                self.wandb = wandb

                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    config=config,
                    id=resume_id,
                    resume="allow" if resume_id else None,
                    dir=str(self.log_dir)
                )
                logger.info(f"Initialized W&B run: {wandb.run.name}")

            except ImportError:
                logger.warning("wandb not installed, falling back to local logging")
                self.use_wandb = False
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

        if config:
            self.save_config(config)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        commit: bool = True,
        console: bool = False
    ):
        if step is not None:
            self.step = step
        if epoch is not None:
            self.epoch = epoch

        log_data = {
            'step': self.step,
            'epoch': self.epoch,
            'timestamp': time.time() - self.start_time,
            **metrics
        }

        self.metrics_history.append(log_data)

        if self.use_wandb:
            self.wandb.log(metrics, step=self.step, commit=commit)

        if console:
            metric_str = ' - '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Step {self.step} [Epoch {self.epoch}]: {metric_str}")

    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: Optional[int] = None,
        caption: Optional[str] = None
    ):
        if not self.use_wandb:
            return

        if step is not None:
            self.step = step

        wandb_images = {}
        for name, tensor in images.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() == 4:
                    tensor = tensor[0]

                if tensor.dim() == 3:
                    if tensor.shape[0] in [1, 3]:
                        tensor = tensor.permute(1, 2, 0)
                    tensor = tensor.detach().cpu().numpy()

                if tensor.max() <= 1.0:
                    tensor = (tensor * 255).astype(np.uint8)
                else:
                    tensor = tensor.astype(np.uint8)

                wandb_images[name] = self.wandb.Image(
                    tensor,
                    caption=caption or f"{name} at step {self.step}"
                )

        self.wandb.log(wandb_images, step=self.step)

    def log_histogram(
        self,
        name: str,
        data: Union[torch.Tensor, np.ndarray, List],
        step: Optional[int] = None
    ):
        if not self.use_wandb:
            return

        if step is not None:
            self.step = step

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif isinstance(data, list):
            data = np.array(data)

        self.wandb.log({
            name: self.wandb.Histogram(data)
        }, step=self.step)

    def log_model_graph(self, model, input_tensor):
        if not self.use_wandb:
            return

        try:
            self.wandb.watch(model, log="gradients", log_freq=100)
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")

    def save_checkpoint(
        self,
        state: Dict,
        filename: str,
        is_best: bool = False,
        best_metric: Optional[float] = None
    ):
        path = self.log_dir / filename
        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}")

        if is_best:
            best_path = self.log_dir / "model_best.pt"
            torch.save(state, best_path)
            logger.info(f"Best model saved to {best_path}")

            if best_metric is not None:
                self.best_metrics = {
                    'value': best_metric,
                    'step': self.step,
                    'epoch': self.epoch
                }
                self.save_best_metrics()

        if self.use_wandb:
            artifact = self.wandb.Artifact(
                f"{self.experiment_name}-checkpoint",
                type="model"
            )
            artifact.add_file(str(path))
            self.wandb.log_artifact(artifact)

    def save_config(self, config: Dict):
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Config saved to {config_path}")

    def save_best_metrics(self):
        metrics_path = self.log_dir / "best_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)

    def get_progress(self) -> Dict:
        return {
            'experiment': self.experiment_name,
            'step': self.step,
            'epoch': self.epoch,
            'elapsed_hours': (time.time() - self.start_time) / 3600,
            'best_metrics': self.best_metrics
        }

    def close(self):
        self.save_best_metrics()

        if self.use_wandb:
            self.wandb.finish()

        logger.info(f"Experiment {self.experiment_name} completed")
        logger.info(f"Logs saved to {self.log_dir}")


class AverageMeter:
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressLogger:
    def __init__(self, total_steps: int, log_interval: int = 100):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time = time.time()
        self.meters = {}

    def add_meter(self, name: str, fmt: str = ':f'):
        self.meters[name] = AverageMeter(name, fmt)

    def update(self, step: int, **kwargs):
        for name, value in kwargs.items():
            if name in self.meters:
                self.meters[name].update(value)

        if step % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            eta = elapsed / step * (self.total_steps - step) if step > 0 else 0

            progress = (step / self.total_steps) * 100
            log_str = f"[{step}/{self.total_steps} {progress:.1f}%] "
            log_str += f"Elapsed: {elapsed/3600:.1f}h ETA: {eta/3600:.1f}h | "

            meter_strs = [str(m) for m in self.meters.values()]
            log_str += ' | '.join(meter_strs)

            logger.info(log_str)

    def get_averages(self) -> Dict:
        return {name: meter.avg for name, meter in self.meters.items()}


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level=logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    return logger


def test_logger():
    exp_logger = ExperimentLogger(
        experiment_name="test_run",
        use_wandb=False,
        config={"learning_rate": 1e-4, "batch_size": 8}
    )

    for i in range(10):
        exp_logger.log_metrics({
            "loss": 0.5 / (i + 1),
            "psnr": 20 + i
        }, step=i, epoch=i//5, console=True)

    exp_logger.save_checkpoint(
        {"epoch": 5, "state_dict": {}},
        "checkpoint_5.pt"
    )

    exp_logger.close()

    progress = ProgressLogger(total_steps=1000, log_interval=100)
    progress.add_meter("loss")
    progress.add_meter("psnr")

    for i in range(1000):
        progress.update(i, loss=0.5/(i+1), psnr=20 + i%10)


if __name__ == "__main__":
    test_logger()
