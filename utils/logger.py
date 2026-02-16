# utils/logger.py
"""
Logging utilities for training
Supports Weights & Biases and local logging
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
import numpy as np

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Main logger for training experiments"""
    
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
        """
        Args:
            experiment_name: Name of this experiment
            log_dir: Directory for local logs
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
            wandb_entity: W&B entity/username
            config: Configuration dictionary to log
            resume_id: W&B run ID to resume
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.start_time = time.time()
        self.step = 0
        self.epoch = 0
        
        # Metrics storage
        self.metrics_history = []
        self.best_metrics = {}
        
        # Initialize W&B
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                
                # Initialize run
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
        
        # Save config locally
        if config:
            self.save_config(config)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        commit: bool = True
    ):
        """Log metrics"""
        if step is not None:
            self.step = step
        if epoch is not None:
            self.epoch = epoch
        
        # Add timestamp and step info
        log_data = {
            'step': self.step,
            'epoch': self.epoch,
            'timestamp': time.time() - self.start_time,
            **metrics
        }
        
        # Store in history
        self.metrics_history.append(log_data)
        
        # Log to W&B
        if self.use_wandb:
            self.wandb.log(metrics, step=self.step, commit=commit)
        
        # Log to console
        metric_str = ' - '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {self.step} [Epoch {self.epoch}]: {metric_str}")
    
    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: Optional[int] = None,
        caption: Optional[str] = None
    ):
        """Log images to W&B"""
        if not self.use_wandb:
            return
        
        if step is not None:
            self.step = step
        
        wandb_images = {}
        for name, tensor in images.items():
            # Convert to numpy and ensure correct format
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() == 4:
                    # Batch of images - take first
                    tensor = tensor[0]
                
                # Convert to [H, W, C] numpy
                if tensor.dim() == 3:
                    if tensor.shape[0] in [1, 3]:
                        tensor = tensor.permute(1, 2, 0)
                    tensor = tensor.detach().cpu().numpy()
                
                # Ensure [0, 255] range
                if tensor.max() <= 1.0:
                    tensor = (tensor * 255).astype(np.uint8)
                else:
                    tensor = tensor.astype(np.uint8)
                
                # Log to W&B
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
        """Log histogram"""
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
        """Log model architecture graph"""
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
        """Save checkpoint"""
        # Save regular checkpoint
        path = self.log_dir / filename
        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}")
        
        # Save best model if applicable
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
        
        # Save as W&B artifact
        if self.use_wandb:
            artifact = self.wandb.Artifact(
                f"{self.experiment_name}-checkpoint",
                type="model"
            )
            artifact.add_file(str(path))
            self.wandb.log_artifact(artifact)
    
    def save_config(self, config: Dict):
        """Save configuration to local file"""
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Config saved to {config_path}")
    
    def save_best_metrics(self):
        """Save best metrics to file"""
        metrics_path = self.log_dir / "best_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
    
    def get_progress(self) -> Dict:
        """Get current progress summary"""
        return {
            'experiment': self.experiment_name,
            'step': self.step,
            'epoch': self.epoch,
            'elapsed_hours': (time.time() - self.start_time) / 3600,
            'best_metrics': self.best_metrics
        }
    
    def close(self):
        """Clean up logger"""
        # Save final metrics
        self.save_best_metrics()
        
        # Close W&B
        if self.use_wandb:
            self.wandb.finish()
        
        logger.info(f"Experiment {self.experiment_name} completed")
        logger.info(f"Logs saved to {self.log_dir}")


class AverageMeter:
    """Computes and stores the average and current value"""
    
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
    """Simple progress logger for training loops"""
    
    def __init__(self, total_steps: int, log_interval: int = 100):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time = time.time()
        self.meters = {}
    
    def add_meter(self, name: str, fmt: str = ':f'):
        """Add a meter to track"""
        self.meters[name] = AverageMeter(name, fmt)
    
    def update(self, step: int, **kwargs):
        """Update meters and log if interval reached"""
        # Update meters
        for name, value in kwargs.items():
            if name in self.meters:
                self.meters[name].update(value)
        
        # Log progress
        if step % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            eta = elapsed / step * (self.total_steps - step) if step > 0 else 0
            
            # Build progress string
            progress = (step / self.total_steps) * 100
            log_str = f"[{step}/{self.total_steps} {progress:.1f}%] "
            log_str += f"Elapsed: {elapsed/3600:.1f}h ETA: {eta/3600:.1f}h | "
            
            # Add meter values
            meter_strs = [str(m) for m in self.meters.values()]
            log_str += ' | '.join(meter_strs)
            
            logger.info(log_str)
    
    def get_averages(self) -> Dict:
        """Get average values from all meters"""
        return {name: meter.avg for name, meter in self.meters.items()}


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level=logging.INFO
) -> logging.Logger:
    """Setup a logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)
    
    # File handler
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
    """Test logger functionality"""
    # Create logger
    exp_logger = ExperimentLogger(
        experiment_name="test_run",
        use_wandb=False,
        config={"learning_rate": 1e-4, "batch_size": 8}
    )
    
    # Log some metrics
    for i in range(10):
        exp_logger.log_metrics({
            "loss": 0.5 / (i + 1),
            "psnr": 20 + i
        }, step=i, epoch=i//5)
    
    # Save dummy checkpoint
    exp_logger.save_checkpoint(
        {"epoch": 5, "state_dict": {}},
        "checkpoint_5.pt"
    )
    
    # Close
    exp_logger.close()
    
    # Test progress logger
    progress = ProgressLogger(total_steps=1000, log_interval=100)
    progress.add_meter("loss")
    progress.add_meter("psnr")
    
    for i in range(1000):
        progress.update(i, loss=0.5/(i+1), psnr=20 + i%10)


if __name__ == "__main__":
    test_logger()