import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import Module

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, weights_dir: Union[str, Path] = "weights"):
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(exist_ok=True)

    def save(self, state: Dict[str, Any], filename: str, is_best: bool = False, best_metric: Optional[float] = None):
        path = self.weights_dir / filename
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")

        if is_best:
            best_path = self.weights_dir / "model_best.pt"
            torch.save(state, best_path)
            if best_metric:
                logger.info(f"Best model saved (metric: {best_metric:.4f})")

    def load(self, path: str, model: Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional = None, strict: bool = False) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=strict)

        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        return checkpoint
