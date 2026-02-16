import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    experiment_name: str = "default_experiment"
    model: str = "realplksr"
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "in_ch": 3,
        "out_ch": 3,
        "dim": 64,
        "n_blocks": 28,
        "kernel_size": 17,
        "split_ratio": 0.25,
        "use_ea": True,
        "norm_groups": 4,
        "dropout": 0.1,
        "dysample": False
    })
    scale: int = 4
    iterations: int = 500000
    batch_size: int = 8
    patch_size: int = 256
    num_workers: int = 4
    seed: int = 42
    train_dataset: Optional[str] = None
    val_dataset: Optional[str] = None
    optimizer: str = "adan_sf"
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    min_lr: float = 1e-7
    warmup_steps: int = 1600  # ADD THIS LINE
    weight_l1: float = 1.0
    weight_l2: float = 0.1
    weight_ms_ssim: float = 1.0
    weight_consistency: float = 1.0
    weight_ldl: float = 1.0
    weight_fdl: float = 0.75
    use_l2: bool = False
    use_ms_ssim: bool = True
    use_consistency: bool = True
    use_ldl: bool = True
    use_fdl: bool = True
    use_amp: bool = False
    grad_clip: float = 1.0
    crop_border: int = 4
    use_wandb: bool = False
    wandb_project: str = "RealPLKSR"
    log_interval: int = 100
    save_interval: int = 5000
    val_interval: int = 5000
    log_images: bool = True
    pretrain_path: Optional[str] = None
    resume_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
