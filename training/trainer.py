import logging
import math
import multiprocessing
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

if sys.platform == 'darwin':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from archs.realplksr import realplksr, realplksr_l, realplksr_s
from loggers import ExperimentLogger
from pipeline.degradations import GoldTransform
from pipeline.medallion_svc import SilverDataset
from training.config import TrainingConfig
from training.losses import LossComposer
from training.metrics import PSNR_SSIM
from training.progress import ProgressTracker, RichProgressDisplay

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self._setup_seed(config.seed)

        self.weights_dir = Path('weights')
        self.weights_dir.mkdir(exist_ok=True)

        self.model = self._build_model()
        self.model = self.model.to(self.device)
        self._log_model_size()

        self.criterion_l1 = nn.L1Loss()

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.use_amp = config.use_amp
        if self.use_amp:
            self.scaler = GradScaler()

        self.train_loader = self._build_dataloader('train')
        self.val_loader = self._build_dataloader('val') if config.val_dataset else None

        self.metrics = PSNR_SSIM(crop_border=config.crop_border)

        self.logger = ExperimentLogger(
            experiment_name=config.experiment_name,
            log_dir='logs',
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            config=config.to_dict()
        )

        self.progress_tracker = ProgressTracker(
            total_iters=config.iterations,
            log_interval=config.log_interval
        )
        self.rich_display = RichProgressDisplay()

        self.current_iter = 0
        self.current_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0

        self.loss_composer = LossComposer(config.to_dict(), device=self.device)

        if config.pretrain_path:
            self._load_pretrained(config.pretrain_path)

        if config.resume_path:
            self._resume_from_checkpoint(config.resume_path)

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple Silicon MPS")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device

    def _setup_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")

    def _build_model(self) -> nn.Module:
        model_name = self.config.model
        model_params = self.config.model_params

        if model_name == 'realplksr_s':
            model = realplksr_s(**model_params)
        elif model_name == 'realplksr_l':
            model = realplksr_l(**model_params)
        else:
            model = realplksr(**model_params)

        logger.info(f"Built model: {model_name}")
        return model

    def _log_model_size(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {num_params:,}")
        logger.info(f"Trainable parameters: {num_trainable:,}")

    def _build_optimizer(self) -> optim.Optimizer:
        opt_type = self.config.optimizer.lower()
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay

        print("\n⚙️  OPTIMIZER CONFIG:")
        print(f"  Type: {opt_type}")
        print(f"  LR: {lr}")
        print(f"  Weight decay: {weight_decay}")

        if opt_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == 'adan_sf':
            from training.optimizers.adan_sf import adan_sf
            optimizer = adan_sf(
                self.model.parameters(),
                lr=lr,
                betas=(0.98, 0.92, 0.99),
                weight_decay=weight_decay,
                warmup_steps=self.config.warmup_steps,
                schedule_free=True
            )
            print(f"  Warmup steps: {self.config.warmup_steps}")
            print("  Betas: (0.98, 0.92, 0.99)")
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

        logger.info(f"Built optimizer: {opt_type} (lr={lr}, weight_decay={weight_decay})")
        return optimizer

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        scheduler_type = self.config.scheduler
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.iterations,
                eta_min=self.config.min_lr
            )
            logger.info(f"Built cosine scheduler (T_max={self.config.iterations}, eta_min={self.config.min_lr})")
            return scheduler
        return None

    def _build_dataloader(self, split: str) -> Optional[DataLoader]:
        dataset_name = getattr(self.config, f'{split}_dataset')
        if not dataset_name:
            return None

        transform = None
        if split == 'train':
            transform = GoldTransform(
                scale=self.config.scale,
                patch_size=self.config.patch_size,
                device=self.device
            )

        dataset = SilverDataset(
            silver_dir='data/silver',
            dataset_name=dataset_name,
            transform=transform
        )

        num_workers = self.config.num_workers
        if sys.platform == 'darwin' and self.device.type == 'mps':
            logger.info("macOS MPS detected: reducing workers to 0 for stability")
            num_workers = 0

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=False,
            drop_last=(split == 'train'),
            persistent_workers=False if num_workers == 0 else True
        )

        logger.info(f"Built {split} dataloader: {len(dataset)} samples (workers={num_workers})")
        return loader

    def _load_pretrained(self, path: str):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            print(f"\n🔍 PRETRAINED CHECKPOINT KEYS: {list(checkpoint.keys())}")

            if 'params' in checkpoint:
                state_dict = checkpoint['params']
                print("✓ Using 'params' key")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("✓ Using 'model' key")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✓ Using 'state_dict' key")
            else:
                state_dict = checkpoint
                print("✓ Using checkpoint directly as state_dict")

            # Remove all DySample-related keys since we're using PixelShuffle
            keys_to_remove = [k for k in state_dict.keys() if 'to_img' in k]
            for k in keys_to_remove:
                del state_dict[k]
            print(f"\n🗑️  Removed {len(keys_to_remove)} DySample keys from pretrained weights")
            if len(keys_to_remove) > 0:
                print(f"  Removed keys: {keys_to_remove[:5]}...")

            print("\n📊 Pretrained weights sample (after DySample removal):")
            for i, (k, v) in enumerate(list(state_dict.items())[:5]):
                print(f"  {k}: {v.shape}, mean={v.mean():.4f}, std={v.std():.4f}")

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v

            # Store first conv weights for comparison
            self.pretrained_conv1 = new_state_dict.get('feats.0.weight', None)
            if self.pretrained_conv1 is not None:
                print("\n🔧 Pretrained first conv:")
                print(f"  Mean: {self.pretrained_conv1.mean():.6f}")
                print(f"  Std: {self.pretrained_conv1.std():.6f}")
                print(f"  Min: {self.pretrained_conv1.min():.6f}")
                print(f"  Max: {self.pretrained_conv1.max():.6f}")

            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print("\n⚠️  Loading results:")
            print(f"  Missing keys: {len(missing)}")
            if len(missing) > 0:
                print(f"  First 10 missing: {list(missing)[:10]}")
            print(f"  Unexpected keys: {len(unexpected)}")

            if len(unexpected) > 0:
                print("\n🔴 UNEXPECTED KEYS FOUND (should be 0 now):")
                for i, key in enumerate(unexpected):
                    print(f"  {i+1}. {key}")

            # Verify first conv layer weights after loading
            model_conv1 = self.model.feats[0].weight
            print("\n🔧 Model first conv after loading:")
            print(f"  Shape: {model_conv1.shape}")
            print(f"  Mean: {model_conv1.mean():.6f}")
            print(f"  Std: {model_conv1.std():.6f}")
            print(f"  Min: {model_conv1.min():.6f}")
            print(f"  Max: {model_conv1.max():.6f}")

            logger.info(f"Loaded pretrained weights from {path}")
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            raise

    def _resume_from_checkpoint(self, path: str):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.current_iter = checkpoint.get('iter', 0)
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_psnr = checkpoint.get('best_psnr', 0)
            self.best_ssim = checkpoint.get('best_ssim', 0)
            logger.info(f"Resumed from checkpoint {path} (iter {self.current_iter})")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_losses = []
        batches_in_epoch = 0

        for batch_idx, hr_patch in enumerate(self.train_loader):
            if self.current_iter + batches_in_epoch >= self.config.iterations:
                logger.info(f"Reached target iterations ({self.config.iterations})")
                break

            hr_patch = hr_patch.to(self.device)

            with torch.no_grad():
                if batch_idx == 0:
                    print("\n🔍 DEGRADATION PIPELINE DEBUG:")
                    print(f"  Input HR shape: {hr_patch.shape}")
                    print(f"  Input HR range: [{hr_patch.min():.3f}, {hr_patch.max():.3f}]")
                    print(f"  Input HR mean: {hr_patch.mean():.3f}")

                lr_patch, hr_original = self.train_loader.dataset.transform.degrade_pair(hr_patch)

                if batch_idx == 0:
                    print("\n  After degradation:")
                    print(f"  LR shape: {lr_patch.shape}")
                    print(f"  LR range: [{lr_patch.min():.3f}, {lr_patch.max():.3f}]")
                    print(f"  LR mean: {lr_patch.mean():.3f}")
                    print(f"  HR original shape: {hr_original.shape}")
                    print(f"  HR original range: [{hr_original.min():.3f}, {hr_original.max():.3f}]")

                # Normalize LR to [-1, 1] for model input (pretrained model expects this)
                lr_patch = lr_patch * 2 - 1
                lr_patch = lr_patch.to(self.device)

                if batch_idx == 0:
                    print("\n  LR after normalization to [-1,1]:")
                    print(f"  LR range: [{lr_patch.min():.3f}, {lr_patch.max():.3f}]")
                    print(f"  LR mean: {lr_patch.mean():.3f}")

            self.optimizer.zero_grad()
            sr_patch = self.model(lr_patch)

            if batch_idx == 0:
                print("\n🤖 MODEL OUTPUT:")
                print(f"  SR shape: {sr_patch.shape}")
                print(f"  SR range: [{sr_patch.min():.3f}, {sr_patch.max():.3f}]")
                print(f"  SR mean: {sr_patch.mean():.3f}")
                print(f"  SR std: {sr_patch.std():.3f}")

            # Convert SR from [-1,1] back to [0,1] for loss computation
            sr_patch_for_loss = (sr_patch + 1) / 2
            sr_patch_for_loss = torch.clamp(sr_patch_for_loss, 0, 1)

            if batch_idx == 0:
                print("\n  SR after conversion to [0,1] for loss:")
                print(f"  SR range: [{sr_patch_for_loss.min():.3f}, {sr_patch_for_loss.max():.3f}]")
                print(f"  SR mean: {sr_patch_for_loss.mean():.3f}")

            loss, losses = self.loss_composer(sr_patch_for_loss, hr_original)

            if batch_idx == 0:
                print("\n📉 LOSSES:")
                for name, val in losses.items():
                    print(f"  {name}: {val:.4f}")
                print(f"  Total: {loss.item():.4f}")

            loss.backward()

            if batch_idx == 0:
                grad_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                print(f"\n⚡ Gradient norm: {grad_norm:.6f}")

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            if batch_idx == 0:
                model_conv1_after = self.model.feats[0].weight
                if hasattr(self, 'pretrained_conv1') and self.pretrained_conv1 is not None:
                    change = (model_conv1_after - self.pretrained_conv1.to(self.device)).abs().mean().item()
                    print(f"\n📊 First conv weight change from pretrained after 1 step: {change:.6f}")
                    if change > 0.01:
                        print("  ⚠️  Large weight change detected! LR might be too high.")

            if self.scheduler:
                self.scheduler.step()

            epoch_losses.append(loss.item())
            batches_in_epoch += 1
            global_step = self.current_iter + batches_in_epoch

            progress_data = self.progress_tracker.update(
                batch_idx,
                loss.item(),
                self.optimizer.param_groups[0]['lr']
            )

            if progress_data:
                self.rich_display.update(progress_data)
                self.logger.log_metrics({
                    'train/loss': progress_data['loss'],
                    'train/l1': losses.get('l1', 0),
                    'train/lr': progress_data['lr'],
                    'train/speed': progress_data['speed']
                }, step=progress_data['iter'], console=False)

            if self.val_loader and global_step % self.config.val_interval == 0:
                self.rich_display.stop()
                self.validate()
                self.model.train()
                self.rich_display.start()

            if global_step > 0 and global_step % self.config.save_interval == 0:
                self.save_checkpoint()

        self.current_iter += batches_in_epoch
        return {'train_loss': np.mean(epoch_losses)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        val_losses = []

        for hr_patch in self.val_loader:
            hr_patch = hr_patch.to(self.device)

            # Create LR with bicubic downsampling
            lr_patch = torch.nn.functional.interpolate(
                hr_patch,
                scale_factor=1/self.config.scale,
                mode='bicubic',
                align_corners=False
            )

            # Normalize LR to [-1,1] for model input
            lr_patch = lr_patch * 2 - 1

            sr_patch = self.model(lr_patch)

            # Convert SR back to [0,1] for metrics
            sr_patch = (sr_patch + 1) / 2
            sr_patch = torch.clamp(sr_patch, 0, 1)

            loss = self.criterion_l1(sr_patch, hr_patch)
            val_losses.append(loss.item())
            self.metrics.update(sr_patch, hr_patch)

        results = self.metrics.get_results()
        results['loss'] = np.mean(val_losses)

        self.logger.log_metrics({
            'val/loss': results['loss'],
            'val/psnr': results['psnr'],
            'val/ssim': results['ssim']
        }, step=self.current_iter, console=False)

        if results['psnr'] > self.best_psnr:
            self.best_psnr = results['psnr']
            self.best_ssim = results['ssim']
            self.save_checkpoint(is_best=True)

        logger.info(f"Validation - PSNR: {results['psnr']:.2f} dB, SSIM: {results['ssim']:.4f}")
        return results

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'params': self.model.state_dict(),
            'iter': self.current_iter,
            'epoch': self.current_epoch,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'config': self.config.to_dict()
        }

        full_checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'iter': self.current_iter,
            'epoch': self.current_epoch,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'config': self.config.to_dict()
        }

        if not is_best:
            weights_filename = f"{self.config.experiment_name}_iter_{self.current_iter}.pth"
            weights_path = self.weights_dir / weights_filename
            torch.save(checkpoint, weights_path)
            logger.info(f"Weights saved to {weights_path}")

            full_filename = f"{self.config.experiment_name}_iter_{self.current_iter}_full.pt"
            full_path = self.weights_dir / full_filename
            torch.save(full_checkpoint, full_path)
        else:
            best_weights = self.weights_dir / f"{self.config.experiment_name}_best.pth"
            torch.save(checkpoint, best_weights)
            logger.info(f"Best model saved to {best_weights}")

            best_full = self.weights_dir / f"{self.config.experiment_name}_best_full.pt"
            torch.save(full_checkpoint, best_full)

            self.logger.save_checkpoint(full_checkpoint, "model_best.pt", is_best=True, best_metric=self.best_psnr)

    def train(self):
        logger.info("Starting training...")

        total_iters = self.config.iterations
        epochs = math.ceil(total_iters / len(self.train_loader))

        self.rich_display.start()
        self.rich_display.create_task(total_steps=total_iters)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch+1}/{epochs}")

            train_metrics = self.train_epoch()

            logger.info(f"Epoch {epoch+1} completed - Loss: {train_metrics['train_loss']:.4f}")

            if self.current_iter >= total_iters:
                break

        if self.val_loader:
            self.rich_display.stop()
            self.validate()
            self.rich_display.start()

        self.save_checkpoint()
        self.rich_display.stop()
        self.logger.close()

        logger.info("Training completed!")
