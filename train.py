# train.py
"""
Main training script for RealPLKSR
Supports MPS (Apple Silicon) and CUDA training with second-order degradation
"""

import os
import sys
import argparse
import yaml
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import random
import numpy as np
import multiprocessing
from rich.console import Console

# Set multiprocessing start method for macOS
if sys.platform == 'darwin':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from archs.realplksr import realplksr, realplksr_s, realplksr_l
from pipeline.medallion_svc import SilverDataset
from pipeline.degradations import GoldTransform
from utils.metrics import PSNR_SSIM, Metrics
from utils.logger import ExperimentLogger, ProgressLogger, setup_logger

# Configure logging
logger = setup_logger('train', log_file='logs/train.log')


class Trainer:
    """RealPLKSR Trainer with second-order degradation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = self._setup_device()
        self._setup_seed(config.get('seed', 42))
        
        # Create directories
        self.weights_dir = Path('weights')
        self.weights_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # Log model size
        self._log_model_size()
        
        # Loss functions
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 = nn.MSELoss()
        
        # Optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Data loaders
        self.train_loader = self._build_dataloader('train')
        self.val_loader = self._build_dataloader('val') if config.get('val_dataset') else None
        
        # Metrics
        self.metrics = PSNR_SSIM(crop_border=config.get('crop_border', 4))
        
        # Logger
        self.logger = ExperimentLogger(
            experiment_name=config['experiment_name'],
            log_dir='logs',
            use_wandb=config.get('use_wandb', False),
            wandb_project=config.get('wandb_project', 'RealPLKSR'),
            config=config
        )
        
        # Training state
        self.current_iter = 0
        self.current_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        
        # Load pretrained if specified
        if config.get('pretrain_path'):
            self._load_pretrained(config['pretrain_path'])
        
        # Resume from checkpoint if specified
        if config.get('resume_path'):
            self._resume_from_checkpoint(config['resume_path'])
    
    def _setup_device(self) -> torch.device:
        """Setup compute device (MPS/CUDA/CPU)"""
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
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")
    
    def _build_model(self) -> nn.Module:
        """Build model from config"""
        model_name = self.config.get('model', 'realplksr')
        model_params = self.config.get('model_params', {})
        
        if model_name == 'realplksr_s':
            model = realplksr_s(**model_params)
        elif model_name == 'realplksr_l':
            model = realplksr_l(**model_params)
        else:
            model = realplksr(**model_params)
        
        logger.info(f"Built model: {model_name}")
        return model
    
    def _log_model_size(self):
        """Log model parameter count"""
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {num_params:,}")
        logger.info(f"Trainable parameters: {num_trainable:,}")
        
        # Log to wandb only if logger is initialized and wants metrics
        try:
            self.logger.log_metrics({
                'total_params': num_params,
                'trainable_params': num_trainable
            }, step=0)
        except (AttributeError, TypeError) as e:
            # Logger might not be fully initialized yet, that's ok
            logger.debug(f"Could not log metrics to wandb: {e}")
            pass
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config"""
        opt_type = self.config.get('optimizer', 'AdamW')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Convert to float if they're strings
        if isinstance(lr, str):
            try:
                lr = float(lr)
            except ValueError:
                logger.error(f"Could not convert learning_rate '{lr}' to float")
                raise
        
        if isinstance(weight_decay, str):
            try:
                weight_decay = float(weight_decay)
            except ValueError:
                logger.error(f"Could not convert weight_decay '{weight_decay}' to float")
                raise
        
        logger.info(f"Using learning_rate: {lr} (type: {type(lr)})")
        logger.info(f"Using weight_decay: {weight_decay} (type: {type(weight_decay)})")
        
        if opt_type.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")
        
        logger.info(f"Built optimizer: {opt_type} (lr={lr}, weight_decay={weight_decay})")
        return optimizer
    

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        # Convert eta_min to float if it's a string
        eta_min = self.config.get('min_lr', 1e-7)
        if isinstance(eta_min, str):
            try:
                eta_min = float(eta_min)
            except ValueError:
                logger.error(f"Could not convert min_lr '{eta_min}' to float")
                raise
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('iterations', 500000),
                eta_min=eta_min
            )
            logger.info(f"Built cosine scheduler (T_max={self.config.get('iterations')}, eta_min={eta_min})")
        else:
            scheduler = None
        
        return scheduler
    
    def _build_dataloader(self, split: str) -> DataLoader:
        """Build dataloader for training or validation"""
        dataset_name = self.config.get(f'{split}_dataset')
        if not dataset_name:
            return None
        
        # Create transform for training
        if split == 'train':
            transform = GoldTransform(
                scale=self.config.get('scale', 4),
                patch_size=self.config.get('patch_size', 256),
                device=self.device
            )
        else:
            transform = None
        
        # Create dataset
        dataset = SilverDataset(
            silver_dir='data/silver',
            dataset_name=dataset_name,
            transform=transform
        )
        
        # Determine optimal number of workers based on platform
        num_workers = self.config.get('num_workers', 4)
        
        # On macOS with MPS, multiprocessing can be unstable with LMDB
        if sys.platform == 'darwin' and self.device.type == 'mps':
            logger.info("macOS MPS detected: reducing workers to 0 for stability")
            num_workers = 0
        
        # Create dataloader with platform-specific settings
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=False,  # Disable pin_memory on MPS (not supported)
            drop_last=(split == 'train'),
            persistent_workers=False if num_workers == 0 else True
        )
        
        logger.info(f"Built {split} dataloader: {len(dataset)} samples (workers={num_workers})")
        return loader
    
    def _load_pretrained(self, path: str):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Loaded pretrained weights from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
    
    def _resume_from_checkpoint(self, path: str):
        """Resume training from checkpoint"""
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
    
    def _compute_loss(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training losses"""
        losses = {}
        
        # L1 loss
        loss_l1 = self.criterion_l1(sr, hr)
        losses['l1'] = loss_l1.item()
        
        # L2 loss (optional)
        if self.config.get('use_l2', False):
            loss_l2 = self.criterion_l2(sr, hr)
            losses['l2'] = loss_l2.item()
        
        # MS-SSIM loss (optional)
        if self.config.get('use_ms_ssim', False):
            try:
                from pytorch_msssim import ms_ssim
                loss_ms_ssim = 1 - ms_ssim(sr, hr, data_range=1.0)
                losses['ms_ssim'] = loss_ms_ssim.item()
            except ImportError:
                logger.warning("pytorch_msssim not installed, skipping MS-SSIM loss")
                loss_ms_ssim = 0
        
        # Perceptual loss (optional)
        if self.config.get('use_perceptual', False):
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
                if not hasattr(self, 'lpips'):
                    self.lpips = LearnedPerceptualImagePatchSimilarity(
                        net_type='vgg',
                        normalize=True
                    ).to(self.device)
                
                # Normalize to [-1, 1] for LPIPS
                sr_norm = sr * 2 - 1
                hr_norm = hr * 2 - 1
                
                loss_perceptual = self.lpips(sr_norm, hr_norm).mean()
                losses['perceptual'] = loss_perceptual.item()
            except ImportError:
                logger.warning("torchmetrics not installed, skipping perceptual loss")
                loss_perceptual = 0
        
        # Combine losses with weights
        total_loss = loss_l1 * self.config.get('weight_l1', 1.0)
        
        if self.config.get('use_l2', False):
            total_loss += loss_l2 * self.config.get('weight_l2', 0.1)
        
        if self.config.get('use_ms_ssim', False) and 'loss_ms_ssim' in locals():
            total_loss += loss_ms_ssim * self.config.get('weight_ms_ssim', 0.2)
        
        if self.config.get('use_perceptual', False) and 'loss_perceptual' in locals():
            total_loss += loss_perceptual * self.config.get('weight_perceptual', 0.1)
        
        return total_loss, losses
    

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Add a simple timer for updates
        last_log_time = time.time()
        log_interval = self.config.get('log_interval', 100)
        
        epoch_losses = []
        
        for batch_idx, hr_patch in enumerate(self.train_loader):
            # Move to device
            hr_patch = hr_patch.to(self.device)
            
            # Apply degradation (Gold transform) to create LR-HR pair
            with torch.no_grad():
                lr_patch, hr_patch = self.train_loader.dataset.transform.degrade_pair(hr_patch)
                lr_patch = lr_patch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            sr_patch = self.model(lr_patch)
            loss, losses = self._compute_loss(sr_patch, hr_patch)
            
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update scheduler per batch
            if self.scheduler:
                self.scheduler.step()
            
            epoch_losses.append(loss.item())
            
            # Log progress every N batches OR every 10 seconds
            current_time = time.time()
            if batch_idx % log_interval == 0 or current_time - last_log_time > 10:
                # Calculate progress
                progress = (batch_idx + 1) / len(self.train_loader) * 100
                elapsed = current_time - self.logger.start_time
                iterations_per_sec = (self.current_iter + batch_idx + 1) / max(elapsed, 1e-8)
                
                # Estimate time remaining
                remaining_iters = self.config.get('iterations', 500000) - (self.current_iter + batch_idx + 1)
                eta_seconds = remaining_iters / max(iterations_per_sec, 1e-8)
                
                # Simple progress log
                logger.info(
                    f"⏱️  Batch {batch_idx+1}/{len(self.train_loader)} ({progress:.1f}%) | "
                    f"Loss: {loss.item():.4f} | "
                    f"Speed: {iterations_per_sec:.1f} it/s | "
                    f"ETA: {eta_seconds/3600:.1f}h"
                )
                
                # Log to wandb
                self.logger.log_metrics({
                    'train/loss': loss.item(),
                    'train/l1': losses['l1'],
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/speed': iterations_per_sec
                }, step=self.current_iter + batch_idx)
                
                last_log_time = current_time
            
            # Validation
            if self.val_loader and (
                self.current_iter + batch_idx + 1
            ) % self.config.get('val_interval', 5000) == 0:
                self.validate()
                self.model.train()
            
            # Save checkpoint
            if (
                self.current_iter + batch_idx + 1
            ) % self.config.get('save_interval', 5000) == 0:
                self.save_checkpoint()
        
        self.current_iter += len(self.train_loader)
        
        return {'train_loss': np.mean(epoch_losses)}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        self.metrics.reset()
        
        val_losses = []
        
        for hr_patch in self.val_loader:
            hr_patch = hr_patch.to(self.device)
            
            # Simple bicubic downsampling for validation
            lr_patch = torch.nn.functional.interpolate(
                hr_patch,
                scale_factor=1/self.config.get('scale', 4),
                mode='bicubic',
                align_corners=False
            )
            
            # Forward pass
            sr_patch = self.model(lr_patch)
            
            # Compute loss
            loss = self.criterion_l1(sr_patch, hr_patch)
            val_losses.append(loss.item())
            
            # Update metrics
            self.metrics.update(sr_patch, hr_patch)
        
        # Get results
        results = self.metrics.get_results()
        results['loss'] = np.mean(val_losses)
        
        # Log to wandb
        self.logger.log_metrics({
            'val/loss': results['loss'],
            'val/psnr': results['psnr'],
            'val/ssim': results['ssim']
        }, step=self.current_iter)
        
        # Log sample images
        if self.config.get('log_images', False):
            self.logger.log_images({
                'val/lr': lr_patch[0],
                'val/sr': sr_patch[0],
                'val/hr': hr_patch[0]
            }, step=self.current_iter)
        
        # Save best model
        if results['psnr'] > self.best_psnr:
            self.best_psnr = results['psnr']
            self.best_ssim = results['ssim']
            self.save_checkpoint(is_best=True)
        
        logger.info(
            f"Validation - PSNR: {results['psnr']:.2f} dB, "
            f"SSIM: {results['ssim']:.4f}, "
            f"Loss: {results['loss']:.4f}"
        )
        
        return results
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'iter': self.current_iter,
            'epoch': self.current_epoch,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'config': self.config
        }
        
        # Save regular checkpoint
        if not is_best:
            filename = f"checkpoint_iter_{self.current_iter}.pt"
            self.logger.save_checkpoint(
                checkpoint,
                filename,
                is_best=False
            )
        else:
            self.logger.save_checkpoint(
                checkpoint,
                "model_best.pt",
                is_best=True,
                best_metric=self.best_psnr
            )
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        total_iters = self.config.get('iterations', 500000)
        epochs = math.ceil(total_iters / len(self.train_loader))
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Show epoch progress (only once)
            epoch_progress = (epoch + 1) / epochs * 100
            logger.info(f"📊 Epoch {epoch+1}/{epochs} ({epoch_progress:.1f}% complete)")
            
            # Train one epoch
            train_metrics = self.train_epoch()
            
            logger.info(
                f"✅ Epoch {epoch+1}/{epochs} completed - "
                f"Loss: {train_metrics['train_loss']:.4f}"
            )
            
            # Check if reached target iterations
            if self.current_iter >= total_iters:
                break
        
        # Final validation and save
        if self.val_loader:
            self.validate()
        
        self.save_checkpoint()
        self.logger.close()
        
        logger.info("🎉 Training completed!")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train RealPLKSR")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--pretrain', type=str,
                       help='Path to pretrained weights')
    parser.add_argument('--device', type=str,
                       help='Override device (cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.resume:
        config['resume_path'] = args.resume
    if args.pretrain:
        config['pretrain_path'] = args.pretrain
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()