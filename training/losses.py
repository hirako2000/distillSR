import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torch import nn

logger = logging.getLogger(__name__)

_LOSS_REGISTRY = {}

def register_loss(name: str):
    def decorator(cls):
        _LOSS_REGISTRY[name.lower()] = cls
        return cls
    return decorator

class BaseLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

@register_loss('l1')
class L1Loss(BaseLoss):
    def forward(self, sr, hr):
        return F.l1_loss(sr, hr) * self.weight

@register_loss('l2')
class L2Loss(BaseLoss):
    def forward(self, sr, hr):
        return F.mse_loss(sr, hr) * self.weight

@register_loss('ms_ssim')
class MSSSIMLoss(BaseLoss):
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        self.ms_ssim = ms_ssim

    def forward(self, sr, hr):
        return (1 - self.ms_ssim(sr, hr, data_range=1.0)) * self.weight

@register_loss('consistency')
class ConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 1.0, device='cpu'):
        super().__init__()
        self.weight = weight
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)

    def rgb_to_luminance(self, img):
        return 0.2126 * img[:, 0:1, :, :] + 0.7152 * img[:, 1:2, :, :] + 0.0722 * img[:, 2:3, :, :]

    def forward(self, sr, hr):
        sr_lum = self.rgb_to_luminance(sr)
        hr_lum = self.rgb_to_luminance(hr)

        luma_loss = F.l1_loss(sr_lum, hr_lum)

        sr_norm = F.normalize(sr, dim=1)
        hr_norm = F.normalize(hr, dim=1)
        cosim = 1 - self.similarity(sr_norm, hr_norm).mean()

        return (luma_loss + 0.5 * cosim) * self.weight

@register_loss('ldl')
class LDLLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, sr, hr):
        grad_sr_x = torch.abs(sr[:, :, :, 1:] - sr[:, :, :, :-1])
        grad_sr_y = torch.abs(sr[:, :, 1:, :] - sr[:, :, :-1, :])
        grad_hr_x = torch.abs(hr[:, :, :, 1:] - hr[:, :, :, :-1])
        grad_hr_y = torch.abs(hr[:, :, 1:, :] - hr[:, :, :-1, :])

        loss_x = F.l1_loss(grad_sr_x, grad_hr_x)
        loss_y = F.l1_loss(grad_sr_y, grad_hr_y)

        return (loss_x + loss_y) * self.weight

@register_loss('fdl')
class FDLLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, sr, hr):
        sr_fft = torch.fft.rfft2(sr, norm='ortho')
        hr_fft = torch.fft.rfft2(hr, norm='ortho')

        sr_mag = torch.abs(sr_fft)
        hr_mag = torch.abs(hr_fft)

        return F.l1_loss(sr_mag, hr_mag) * self.weight

class LossComposer(nn.Module):
    def __init__(self, config: Dict[str, Any], device='cpu'):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.losses['l1'] = L1Loss(weight=config.get('weight_l1', 1.0))

        if config.get('use_ms_ssim', False):
            self.losses['ms_ssim'] = MSSSIMLoss(weight=config.get('weight_ms_ssim', 1.0))

        if config.get('use_consistency', False):
            self.losses['consistency'] = ConsistencyLoss(
                weight=config.get('weight_consistency', 1.0),
                device=device
            )

        if config.get('use_ldl', False):
            self.losses['ldl'] = LDLLoss(weight=config.get('weight_ldl', 1.0))

        if config.get('use_fdl', False):
            self.losses['fdl'] = FDLLoss(weight=config.get('weight_fdl', 0.75))

    def forward(self, sr, hr):
        losses = {}
        total = 0

        for name, loss_fn in self.losses.items():
            val = loss_fn(sr, hr)
            losses[name] = val.item()
            total += val

        return total, losses
