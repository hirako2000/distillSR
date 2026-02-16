# utils/metrics.py
"""
Evaluation metrics for super-resolution
PSNR, SSIM, LPIPS calculations
"""

import math
from typing import Union

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_skimage


class Metrics:
    """Collection of image quality metrics"""

    @staticmethod
    def psnr(
        img1: Union[np.ndarray, torch.Tensor],
        img2: Union[np.ndarray, torch.Tensor],
        max_val: float = 1.0
    ) -> float:
        """
        Peak Signal-to-Noise Ratio
        
        Args:
            img1, img2: Input images (same shape) [0, max_val]
            max_val: Maximum pixel value
        """
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')

        return 20 * math.log10(max_val / math.sqrt(mse))

    @staticmethod
    def psnr_torch(
        img1: torch.Tensor,
        img2: torch.Tensor,
        max_val: float = 1.0
    ) -> torch.Tensor:
        """PyTorch version of PSNR"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'), device=img1.device)

        return 20 * torch.log10(max_val / torch.sqrt(mse))

    @staticmethod
    def ssim(
        img1: Union[np.ndarray, torch.Tensor],
        img2: Union[np.ndarray, torch.Tensor],
        max_val: float = 1.0,
        win_size: int = 11,
        multichannel: bool = True
    ) -> float:
        """
        Structural Similarity Index
        
        Args:
            img1, img2: Input images
            max_val: Maximum pixel value
            win_size: Window size for SSIM
            multichannel: If True, treat last dimension as channels
        """
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()

        # Convert to [H, W, C] if needed
        if img1.shape[0] == 3 and len(img1.shape) == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))

        # Ensure float and correct range
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # Convert to uint8 range if needed
        if max_val <= 1.0:
            img1 = (img1 * 255).clip(0, 255).astype(np.uint8)
            img2 = (img2 * 255).clip(0, 255).astype(np.uint8)
            data_range = 255
        else:
            data_range = max_val

        # Compute SSIM
        if multichannel and len(img1.shape) == 3:
            # Multichannel SSIM
            ssim_val = ssim_skimage(
                img1, img2,
                win_size=win_size,
                data_range=data_range,
                channel_axis=-1
            )
        else:
            # Grayscale SSIM
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

            ssim_val = ssim_skimage(
                img1, img2,
                win_size=win_size,
                data_range=data_range
            )

        return float(ssim_val)

    @staticmethod
    def lpips(
        img1: torch.Tensor,
        img2: torch.Tensor,
        net: str = 'alex'
    ) -> torch.Tensor:
        """
        Learned Perceptual Image Patch Similarity
        Requires LPIPS package: pip install lpips
        
        Args:
            img1, img2: Input tensors [B, C, H, W] in [-1, 1] range
            net: Backbone network ('alex', 'vgg', 'squeeze')
        """
        try:
            import lpips
        except ImportError:
            raise ImportError("Please install lpips: pip install lpips")

        # Initialize LPIPS model if not already done
        if not hasattr(Metrics, '_lpips_model'):
            Metrics._lpips_model = lpips.LPIPS(net=net)
            if torch.cuda.is_available():
                Metrics._lpips_model = Metrics._lpips_model.cuda()

        # Normalize to [-1, 1] if needed
        if img1.max() <= 1.0:
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1

        # Ensure 4D
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        with torch.no_grad():
            dist = Metrics._lpips_model(img1, img2)

        return dist.squeeze()

    @staticmethod
    def niqe(img: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Natural Image Quality Evaluator
        Requires piq package: pip install piq
        
        Args:
            img: Input image [0, 255] range
        """
        try:
            import piq
        except ImportError:
            raise ImportError("Please install piq: pip install piq")

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        # Convert to [0, 255] uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Ensure [H, W, C]
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        # Convert to tensor for piq
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            score = piq.niqe(img_tensor, data_range=255)

        return float(score)

    @staticmethod
    def calculate_all(
        sr: Union[np.ndarray, torch.Tensor],
        hr: Union[np.ndarray, torch.Tensor],
        max_val: float = 1.0
    ) -> dict:
        """
        Calculate all metrics
        
        Returns:
            Dictionary with PSNR, SSIM, LPIPS
        """
        metrics = {}

        # PSNR
        metrics['psnr'] = Metrics.psnr(sr, hr, max_val)

        # SSIM
        metrics['ssim'] = Metrics.ssim(sr, hr, max_val)

        # LPIPS (if both are tensors)
        if isinstance(sr, torch.Tensor) and isinstance(hr, torch.Tensor):
            try:
                metrics['lpips'] = float(Metrics.lpips(sr, hr))
            except:
                metrics['lpips'] = float('nan')

        return metrics


class PSNR_SSIM:
    """Wrapper for validation loop"""

    def __init__(self, crop_border: int = 4):
        """
        Args:
            crop_border: Number of pixels to crop from border
        """
        self.crop_border = crop_border
        self.reset()

    def reset(self):
        """Reset accumulated values"""
        self.psnr_list = []
        self.ssim_list = []

    def update(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        max_val: float = 1.0
    ):
        """
        Update metrics with new batch
        
        Args:
            sr: Super-resolved image [B, C, H, W]
            hr: Ground truth [B, C, H, W]
        """
        sr_np = sr.detach().cpu().numpy()
        hr_np = hr.detach().cpu().numpy()

        for i in range(sr_np.shape[0]):
            # Crop border if needed
            if self.crop_border > 0:
                sr_i = sr_np[i, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
                hr_i = hr_np[i, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            else:
                sr_i = sr_np[i]
                hr_i = hr_np[i]

            # Compute metrics
            psnr = Metrics.psnr(sr_i, hr_i, max_val)
            ssim = Metrics.ssim(sr_i, hr_i, max_val)

            self.psnr_list.append(psnr)
            self.ssim_list.append(ssim)

    def get_results(self) -> dict:
        """Get average results"""
        return {
            'psnr': np.mean(self.psnr_list) if self.psnr_list else 0,
            'ssim': np.mean(self.ssim_list) if self.ssim_list else 0,
            'psnr_std': np.std(self.psnr_list) if self.psnr_list else 0,
            'ssim_std': np.std(self.ssim_list) if self.ssim_list else 0,
            'count': len(self.psnr_list)
        }


def test_metrics():
    """Test metrics functions"""
    # Create test images
    hr = torch.rand(3, 256, 256) * 0.8 + 0.1
    sr = hr + torch.randn_like(hr) * 0.05
    sr = sr.clamp(0, 1)

    # Calculate metrics
    metrics = Metrics.calculate_all(sr, hr)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"LPIPS: {metrics['lpips']:.4f}")

    # Test accumulator
    acc = PSNR_SSIM(crop_border=4)
    acc.update(sr.unsqueeze(0), hr.unsqueeze(0))
    results = acc.get_results()
    print(f"\nAccumulated: {results}")


if __name__ == "__main__":
    test_metrics()
