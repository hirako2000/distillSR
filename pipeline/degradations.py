# pipeline/degradations.py
"""
Second-Order Degradation Engine for Gold Tier processing
Simulates "Internet damage" through recursive degradation pipeline
"""

import random
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import special
from scipy.signal import convolve2d


class DegradationEngine:
    """
    Second-Order Stochastic Degradation Pipeline
    
    Simulates real-world image damage through:
    - Multiple blur types (Gaussian, anisotropic, sinc)
    - Multi-interpolation resizing
    - Noise injection (Gaussian, Poisson)
    - JPEG compression with chroma subsampling
    """

    def __init__(
        self,
        scale: int = 4,
        device: torch.device = None
    ):
        self.scale = scale
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Blur kernels
        self.blur_types = ['iso_gaussian', 'aniso_gaussian', 'generalized_gaussian', 'plateau']

        # Interpolation methods for resizing
        self.interp_methods = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }

    def random_blur(
        self,
        img: np.ndarray,
        kernel_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply random blur from stochastic selection
        
        Args:
            img: Input image [H, W, C] or [H, W]
            kernel_size: Blur kernel size (odd)
        """
        h, w = img.shape[:2]

        # Random kernel size (odd)
        if kernel_size is None:
            kernel_size = random.choice([7, 9, 11, 13, 15, 17, 19, 21])

        # Ensure odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Randomly select blur type
        blur_type = random.choice(self.blur_types)

        if blur_type == 'iso_gaussian':
            # Isotropic Gaussian (same sigma in all directions)
            sigma = random.uniform(0.5, 4.0)
            kernel = self._gaussian_kernel(kernel_size, sigma)

        elif blur_type == 'aniso_gaussian':
            # Anisotropic Gaussian (different sigma per axis)
            sigma_x = random.uniform(0.5, 6.0)
            sigma_y = random.uniform(0.5, 6.0)
            kernel = self._anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y)

        elif blur_type == 'generalized_gaussian':
            # Generalized Gaussian (adjustable shape)
            sigma = random.uniform(0.5, 4.0)
            beta = random.uniform(0.5, 2.0)  # Shape parameter
            kernel = self._generalized_gaussian_kernel(kernel_size, sigma, beta)

        elif blur_type == 'plateau':
            # Plateau blur (flat center)
            sigma = random.uniform(1.0, 5.0)
            plateau_ratio = random.uniform(0.3, 0.7)
            kernel = self._plateau_kernel(kernel_size, sigma, plateau_ratio)

        # Normalize kernel
        kernel = kernel / kernel.sum()

        # Apply blur
        if len(img.shape) == 3:
            # Color image
            blurred = np.zeros_like(img)
            for c in range(img.shape[2]):
                blurred[:, :, c] = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        else:
            # Grayscale
            blurred = convolve2d(img, kernel, mode='same', boundary='symm')

        return blurred.astype(np.float32)

    def _gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Create isotropic Gaussian kernel"""
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel

    def _anisotropic_gaussian_kernel(self, size: int, sigma_x: float, sigma_y: float) -> np.ndarray:
        """Create anisotropic Gaussian kernel"""
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) / np.square(sigma_x) + np.square(yy) / np.square(sigma_y)))
        return kernel

    def _generalized_gaussian_kernel(self, size: int, sigma: float, beta: float) -> np.ndarray:
        """Create generalized Gaussian kernel"""
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        r = np.sqrt(np.square(xx) + np.square(yy))
        kernel = np.exp(-np.power(np.abs(r) / sigma, beta))
        return kernel

    def _plateau_kernel(self, size: int, sigma: float, plateau_ratio: float) -> np.ndarray:
        """Create plateau blur kernel (flat center)"""
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        r = np.sqrt(np.square(xx) + np.square(yy))

        # Flat center, gaussian falloff
        plateau_r = sigma * plateau_ratio
        kernel = np.where(r <= plateau_r, 1.0, np.exp(-0.5 * np.square((r - plateau_r) / sigma)))
        return kernel

    def sinc_filter(
        self,
        img: np.ndarray,
        kernel_size: int = 21,
        cutoff: float = 0.7  # Cutoff frequency (0-1)
    ) -> np.ndarray:
        """
        Apply sinc filter to simulate ringing artifacts
        
        Args:
            img: Input image
            kernel_size: Size of sinc kernel (odd)
            cutoff: Cutoff frequency (lower = more ringing)
        """
        # Create sinc kernel
        kernel = self._sinc_kernel(kernel_size, cutoff)
        kernel = kernel / kernel.sum()

        # Apply
        if len(img.shape) == 3:
            filtered = np.zeros_like(img)
            for c in range(img.shape[2]):
                filtered[:, :, c] = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        else:
            filtered = convolve2d(img, kernel, mode='same', boundary='symm')

        return filtered.astype(np.float32)

    def _sinc_kernel(self, size: int, cutoff: float) -> np.ndarray:
        """Create 2D sinc filter kernel"""
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        r = np.sqrt(np.square(xx) + np.square(yy))

        # 2D sinc = J1(2πr) / r
        with np.errstate(divide='ignore', invalid='ignore'):
            kernel = np.where(r == 0, 1.0, special.j1(2 * np.pi * cutoff * r) / (np.pi * r))

        # Apply Lanczos window to reduce ringing
        window = np.sinc(r / size) * np.sinc(r / size)
        kernel = kernel * window

        return kernel

    def random_resize(
        self,
        img: np.ndarray,
        scale_factor: Optional[float] = None,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Resize image with random interpolation method
        
        Args:
            img: Input image
            scale_factor: Resize factor (<1 for downsampling, >1 for upsampling)
            method: Specific interpolation method (if None, random)
        """
        h, w = img.shape[:2]

        # Random scale factor if not provided
        if scale_factor is None:
            # For degradation, typically downsample
            scale_factor = random.uniform(0.25, 1.0)

        # Random interpolation method
        if method is None:
            method = random.choice(list(self.interp_methods.keys()))

        interp = self.interp_methods[method]

        # Calculate new dimensions
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        # Ensure minimum size
        new_h = max(8, new_h)
        new_w = max(8, new_w)

        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        return resized

    def add_noise(
        self,
        img: np.ndarray,
        noise_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Add random noise to image
        
        Args:
            img: Input image [0, 1] range
            noise_type: 'gaussian', 'poisson', or None for random
        """
        if noise_type is None:
            noise_type = random.choice(['gaussian', 'poisson'])

        # Ensure image is float and within [0, 1]
        img_float = img.astype(np.float32)
        img_float = np.clip(img_float, 0, 1)  # Clip to valid range

        if noise_type == 'gaussian':
            # Gaussian noise (sensor hiss)
            sigma = random.uniform(0.01, 0.1)
            noise = np.random.normal(0, sigma, img.shape)
            noisy = img_float + noise

        elif noise_type == 'poisson':
            # Poisson (shot) noise
            # to reasonable photon counts
            scale = random.uniform(10, 100)

            # Ensure no negative values for Poisson
            # Poisson requires lambda >= 0
            scaled = img_float * scale
            scaled = np.maximum(scaled, 0)

            # Handle potential NaN
            if np.any(np.isnan(scaled)):
                logger.warning("NaNs detected in Poisson input, using zeros") # noqa: F821
                scaled = np.zeros_like(scaled)

            try:
                noisy = np.random.poisson(scaled).astype(np.float32) / scale
            except ValueError as e:
                logger.warning(f"Poisson noise failed: {e}, falling back to Gaussian") # noqa: F821
                sigma = random.uniform(0.01, 0.05)
                noise = np.random.normal(0, sigma, img.shape)
                noisy = img_float + noise

        else:
            noisy = img_float

        noisy = np.clip(noisy, 0, 1)

        return noisy.astype(np.float32)

    def jpeg_compress(
        self,
        img: np.ndarray,
        quality: Optional[int] = None,
        subsample: bool = True
    ) -> np.ndarray:
        """
        Apply JPEG compression with optional chroma subsampling
        
        Args:
            img: Input image [0, 1] range
            quality: JPEG quality (1-100)
            subsample: Apply chroma subsampling (4:2:0)
        """
        if quality is None:
            # Random quality based on degradation level
            quality = random.choice([
                random.randint(10, 30),   # Heavy
                random.randint(31, 60),   # Medium
                random.randint(61, 85),   # Light
                random.randint(86, 98)    # Minimal
            ])

        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)

        # Apply chroma subsampling if requested
        if subsample and random.random() > 0.5:
            # Convert to YCrCb
            if len(img_uint8.shape) == 3:
                ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)

                # Subsample chroma channels (4:2:0)
                h, w = ycrcb.shape[:2]
                ycrcb[:, :, 1] = cv2.resize(
                    cv2.resize(ycrcb[:, :, 1], (w//2, h//2), cv2.INTER_AREA),
                    (w, h), cv2.INTER_LINEAR
                )
                ycrcb[:, :, 2] = cv2.resize(
                    cv2.resize(ycrcb[:, :, 2], (w//2, h//2), cv2.INTER_AREA),
                    (w, h), cv2.INTER_LINEAR
                )

                # Convert back
                img_uint8 = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        # JPEG encode/decode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        # Back to float
        return decoded.astype(np.float32) / 255.0


class SecondOrderDegradation:
    """
    Two-pass degradation pipeline simulating multi-generational damage
    
    First pass: D1 (original encoding)
    Second pass: D2 (re-upload/screenshot damage)
    """

    def __init__(
        self,
        scale: int = 4,
        device: torch.device = None
    ):
        self.engine = DegradationEngine(scale, device)
        self.scale = scale

    def degrade(
        self,
        clean_img: Union[np.ndarray, torch.Tensor],
        return_intermediate: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply second-order degradation to clean image
        
        Args:
            clean_img: Clean HR image [0, 1] range
            return_intermediate: Return both D1 and D2 results
            
        Returns:
            Degraded LR image (and optionally intermediate)
        """
        # Convert to numpy if tensor
        if isinstance(clean_img, torch.Tensor):
            clean_img = clean_img.cpu().numpy().transpose(1, 2, 0)

        # Ensure float [0, 1]
        if clean_img.max() > 1:
            clean_img = clean_img / 255.0

        # --- First Degradation (D1) - Original encoding ---
        img = clean_img.copy()

        # 1. Random blur
        if random.random() > 0.2:  # 80% chance of blur
            img = self.engine.random_blur(img)

        # 2. First resize (downsample)
        d1_scale = random.uniform(0.5, 1.0)
        img = self.engine.random_resize(img, scale_factor=d1_scale)

        # 3. Add noise
        if random.random() > 0.3:
            img = self.engine.add_noise(img)

        # 4. First JPEG compression (higher quality - original encoding)
        q1 = random.randint(75, 95)
        img = self.engine.jpeg_compress(img, quality=q1, subsample=True)

        d1_result = img.copy()

        # --- Second Degradation (D2) - Re-upload damage ---

        # 1. Sinc filter (ringing artifacts)
        if random.random() > 0.4:
            img = self.engine.sinc_filter(
                img,
                kernel_size=random.choice([15, 21]),
                cutoff=random.uniform(0.5, 0.8)
            )

        # 2. Second blur (different type)
        if random.random() > 0.3:
            img = self.engine.random_blur(img)

        # 3. Second resize (final downsampling to target scale)
        target_h = clean_img.shape[0] // self.scale
        target_w = clean_img.shape[1] // self.scale
        current_h, current_w = img.shape[:2]

        scale_factor = min(target_h / current_h, target_w / current_w)
        img = self.engine.random_resize(img, scale_factor=scale_factor)

        # 4. Second noise injection
        if random.random() > 0.5:
            img = self.engine.add_noise(img)

        # 5. Final JPEG compression (lower quality - re-upload)
        q2 = random.randint(30, 85)
        img = self.engine.jpeg_compress(img, quality=q2, subsample=True)

        if return_intermediate:
            return img, d1_result
        else:
            return img

class GoldTransform:
    """
    PyTorch transform for on-the-fly Gold degradation
    Off the Silver Dataset
    """

    def __init__(
        self,
        scale: int = 4,
        patch_size: int = 256,
        device: torch.device = None
    ):
        self.scale = scale
        self.patch_size = patch_size
        self.degradation = SecondOrderDegradation(scale, device)

    def __call__(self, clean_patch: torch.Tensor) -> torch.Tensor:
        """
        Apply degradation to clean patch and return LR-HR pair
        
        Args:
            clean_patch: Clean HR patch [C, H, W] in [0, 1] range
            
        Returns:
            clean_patch (the original HR image) - the dataloader expects just the HR image
            The degradation is applied in the training loop
        """
        # degradation will be applied in training loop
        return clean_patch


    def degrade_pair(self, clean_patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate LR-HR pair for training
        This is called explicitly in the training loop
        
        Args:
            clean_patch: Clean HR patch [B, C, H, W] or [C, H, W] in [0, 1] range
            
        Returns:
            Tuple of (degraded LR, clean HR) with same batch dimension
        """
        # Handle batch dimension
        if clean_patch.dim() == 4:
            # Batch of images
            batch_size = clean_patch.shape[0]
            lr_patches = []

            for i in range(batch_size):
                lr, _ = self._degrade_single(clean_patch[i])
                lr_patches.append(lr)

            lr_batch = torch.stack(lr_patches)
            return lr_batch, clean_patch

        elif clean_patch.dim() == 3:
            # Single image
            return self._degrade_single(clean_patch)

        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {clean_patch.dim()}D")

    def _degrade_single(self, clean_patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Degrade a single image [C, H, W]
        """
        # to ensure clean_patch is in [0, 1] range
        if clean_patch.max() > 1.0 or clean_patch.min() < 0:
            logger.warning(f"Input patch out of range: min={clean_patch.min()}, max={clean_patch.max()}, clipping to [0,1]") # noqa: F821
            clean_patch = torch.clamp(clean_patch, 0, 1)

        # to numpy [H, W, C]
        img_np = clean_patch.permute(1, 2, 0).cpu().numpy()

        lr_np = self.degradation.degrade(img_np)

        lr_h = self.patch_size // self.scale
        lr_w = self.patch_size // self.scale

        if lr_np.shape[:2] != (lr_h, lr_w):
            lr_np = cv2.resize(lr_np, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

        # Convert back to tensor and ensure valid range
        lr_tensor = torch.from_numpy(lr_np).permute(2, 0, 1).float()
        lr_tensor = torch.clamp(lr_tensor, 0, 1)

        return lr_tensor, clean_patch


def test_degradation():
    """Test the degradation pipeline"""

    # Create test image
    test_img = np.random.rand(512, 512, 3).astype(np.float32)

    # Add some patterns
    test_img[100:200, 100:200] = 0.8
    test_img[300:400, 300:400] = 0.2

    # Apply degradation
    deg = SecondOrderDegradation(scale=4)
    lr, intermediate = deg.degrade(test_img, return_intermediate=True)

    print(f"Original shape: {test_img.shape}")
    print(f"Intermediate shape: {intermediate.shape}")
    print(f"LR shape: {lr.shape}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_img)
    axes[0].set_title('Original')
    axes[1].imshow(intermediate)
    axes[1].set_title('After D1')
    axes[2].imshow(lr)
    axes[2].set_title('Final LR')
    plt.show()


if __name__ == "__main__":
    test_degradation()
