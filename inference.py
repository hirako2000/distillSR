# inference.py
"""
Tiled inference for high-resolution images
Handles 4K/8K images with memory-efficient tiling strategy
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union
import math
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from archs.realplksr import realplksr, realplksr_s, realplksr_l
from utils.metrics import Metrics


class TiledInference:
    """
    Memory-efficient tiled inference for large images
    Uses simple overlapping tiles with linear blending
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        tile_size: int = 512,
        halo_size: int = 32,
        scale: int = 4
    ):
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.halo_size = halo_size
        self.scale = scale
        self.effective_tile = tile_size + 2 * halo_size
        
    @torch.no_grad()
    def infer(self, lr_image: np.ndarray) -> np.ndarray:
        """Run inference on input image following neosr's approach"""
        self.model.eval()
        
        # Prepare input tensor - neosr uses img2tensor with bgr2rgb=True
        # This converts BGR (OpenCV) to RGB and normalizes to [0, 1]
        if lr_image.max() > 1.0:
            lr_image = lr_image.astype(np.float32) / 255.0
        
        # Convert to RGB if needed (OpenCV loads as BGR)
        if lr_image.shape[-1] == 3:
            # Assuming input is BGR from OpenCV
            lr_image_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        else:
            lr_image_rgb = lr_image
        
        # Convert to tensor [C, H, W] and add batch dimension
        img_tensor = torch.from_numpy(lr_image_rgb).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        h, w = img_tensor.shape[2:]
        
        # Pad to tile size if needed
        pad_h = max(0, self.effective_tile - h)
        pad_w = max(0, self.effective_tile - w)
        if pad_h > 0 or pad_w > 0:
            img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        else:
            img_padded = img_tensor
        
        # Run model
        sr_full = self.model(img_padded)
        
        # Remove padding
        sr_full = sr_full[:, :, :h*self.scale, :w*self.scale]
        
        print(f"Raw output range: [{sr_full.min():.3f}, {sr_full.max():.3f}]")
        
        # Convert to numpy following neosr's tensor2img approach
        # They clamp to min_max (default 0,1) and normalize
        min_val, max_val = 0, 1
        sr_np = sr_full.squeeze(0).cpu().clamp(min_val, max_val)
        
        # Normalize to [0, 1] (though it should already be there after clamp)
        sr_np = (sr_np - min_val) / (max_val - min_val)
        
        # Convert to numpy and permute to HWC
        sr_np = sr_np.permute(1, 2, 0).numpy()
        
        # Scale to 0-255 and convert to uint8
        sr_np = (sr_np * 255.0).round().astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV saving (if needed)
        # The model outputs RGB, but OpenCV saves as BGR
        sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
        
        return sr_bgr
    
    @torch.no_grad()
    def infer_old(self, lr_image: np.ndarray) -> np.ndarray:
        """Run inference on input image with proper halo handling"""
        self.model.eval()
        
        # Prepare input tensor
        if lr_image.max() > 1.0:
            lr_image = lr_image.astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(lr_image).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        h, w = img_tensor.shape[2:]
        
        # For small images, just run the model directly without tiling
        if h <= self.effective_tile and w <= self.effective_tile:
            print(f"Image smaller than tile, running direct inference...")
            
            # Pad to tile size if needed
            pad_h = max(0, self.effective_tile - h)
            pad_w = max(0, self.effective_tile - w)
            if pad_h > 0 or pad_w > 0:
                img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
            else:
                img_padded = img_tensor
            
            # Run model
            sr_full = self.model(img_padded)
            
            # Remove padding
            #sr_full = sr_full[:, :, :h*self.scale, :w*self.scale]
            
            # Normalize output
            #sr_full = torch.tanh(sr_full)
            #sr_full = (sr_full + 1) / 2
            #sr_full = sr_full.clamp(0, 1)
            
            # Convert to numpy
            sr_np = sr_full.squeeze(0).cpu().permute(1, 2, 0).numpy()
            sr_np = (sr_np * 255).clip(0, 255).astype(np.uint8)
            
            return sr_np
        
        # For larger images, use tiling
        print(f"Image larger than tile, using tiling...")
        
        # Pad image
        pad_h = (self.effective_tile - (h % self.effective_tile)) % self.effective_tile
        pad_w = (self.effective_tile - (w % self.effective_tile)) % self.effective_tile
        img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        
        padded_h, padded_w = img_padded.shape[2:]
        n_h = (padded_h - self.halo_size * 2) // self.tile_size
        n_w = (padded_w - self.halo_size * 2) // self.tile_size
        
        out_h = h * self.scale
        out_w = w * self.scale
        
        # Initialize output accumulator
        output = torch.zeros(3, out_h, out_w, device=self.device)
        weight = torch.ones(3, out_h, out_w, device=self.device) * 1e-8
        
        print(f"Processing {n_h}x{n_w} tiles...")
        
        for i in range(n_h):
            for j in range(n_w):
                # Calculate tile boundaries with halo
                y_start = i * self.tile_size
                x_start = j * self.tile_size
                y_end = y_start + self.effective_tile
                x_end = x_start + self.effective_tile
                
                # Extract tile
                tile = img_padded[:, :, y_start:y_end, x_start:x_end]
                
                # Run model
                sr_tile = self.model(tile)
                
                # Normalize
                sr_tile = torch.tanh(sr_tile)
                sr_tile = (sr_tile + 1) / 2
                sr_tile = sr_tile.clamp(0, 1)
                sr_tile = sr_tile.squeeze(0)
                
                # Calculate output boundaries (without halo)
                out_y_start = y_start * self.scale
                out_x_start = x_start * self.scale
                out_y_end = min((y_start + self.tile_size) * self.scale, out_h)
                out_x_end = min((x_start + self.tile_size) * self.scale, out_w)
                
                # Calculate tile region (without halo)
                tile_y_start = self.halo_size * self.scale
                tile_x_start = self.halo_size * self.scale
                tile_y_end = tile_y_start + (out_y_end - out_y_start)
                tile_x_end = tile_x_start + (out_x_end - out_x_start)
                
                # Add to output (no blending weights for now to debug)
                output[:, out_y_start:out_y_end, out_x_start:out_x_end] += \
                    sr_tile[:, tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                weight[:, out_y_start:out_y_end, out_x_start:out_x_end] += 1
        
        # Average overlapping regions
        output = output / weight
        
        # Convert to numpy
        output_np = output.cpu().permute(1, 2, 0).numpy()
        output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
        
        return output_np

class InferenceEngine:
    """Main inference engine with multiple input/output options"""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        scale: int = 4,
        tile_size: int = 512,
        halo_size: int = 32,
        model_type: str = 'realplksr'
    ):
        """
        Args:
            model_path: Path to model checkpoint
            device: Compute device (cuda/mps/cpu)
            scale: Upscaling factor
            tile_size: Tile size for inference
            halo_size: Halo size for context
            model_type: Model architecture
        """
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path, model_type, scale)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create tiled inference
        self.tiler = TiledInference(
            model=self.model,
            device=self.device,
            tile_size=tile_size,
            halo_size=halo_size,
            scale=scale
        )
        
        self.scale = scale
    
    def _load_model(
        self,
        model_path: str,
        model_type: str,
        scale: int
    ) -> nn.Module:
        """Load model from checkpoint with MPS-optimized DySample and debugging"""
        from archs.realplksr import realplksr
        import traceback
        
        # Create model with dysample=True to match pretrained weights
        model = realplksr(
            in_ch=3,
            out_ch=3,
            dim=64,
            n_blocks=28,
            upscaling_factor=scale,
            kernel_size=17,
            split_ratio=0.25,
            use_ea=True,
            norm_groups=4,
            dropout=0.1,
            dysample=True  # Must be True for pretrained
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"\n📊 Loading results:")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        # Move model to device
        model = model.to(self.device)
        
        # MPS Optimization: Patch the DySample forward method
        if self.device.type == 'mps':
            print("🔧 MPS detected: Optimizing DySample for MPS...")
            
            # Define MPS-compatible forward method with extensive debugging
            def mps_safe_forward(self, x):
                print(f"\n  🔍 DySample forward debug:")
                print(f"    Input shape: {x.shape}")
                print(f"    Device: {x.device}")
                
                try:
                    # Step 1: Compute offset
                    offset = self.offset(x)
                    print(f"    Step 1 - offset shape: {offset.shape}")
                    
                    scope = self.scope(x).sigmoid()
                    print(f"    Step 1 - scope shape: {scope.shape}")
                    
                    offset = offset * scope * 0.5
                    print(f"    Step 1 - after multiplication: {offset.shape}")
                    
                    offset = offset + self.init_pos
                    print(f"    Step 1 - after +init_pos: {offset.shape}")
                    
                    B, _, H, W = offset.shape
                    print(f"    Step 1 - B={B}, H={H}, W={W}")
                    
                    # Step 2: Reshape offset
                    offset = offset.view(B, 2, -1, H, W)
                    print(f"    Step 2 - offset after view: {offset.shape}")
                    
                    # Step 3: Create coordinate grid
                    coords_h = torch.arange(H, device=x.device) + 0.5
                    coords_w = torch.arange(W, device=x.device) + 0.5
                    print(f"    Step 3 - coords_h shape: {coords_h.shape}")
                    print(f"    Step 3 - coords_w shape: {coords_w.shape}")
                    
                    mesh = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
                    print(f"    Step 3 - mesh shape: {mesh.shape}")
                    
                    coords = mesh.transpose(1, 2).unsqueeze(1).unsqueeze(0)
                    print(f"    Step 3 - coords after transforms: {coords.shape}")
                    
                    coords = coords.type(x.dtype)
                    print(f"    Step 3 - coords after type cast: {coords.shape}")
                    
                    # Step 4: Normalize coordinates
                    normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device)
                    print(f"    Step 4 - normalizer: {normalizer}")
                    
                    normalizer = normalizer.view(1, 2, 1, 1, 1)
                    print(f"    Step 4 - normalizer after view: {normalizer.shape}")
                    
                    coords = 2 * (coords + offset) / normalizer - 1
                    print(f"    Step 4 - coords after normalize: {coords.shape}")
                    
                    # Step 5: Reshape for pixel_shuffle
                    coords_reshaped = coords.reshape(B, -1, H, W)
                    print(f"    Step 5 - coords after reshape: {coords_reshaped.shape}")
                    
                    coords_shuffled = F.pixel_shuffle(coords_reshaped, self.scale)
                    print(f"    Step 5 - after pixel_shuffle: {coords_shuffled.shape}")
                    
                    coords = coords_shuffled.view(B, 2, -1, self.scale * H, self.scale * W)
                    print(f"    Step 5 - after view: {coords.shape}")
                    
                    coords = coords.permute(0, 2, 3, 4, 1).contiguous()
                    print(f"    Step 5 - after permute: {coords.shape}")
                    
                    coords = coords.flatten(0, 1)
                    print(f"    Step 5 - after flatten: {coords.shape}")
                    
                    # Step 6: Reshape input for grid_sample
                    x_reshaped = x.reshape(B * self.groups, -1, H, W)
                    print(f"    Step 6 - input reshaped: {x_reshaped.shape}")
                    
                    # Step 7: Grid sample
                    output = F.grid_sample(
                        x_reshaped,
                        coords,
                        mode="bilinear",
                        align_corners=False,
                        padding_mode="zeros",
                    )
                    print(f"    Step 7 - after grid_sample: {output.shape}")
                    
                    # Step 8: Reshape output
                    output = output.view(B, -1, self.scale * H, self.scale * W)
                    print(f"    Step 8 - output after view: {output.shape}")
                    
                    # Step 9: End convolution
                    if self.end_convolution:
                        output = self.end_conv(output)
                        print(f"    Step 9 - after end_conv: {output.shape}")
                    
                    print(f"  ✅ DySample forward successful")
                    return output
                    
                except Exception as e:
                    print(f"  ❌ DySample forward failed at step: {e}")
                    traceback.print_exc()
                    raise
            
            # Bind the new method to the instance
            import types
            model.to_img.forward = types.MethodType(mps_safe_forward, model.to_img)
            print("  ✅ DySample patched for MPS compatibility")
        
        # Test with dummy input
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64).to(self.device)
            print(f"\n🔍 Testing forward pass on {self.device.type}:")
            print(f"  Input shape: {dummy.shape}")
            
            try:
                # Test each component separately
                print("\n  Testing model.feats:")
                feats_out = model.feats(dummy)
                print(f"    feats output shape: {feats_out.shape}")
                
                print("\n  Testing repeat_op:")
                repeat_out = model.repeat_op(dummy)
                print(f"    repeat_op output shape: {repeat_out.shape}")
                
                print("\n  Testing skip connection:")
                skip_out = feats_out + repeat_out
                print(f"    skip connection shape: {skip_out.shape}")
                
                if model.dysample and model.upscale != 1:
                    print("\n  Testing to_img (DySample):")
                    final_out = model.to_img(skip_out)
                    print(f"    to_img output shape: {final_out.shape}")
                else:
                    final_out = skip_out
                
                print(f"\n✅ Full test forward pass successful")
                print(f"  Final output shape: {final_out.shape}")
                print(f"  Output range: [{final_out.min():.3f}, {final_out.max():.3f}]")
                
            except Exception as e:
                print(f"\n❌ Test forward pass failed: {e}")
                traceback.print_exc()
                raise
        
        return model

    def _load_model_old(
        self,
        model_path: str,
        model_type: str,
        scale: int
    ) -> nn.Module:
        """Load model from checkpoint"""
        # Create model
        if model_type == 'realplksr_s':
            model = realplksr_s(upscaling_factor=scale)
        elif model_type == 'realplksr_l':
            model = realplksr_l(upscaling_factor=scale)
        else:
            model = realplksr(upscaling_factor=scale)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'params' in checkpoint:
            # This is the format from your error: has 'params' key
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
        # Try to load with strict=False to allow missing/unexpected keys
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"Loaded model from {model_path} with strict=True")
        except Exception as e:
            print(f"Strict loading failed, trying with strict=False...")
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
            if len(missing) > 0:
                print(f"Sample missing: {list(missing)[:5]}")
            if len(unexpected) > 0:
                print(f"Sample unexpected: {list(unexpected)[:5]}")
            print(f"Loaded model from {model_path} with strict=False")
        
        return model

    def process_image(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        benchmark: bool = False
    ) -> Optional[np.ndarray]:
        """
        Process a single image
        
        Args:
            input_path: Path to input image
            output_path: Path to save output (if None, returns array)
            benchmark: Print timing information
            
        Returns:
            Super-resolved image as numpy array [H, W, C] in [0, 255] (BGR for OpenCV)
        """
        print(f"Processing {input_path}...")
        
        # Load image - OpenCV loads as BGR
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to load image: {input_path}")
        
        h, w = img.shape[:2]
        print(f"Input size: {w}x{h}")
        
        # Run inference - model expects RGB, returns BGR
        start_time = time.time()
        
        sr_img_bgr = self.tiler.infer(img)  # Now returns BGR
        
        inference_time = time.time() - start_time
        
        if benchmark:
            mpixels = (w * h) / 1e6
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Time per MPixel: {inference_time / mpixels:.2f}s/MP")
            print(f"Output size: {sr_img_bgr.shape[1]}x{sr_img_bgr.shape[0]}")
        
        # Save or return
        if output_path:
            cv2.imwrite(output_path, sr_img_bgr)  # Direct save (already BGR)
            print(f"Saved to {output_path}")
        else:
            return sr_img_bgr
        
    def process_image_old(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        benchmark: bool = False
    ) -> Optional[np.ndarray]:
        """
        Process a single image
        
        Args:
            input_path: Path to input image
            output_path: Path to save output (if None, returns array)
            benchmark: Print timing information
            
        Returns:
            Super-resolved image if output_path is None
        """
        print(f"Processing {input_path}...")
        
        # Load image - ensure RGB
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to load image: {input_path}")
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        img_rgb = img #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        print(f"Input size: {w}x{h}")
        
        # Run inference
        start_time = time.time()
        
        sr_img = self.tiler.infer(img_rgb)  # Pass RGB image
        
        inference_time = time.time() - start_time
        
        if benchmark:
            mpixels = (w * h) / 1e6
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Time per MPixel: {inference_time / mpixels:.2f}s/MP")
            print(f"Output size: {sr_img.shape[1]}x{sr_img.shape[0]}")
        
        # Save or return
        if output_path:
            # Convert back to BGR for OpenCV saving
            sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, sr_img_bgr)
            print(f"Saved to {output_path}")
        else:
            return sr_img
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = ['.png', '.jpg', '.jpeg', '.bmp']
    ):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        images = []
        for ext in extensions:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(images)} images to process")
        
        total_time = 0
        total_pixels = 0
        
        for img_path in images:
            # Generate output path
            out_path = output_path / f"{img_path.stem}_sr{img_path.suffix}"
            
            # Process
            start_time = time.time()
            self.process_image(str(img_path), str(out_path))
            process_time = time.time() - start_time
            
            # Update stats
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            total_time += process_time
            total_pixels += w * h
        
        # Print summary
        avg_time = total_time / len(images)
        mpixels_total = total_pixels / 1e6
        print(f"\nProcessed {len(images)} images")
        print(f"Average time per image: {avg_time:.2f}s")
        print(f"Total pixels: {mpixels_total:.2f} MP")
        print(f"Overall throughput: {mpixels_total / total_time:.2f} MP/s")
    
    @torch.no_grad()
    def benchmark(self, input_size: Tuple[int, int] = (1920, 1080)):
        """Benchmark inference speed"""
        print(f"Benchmarking with input size {input_size[0]}x{input_size[1]}")
        
        # Create dummy input
        dummy = torch.randn(1, 3, input_size[1], input_size[0]).to(self.device)
        
        # Warmup
        for _ in range(5):
            _ = self.tiler.infer(dummy)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        times = []
        for _ in range(10):
            start = time.time()
            _ = self.tiler.infer(dummy)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        mpixels = (input_size[0] * input_size[1]) / 1e6
        
        print(f"Average inference time: {avg_time*1000:.1f} ms")
        print(f"Time per MPixel: {avg_time / mpixels:.2f}s/MP")
        print(f"Throughput: {mpixels / avg_time:.2f} MP/s")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="RealPLKSR Inference")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str,
                       help='Output path (file or directory)')
    parser.add_argument('--device', type=str,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Compute device')
    parser.add_argument('--scale', type=int, default=4,
                       help='Upscaling factor')
    parser.add_argument('--tile-size', type=int, default=512,
                       help='Tile size for inference')
    parser.add_argument('--halo-size', type=int, default=32,
                       help='Halo size for context')
    parser.add_argument('--model-type', type=str,
                       default='realplksr',
                       choices=['realplksr', 'realplksr_s', 'realplksr_l'],
                       help='Model architecture')
    parser.add_argument('--benchmark', action='store_true',
                       help='Print benchmark info')
    
    args = parser.parse_args()
    
    # Determine input type
    input_path = Path(args.input)
    
    # Create engine
    engine = InferenceEngine(
        model_path=args.model,
        device=args.device,
        scale=args.scale,
        tile_size=args.tile_size,
        halo_size=args.halo_size,
        model_type=args.model_type
    )
    
    # Process
    if input_path.is_file():
        # Single image
        if not args.output:
            args.output = input_path.parent / f"{input_path.stem}_sr{input_path.suffix}"
        
        engine.process_image(
            str(input_path),
            str(args.output),
            benchmark=args.benchmark
        )
        
    elif input_path.is_dir():
        # Directory
        if not args.output:
            args.output = input_path / "sr_output"
        
        engine.process_directory(
            str(input_path),
            str(args.output)
        )
        
    else:
        raise ValueError(f"Input not found: {input_path}")
    
    # Run benchmark if requested
    if args.benchmark and not input_path.is_dir():
        engine.benchmark()


if __name__ == "__main__":
    main()