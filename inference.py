"""
Tiled inference for high-resolution images
Handles 4K/8K images with memory-efficient tiling strategy
"""

import argparse
import sys
import time
import types
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.append(str(Path(__file__).parent))
import traceback

from archs.realplksr import realplksr, realplksr_l, realplksr_s


class TiledInference:
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
        self.model.eval()

        if lr_image.max() > 1.0:
            lr_image = lr_image.astype(np.float32) / 255.0

        if lr_image.shape[-1] == 3:
            lr_image_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        else:
            lr_image_rgb = lr_image

        img_tensor = torch.from_numpy(lr_image_rgb).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        h, w = img_tensor.shape[2:]

        pad_h = max(0, self.effective_tile - h)
        pad_w = max(0, self.effective_tile - w)
        if pad_h > 0 or pad_w > 0:
            img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        else:
            img_padded = img_tensor

        sr_full = self.model(img_padded)
        sr_full = sr_full[:, :, :h*self.scale, :w*self.scale]

        print(f"Raw output range: [{sr_full.min():.3f}, {sr_full.max():.3f}]")

        # Always use the same post-processing that works for both model types
        min_val, max_val = 0, 1
        sr_np = sr_full.squeeze(0).cpu().clamp(min_val, max_val)
        sr_np = (sr_np - min_val) / (max_val - min_val)
        sr_np = sr_np.permute(1, 2, 0).numpy()
        sr_np = (sr_np * 255.0).round().astype(np.uint8)
        sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

        return sr_bgr

    @torch.no_grad()
    def infer_old(self, lr_image: np.ndarray) -> np.ndarray:
        self.model.eval()

        if lr_image.max() > 1.0:
            lr_image = lr_image.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(lr_image).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        h, w = img_tensor.shape[2:]

        if h <= self.effective_tile and w <= self.effective_tile:
            print("Image smaller than tile, running direct inference...")

            pad_h = max(0, self.effective_tile - h)
            pad_w = max(0, self.effective_tile - w)
            if pad_h > 0 or pad_w > 0:
                img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
            else:
                img_padded = img_tensor

            sr_full = self.model(img_padded)
            sr_np = sr_full.squeeze(0).cpu().permute(1, 2, 0).numpy()
            sr_np = (sr_np * 255).clip(0, 255).astype(np.uint8)

            return sr_np

        print("Image larger than tile, using tiling...")

        pad_h = (self.effective_tile - (h % self.effective_tile)) % self.effective_tile
        pad_w = (self.effective_tile - (w % self.effective_tile)) % self.effective_tile
        img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')

        padded_h, padded_w = img_padded.shape[2:]
        n_h = (padded_h - self.halo_size * 2) // self.tile_size
        n_w = (padded_w - self.halo_size * 2) // self.tile_size

        out_h = h * self.scale
        out_w = w * self.scale

        output = torch.zeros(3, out_h, out_w, device=self.device)
        weight = torch.ones(3, out_h, out_w, device=self.device) * 1e-8

        print(f"Processing {n_h}x{n_w} tiles...")

        for i in range(n_h):
            for j in range(n_w):
                y_start = i * self.tile_size
                x_start = j * self.tile_size
                y_end = y_start + self.effective_tile
                x_end = x_start + self.effective_tile

                tile = img_padded[:, :, y_start:y_end, x_start:x_end]
                sr_tile = self.model(tile)
                sr_tile = torch.tanh(sr_tile)
                sr_tile = (sr_tile + 1) / 2
                sr_tile = sr_tile.clamp(0, 1)
                sr_tile = sr_tile.squeeze(0)

                out_y_start = y_start * self.scale
                out_x_start = x_start * self.scale
                out_y_end = min((y_start + self.tile_size) * self.scale, out_h)
                out_x_end = min((x_start + self.tile_size) * self.scale, out_w)

                tile_y_start = self.halo_size * self.scale
                tile_x_start = self.halo_size * self.scale
                tile_y_end = tile_y_start + (out_y_end - out_y_start)
                tile_x_end = tile_x_start + (out_x_end - out_x_start)

                output[:, out_y_start:out_y_end, out_x_start:out_x_end] += \
                    sr_tile[:, tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                weight[:, out_y_start:out_y_end, out_x_start:out_x_end] += 1

        output = output / weight
        output_np = output.cpu().permute(1, 2, 0).numpy()
        output_np = (output_np * 255).clip(0, 255).astype(np.uint8)

        return output_np


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        scale: int = 4,
        tile_size: int = 512,
        halo_size: int = 32,
        model_type: str = 'realplksr',
        dysample: bool = False  # Add this parameter
    ):
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
        print(f"Model uses DySample: {dysample}")

        # Pass dysample to _load_model
        self.model = self._load_model(model_path, model_type, scale, dysample)
        self.model = self.model.to(self.device)
        self.model.eval()

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
        scale: int,
        dysample: bool
    ) -> nn.Module:
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        print("\n📊 Model configuration:")
        print(f"  Explicit dysample flag: {dysample}")

        # Create model with explicit dysample flag
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
            dysample=dysample
        )

        # Store model type for inference
        model.upsampler_type = 'dysample' if dysample else 'pixelshuffle'

        # Clean state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v

        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print("\n📊 Loading results:")
        print(f"  Missing keys: {len(missing)}")
        if missing:
            print(f"  First 5 missing: {list(missing)[:5]}")
        print(f"  Unexpected keys: {len(unexpected)}")

        model = model.to(self.device)

        # For DySample models on MPS, use the EXACT working implementation from your original code
        if dysample and self.device.type == 'mps':
            print("🔧 MPS detected with DySample model - applying original working DySample implementation...")

            # EXACT original working MPS forward method with all debug prints
            def mps_safe_forward(self, x):
                print("\n  🔍 DySample forward debug:")
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

                    print("  ✅ DySample forward successful")
                    return output

                except Exception as e:
                    print(f"  ❌ DySample forward failed at step: {e}")
                    traceback.print_exc()
                    raise

            model.to_img.forward = types.MethodType(mps_safe_forward, model.to_img)
            print("  ✅ DySample patched with original working implementation")

        model.eval()

        # Test forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64).to(self.device)
            try:
                final_out = model(dummy)
                print("\n✅ Test forward pass successful")
                print(f"  Output shape: {final_out.shape}")
                print(f"  Output range: [{final_out.min():.3f}, {final_out.max():.3f}]")
                print(f"  Model type: {model.upsampler_type}")
            except Exception as e:
                print(f"\n❌ Test forward pass failed: {e}")
                traceback.print_exc()

        return model

    def _load_model_pixel_suffle(
        self,
        model_path: str,
        model_type: str,
        scale: int
    ) -> nn.Module:

        checkpoint = torch.load(model_path, map_location='cpu')

        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        # Detect what the checkpoint actually contains
        has_dysample = False
        first_key = next(iter(state_dict.keys()))
        print(f"\n📊 Checkpoint first key: {first_key}")

        # Look for DySample-specific keys
        for key in state_dict.keys():
            if 'to_img.offset' in key or 'to_img.scope' in key or 'to_img.init_pos' in key:
                has_dysample = True
                break

        print(f"  Upsampler type: {'DySample' if has_dysample else 'PixelShuffle'}")

        # Check if this is a full checkpoint or just state dict
        if 'config' in checkpoint:
            print("  Full checkpoint with config")

        # Create model with the CORRECT upsampler type
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
            dysample=has_dysample  # Use what the checkpoint actually has
        )

        # Clean state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            # Remove any DySample keys if we're not using them
            if not has_dysample and any(x in k for x in ['offset', 'scope', 'init_pos', 'end_conv']):
                print(f"  Skipping DySample key: {k}")
                continue
            new_state_dict[k] = v

        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print("\n📊 Loading results:")
        print(f"  Missing keys: {len(missing)}")
        if missing:
            print(f"  First 5 missing: {list(missing)[:5]}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if unexpected:
            print(f"  First 5 unexpected: {list(unexpected)[:5]}")

        model = model.to(self.device)
        model.eval()

        # Test forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64).to(self.device)
            try:
                final_out = model(dummy)
                print("\n✅ Test forward pass successful")
                print(f"  Output shape: {final_out.shape}")
                print(f"  Output range: [{final_out.min():.3f}, {final_out.max():.3f}]")
            except Exception as e:
                print(f"\n❌ Test forward pass failed: {e}")
                traceback.print_exc()

        return model

    def _load_model_old(
        self,
        model_path: str,
        model_type: str,
        scale: int
    ) -> nn.Module:
        if model_type == 'realplksr_s':
            model = realplksr_s(upscaling_factor=scale)
        elif model_type == 'realplksr_l':
            model = realplksr_l(upscaling_factor=scale)
        else:
            model = realplksr(upscaling_factor=scale)

        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Checkpoint keys: {checkpoint.keys()}")

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

        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"Loaded model from {model_path} with strict=True")
        except Exception:
            print("Strict loading failed, trying with strict=False...")
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
        print(f"Processing {input_path}...")

        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to load image: {input_path}")

        h, w = img.shape[:2]
        print(f"Input size: {w}x{h}")

        start_time = time.time()
        sr_img_bgr = self.tiler.infer(img)
        inference_time = time.time() - start_time

        if benchmark:
            mpixels = (w * h) / 1e6
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Time per MPixel: {inference_time / mpixels:.2f}s/MP")
            print(f"Output size: {sr_img_bgr.shape[1]}x{sr_img_bgr.shape[0]}")

        if output_path:
            cv2.imwrite(output_path, sr_img_bgr)
            print(f"Saved to {output_path}")
        else:
            return sr_img_bgr

    def process_image_old(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        benchmark: bool = False
    ) -> Optional[np.ndarray]:
        print(f"Processing {input_path}...")

        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to load image: {input_path}")

        img_rgb = img
        h, w = img_rgb.shape[:2]
        print(f"Input size: {w}x{h}")

        start_time = time.time()
        sr_img = self.tiler.infer(img_rgb)
        inference_time = time.time() - start_time

        if benchmark:
            mpixels = (w * h) / 1e6
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Time per MPixel: {inference_time / mpixels:.2f}s/MP")
            print(f"Output size: {sr_img.shape[1]}x{sr_img.shape[0]}")

        if output_path:
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
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        images = []
        for ext in extensions:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))

        print(f"Found {len(images)} images to process")

        total_time = 0
        total_pixels = 0

        for img_path in images:
            out_path = output_path / f"{img_path.stem}_sr{img_path.suffix}"
            start_time = time.time()
            self.process_image(str(img_path), str(out_path))
            process_time = time.time() - start_time

            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            total_time += process_time
            total_pixels += w * h

        avg_time = total_time / len(images)
        mpixels_total = total_pixels / 1e6
        print(f"\nProcessed {len(images)} images")
        print(f"Average time per image: {avg_time:.2f}s")
        print(f"Total pixels: {mpixels_total:.2f} MP")
        print(f"Overall throughput: {mpixels_total / total_time:.2f} MP/s")

    @torch.no_grad()
    def benchmark(self, input_size: Tuple[int, int] = (1920, 1080)):
        print(f"Benchmarking with input size {input_size[0]}x{input_size[1]}")

        dummy = torch.randn(1, 3, input_size[1], input_size[0]).to(self.device)

        for _ in range(5):
            _ = self.tiler.infer(dummy)

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

    input_path = Path(args.input)

    engine = InferenceEngine(
        model_path=args.model,
        device=args.device,
        scale=args.scale,
        tile_size=args.tile_size,
        halo_size=args.halo_size,
        model_type=args.model_type
    )

    if input_path.is_file():
        if not args.output:
            args.output = input_path.parent / f"{input_path.stem}_sr{input_path.suffix}"

        engine.process_image(
            str(input_path),
            str(args.output),
            benchmark=args.benchmark
        )

    elif input_path.is_dir():
        if not args.output:
            args.output = input_path / "sr_output"

        engine.process_directory(
            str(input_path),
            str(args.output)
        )

    else:
        raise ValueError(f"Input not found: {input_path}")

    if args.benchmark and not input_path.is_dir():
        engine.benchmark()


if __name__ == "__main__":
    main()
