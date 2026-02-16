Apple Silicon with Metal Performance Shaders has its challenges for super-resolution training due to partial operator support and memory architecture. The pipeline implements several device-specific adaptations to optimize stability and performance on MPS hardware.

## Worker

Multiprocessing with LMDB on macOS can cause instability during training. The dataloader can use a reduced worker count (0) for MPS device, it crashed otherwise on the Macbook M4 I tested with. Fork-based multiprocessing interacts poorly with certain PyTorch operations, leading to segment faults.

## Memory

MPS does not support pinned memory transfers, so the pin_memory flag is disabled when running on Apple Silicon. This prevents errors from unsupported operations and ensures data loading proceeds through standard memory pathways. A terabyte virtual memory map in LMDB remains effective on MPS though, providing the same cached storage benefits as on other platforms.

## DySample

The DySample upsampler relies on operations that have incomplete Pytorch MPS support: complex coordinate grid generation and reshaping sequences. Two approaches address this limitation. For training, DySample is on MPS in favor of PixelShuffle, accepting the quality trade-off for stability. For inference with pre-trained DySample models, the forward method is patched with a device-aware implementation that keeps all operations on the MPS device while avoiding unsupported patterns.

## Mixed Precision

Automatic mixed precision training with torch.cuda.amp is designed for CUDA and causes errors on MPS. The use_amp flag needs to be disabled for MPS training.

## Environment Variables

Several environment variables can widespread tune MPS behavior for most workloads. PYTORCH_ENABLE_MPS_FALLBACK enables CPU fallback for unsupported operations, (typically) preventing crashes at the cost of some performance. PYTORCH_MPS_HIGH_WATERMARK_RATIO controls memory pressure tolerance, with values above the default 1.7 potentially improving performance on systems with ample unified memory.