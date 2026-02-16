Docs guide through the project documentation, from understanding the data pipeline through training and inference.

## Data Preparation

The pipeline begins with raw image ingestion and transforms high-resolution imagery into training-ready patches through a three-tier medallion architecture.

[data-engineering.md](./data-engineering.md) explains the Bronze, Silver, and Gold tier architecture. Bronze handles raw ingestion from Hugging Face Hub. Silver validates, crops, and stores patches in LMDB with PNG compression. Gold generates on-the-fly degraded samples during training.

[gold-blur.md](./gold-blur.md) details the stochastic blur selection including isotropic Gaussian, anisotropic Gaussian, generalized Gaussian, and plateau kernels. Also covers Gaussian and Poisson noise injection with randomized parameters per batch.

[gold-resize-compress.md](./gold-resize-compress.md) - Describes multi-interpolation resizing with five interpolation methods and JPEG compression simulation including blocking artifacts, ringing, mosquito noise, and chroma subsampling. Explains the second-order loop with two compression passes.

[gold-filter.md](./gold-filter.md) covers the sinc filter implementation with Bessel functions and Lanczos windowing for ringing artifact simulation. Integrates all degradation components into the complete second-order pipeline.

## Architecture

The neural network design balances receptive field expansion with computational efficiency through channel-wise operations.

[registry-system.md](./registry-system.md) explains the decorator-based registration system for models, blocks, and upsamplers. Shows how components are discovered and instantiated dynamically from configuration strings.

## Training Configuration

Training setup is managed through a type-safe configuration system with YAML integration.

[adan-optimizer.md](./adan-optimizer.md) details the Adan optimizer with three momentum buffers and schedule-free training. Explains the mathematical foundations and integration with warmup and cosine annealing.

[loss-functions.md](./loss-functions.md) covers the loss suite including L1, MS-SSIM, consistency loss in Oklab space, local detail loss via gradients, and frequency domain loss through Fourier transforms.

[checkpoint-management.md](./checkpoint-management.md) documents the dual-format saving strategy with .pt files for full training state and .pth files for weight-only exports. Explains loading logic for pretrained weights and training resumption.

## Training

The training loop integrates several components with platform-specific optimizations.

[training.md](./training.md) walks through the training loop from stochastic degradation through forward pass, loss calculation, and backpropagation. Covers optimizer selection, scheduling, gradient clipping, and validation intervals.

[mps-optimizations.md](./mps-optimisations.md) details Apple Silicon specific subtleties including worker reduction, pin_memory disabling, DySample compatibility patches, and mixed precision handling. Explains environment variables for safer MPS stability.

## Evaluation and Testing

Validation and benchmarking ensure model quality and performance.

[testing.md](./testing.md) touches on the pytest-based test suite with fixtures for model discovery, image loading, and result aggregation. Explains benchmarking workflow and JSON result generation.

## Inference

Once trained, models are deployed through tiled inference and exported to production formats.

[inference-tiling.md](./inference-tiling.md) describes the overlapping tile strategy for processing arbitrarily large images within memory constraints. Explains halo sizing, receptive field considerations, and blending through weighted averaging.

[quantize-and-export.md](./quantize-and-export.md) details cross-platform precision strategies for different hardware targets. Covers ONNX export with dynamic axes, FP16 conversion, INT8 quantization, CoreML for Apple Neural Engine, and TensorRT for NVIDIA GPUs. Includes model card generation for provenance tracking.

## Getting Started

I would start with [data-engineering.md](./data-engineering.md) to understand the data pipeline, then proceed through the documents in the order presented. The [training.md](./training.md) document ties together concepts from configuration, optimization, and loss functions into the complete training workflow.

It's a mouthful. But would help if you consider fine tuning your own model, which is how this project started. Or if you are simply curious about what it takes to build a high speed/performance super resolution model.