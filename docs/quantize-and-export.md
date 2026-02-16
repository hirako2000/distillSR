# Cross-Platform Precision Strategy

Optimization is not one-size-fits-all. The optimal precision depends on the hardware target. While training typically uses FP32 full precision, deployment should target lower bit-widths to maximize throughput.

NVIDIA GPUs with CUDA benefit from FP16 precision. Tensor Cores are optimized for FP16 and BF16 math, often doubling speed compared to FP32 with negligible visual quality loss. Optimization backends include TensorRT and CUDA Graphs.

Apple Silicon with MPS thrives on BF16. Bfloat16 offers the same dynamic range as FP32 with half the memory, preventing NaN errors often encountered with standard FP16 during deep restoration. Metal Performance Shaders provide the optimization backend.

Intel and AMD CPUs gain most from INT8 quantization. Since CPUs lack the massive parallel floating-point power of GPUs, converting weights to eight-bit integers allows the model to utilize VNNI or AVX-512 instructions for substantial speed improvements.

Edge and mobile devices target INT8 or FP8 precision through CoreML and ONNX Runtime backends.

## Unified Quantization Logic

Quantization reduces memory footprint and accelerates arithmetic operations. For image restoration, weight-only quantization or static quantization maintain high structural integrity by preserving activation precision while compressing model weights.

# Model Export Universal

## Standardizing for Production with ONNX and Beyond

To ensure the model runs anywhere without requiring the original PyTorch source code, we use a tiered export strategy with ONNX as the universal intermediate representation.

## ONNX Universal Intermediate

ONNX, the Open Neural Network Exchange format, serves as the primary bridge between training and deployment. An exported ONNX file runs via ONNX Runtime, which automatically selects the optimal provider for the available hardware at runtime. The same ONNX file can execute through ORT-CUDA on NVIDIA hardware, ORT-MPS on Apple Silicon, or ORT-OpenVINO on Intel processors.

## Export Constraints

When exporting to ONNX, dynamic axes must be enabled. This allows the model to process images of any size rather than being locked to a fixed resolution, essential for the tiling strategy that handles arbitrarily large inputs.

## Hardware-Specific Last Mile Conversion

While ONNX provides universal compatibility, some platforms offer deeper optimizations through hardware-specific formats. For Apple hardware, ONNX converts to CoreML as a mlpackage to utilize the dedicated Apple Neural Engine. For NVIDIA hardware, ONNX converts to a TensorRT engine as a plan file to fuse layers and optimize memory kernels for specific GPU architectures such as Ada Lovelace or Ampere.

## Advice

For production release, three weight variants should be provided. The fp16 ONNX model serves as the standard high-speed version for most GPUs. The int8 ONNX model provides a lightweight version for CPU-only environments. The full precision PyTorch checkpoint allows further training or fine-tuning.