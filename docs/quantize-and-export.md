# Cross Platform & precision optimisation

Not one-size-fits-all. Optimal precision depends on the hardware target. While training typically uses FP32 full precision, deployment should target lower bit-widths to maximize throughput.

NVIDIA GPUs with CUDA benefit from FP16 precision. Tensor Cores are optimized for FP16 and BF16 math, often doubling speed compared to FP32 with negligible visual quality loss. ONNX models can be further compiled into TensorRT engines that fuse layers and optimize memory access for specific GPU architectures.

Apple Silicon with MPS thrives on BF16. Bfloat16 offers the same dynamic range as FP32 with half the memory, preventing NaN errors often encountered with standard FP16 during deep restoration. CoreML conversion enables deployment through the Apple Neural Engine for further acceleration.

Intel and AMD CPUs gain most from INT8 quantization. Converting weights to eight-bit integers allows the model to utilize VNNI or AVX-512 instructions for substantial speed improvements. ONNX Runtime with OpenVINO can further optimize CPU inference.

Edge and mobile devices target INT8 or FP8 precision through CoreML and ONNX Runtime backends, balancing model size against inference speed on resource-constrained hardware.

## Quantization

Quantization reduces numerical precision from 32-bit floating point to lower bit-widths such as 16-bit float or 8-bit integer. This compresses model size and accelerates arithmetic operations at the cost of marginal precision loss that rarely affects visual quality. 

For image restoration, dynamic quantization compresses weights to INT8 while preserving activation precision, maintaining structural integrity. The export pipeline supports both FP16 conversion and INT8 dynamic quantization.

# Model Exports

ONNX is an universal standard.

## ONNX exporter

ONNX serves as the intermediate representation between training and deployment. An exported ONNX file runs via the ONNX Runtime, which automatically selects the optimal provider for available hardware: ORT-CUDA on NVIDIA, ORT-MPS on Apple Silicon, or ORT-OpenVINO on Intel processors. Export uses opset version 18 with constant folding optimization.

## Export Constraints

Dynamic axes must be enabled during ONNX export. This allows the model to process images of any size rather than being locked to a fixed resolution, essential for the tiling strategy that handles arbitrarily large inputs.

## Hardware-Specific Conversion

For Apple hardware, ONNX converts to CoreML as a mlpackage to utilize the Apple Neural Engine. For NVIDIA hardware, ONNX converts to a TensorRT engine that fuses layers and optimizes memory kernels for specific GPU architectures.

## Model Card

Export generates a JSON model card containing architecture details, parameter count, training provenance, validation metrics, and exported formats. This metadata accompanies the weights for provenance tracking and deployment documentation.

## Advice

For production release, provide three weight variants. The fp16 ONNX model serves as the standard high-speed version for most GPUs. The int8 ONNX model provides a lightweight version for CPU-only environments. The full precision PyTorch checkpoint allows further training or fine-tuning. Configure input shapes with dynamic axes to maintain flexibility for variable-sized inputs required by the tiling strategy.