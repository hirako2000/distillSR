#!/usr/bin/env python3
"""
Better Web Converter - One-step ONNX optimization for web
Can run with: uv run better-web-converter.py
"""

import onnx
import onnxoptimizer
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
from pathlib import Path
import shutil
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Known good optimization passes that I guess would work across ONNX versions
SAFE_PASSES = [
    "eliminate_identity",
    "eliminate_nop_transpose",
    "eliminate_nop_pad",
    "fuse_consecutive_transposes",
    "fuse_transpose_into_gemm",
    "eliminate_deadend",
    "eliminate_duplicate_initializer",
    "eliminate_unused_initializer",
    "eliminate_nop_dropout",
    "eliminate_nop_cast",
    "eliminate_nop_monotone_argmax",
    "constant_folding",
]


def safe_optimize(model, passes):
    """Safely run optimization with only passes that exist"""
    available_passes = set(onnxoptimizer.get_available_passes())
    print(f"  Available passes: {len(available_passes)}")
    
    valid_passes = [p for p in passes if p in available_passes]
    skipped_passes = [p for p in passes if p not in available_passes]
    
    if skipped_passes:
        print(f"  ⚠️ Skipping unavailable passes: {skipped_passes}")
    
    if not valid_passes:
        print("  ⚠️ No valid optimization passes, returning original model")
        return model
    
    print(f"  Running optimizations: {valid_passes}")
    return onnxoptimizer.optimize(model, valid_passes)


def convert_to_fp16(model_path, output_path):
    """Convert model to FP16 precision"""
    print(f"\n🔄 Converting to FP16...")
    
    try:
        from onnxconverter_common import float16
    except ImportError:
        print("  Installing onnxconverter-common...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxconverter-common"])
        from onnxconverter_common import float16
    
    model = onnx.load(model_path)
    
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # as FP32 for compatibility
        op_block_list=['RandomNormal', 'RandomNormalLike', 'RandomUniform', 'RandomUniformLike']
    )
    
    onnx.save(model_fp16, output_path, save_as_external_data=False)
    
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  ✓ FP16 conversion complete! Size: {size_mb:.1f} MB")
    return True


def quantize_to_int8(model_path, output_path):
    """Quantize model to INT8"""
    print(f"\n🔄 Quantizing to INT8...")
    
    try:
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=['Conv', 'MatMul', 'Gemm', 'Add', 'Mul'],
            extra_options={
                'EnableSubgraph': True,
            }
        )
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ✓ INT8 quantization complete! Size: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  ❌ INT8 quantization failed: {e}")
        return False


def optimize_for_web(input_path, output_path, optimize=True):
    """
    Optimize ONNX model for web inference
    """
    print(f"\n🔄 Processing {input_path}...")
    
    print("  Loading model...")
    model = onnx.load(input_path)
    print(f"  ✓ Model loaded: {model.graph.name}")
    
    print(f"  Inputs: {len(model.graph.input)}")
    for i, inp in enumerate(model.graph.input):
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in inp.type.tensor_type.shape.dim]
        print(f"    - {inp.name}: {shape}")
    
    print(f"  Outputs: {len(model.graph.output)}")
    for out in model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in out.type.tensor_type.shape.dim]
        print(f"    - {out.name}: {shape}")
    
    print(f"  Initializers: {len(model.graph.initializer)}")
    print(f"  Nodes: {len(model.graph.node)}")
    
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    print(f"  Operations: {sorted(list(ops))}")
    
    if optimize:
        print("\n  Optimizing model...")
        try:
            model = safe_optimize(model, SAFE_PASSES)
            print("  ✓ Optimization complete")
        except Exception as e:
            print(f"  ⚠️ Optimization failed: {e}")
            print("  Continuing with original model...")
    
    has_external = False
    for initializer in model.graph.initializer:
        if initializer.HasField('data_location') and initializer.data_location == onnx.TensorProto.EXTERNAL:
            has_external = True
            break
    
    if has_external:
        print("  ⚠️ Model has external data, will embed all tensors")
    
    print(f"  Saving to {output_path}...")
    onnx.save(
        model,
        output_path,
        save_as_external_data=False,
        all_tensors_to_one_file=True
    )
    
    print("  Verifying model...")
    try:
        onnx.checker.check_model(output_path)
        print("  ✓ Model verification passed")
    except Exception as e:
        print(f"  ⚠️ Verification warning: {e}")
    
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved! Size: {size_mb:.1f} MB")
    
    return True


def inspect_model(model_path):
    """Print detailed model information"""
    print(f"\n📊 Model Inspection: {model_path}")
    model = onnx.load(model_path)
    
    print(f"  Model name: {model.graph.name}")
    print(f"  Producer: {model.producer_name} {model.producer_version}")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset version: {model.opset_import[0].version if model.opset_import else 'Unknown'}")
    
    print(f"\n  Inputs:")
    for input in model.graph.input:
        shape = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(str(dim.dim_value))
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append('?')
        print(f"    - {input.name}: {' × '.join(shape)}")
    
    print(f"\n  Outputs:")
    for output in model.graph.output:
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(str(dim.dim_value))
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append('?')
        print(f"    - {output.name}: {' × '.join(shape)}")
    
    print(f"\n  Initializers: {len(model.graph.initializer)}")
    print(f"  Nodes: {len(model.graph.node)}")
    
    node_types = {}
    for node in model.graph.node:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
    
    print(f"\n  Operations:")
    for op_type, count in sorted(node_types.items()):
        print(f"    - {op_type}: {count}")
    
    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"\n  File size: {size_mb:.1f} MB")
    
    return True


def main():
    """Main function - processes all model variants"""
    print("=" * 60)
    print("🔄 Better Web Converter - ONNX to Web Optimization")
    print("=" * 60)
    
    input_model = "realplksr_4x.onnx"
    output_dir = Path("web_models_optimized")
    
    if not Path(input_model).exists():
        print(f"\n❌ Input model not found: {input_model}")
        print("Please make sure realplksr_4x.onnx is in the current directory")
        return
    
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}/")
    
    inspect_model(input_model)
    
    print("\n" + "=" * 60)
    print("📦 Step 1: Creating Optimized FP32 Model")
    print("=" * 60)
    
    fp32_output = output_dir / "realplksr_web_fp32.onnx"
    optimize_for_web(input_model, str(fp32_output), optimize=True)
    
    print("\n" + "=" * 60)
    print("📦 Step 2: Creating FP16 Model")
    print("=" * 60)
    
    fp16_output = output_dir / "realplksr_web_fp16.onnx"
    try:
        convert_to_fp16(str(fp32_output), str(fp16_output))
    except Exception as e:
        print(f"  ❌ FP16 conversion failed: {e}")
        print("  Trying conversion from original model...")
        convert_to_fp16(input_model, str(fp16_output))
    
    print("\n" + "=" * 60)
    print("📦 Step 3: Creating INT8 Quantized Model")
    print("=" * 60)
    
    int8_output = output_dir / "realplksr_web_int8.onnx"
    success = quantize_to_int8(str(fp32_output), str(int8_output))
    
    if not success:
        print("  Trying quantization from original model...")
        quantize_to_int8(input_model, str(int8_output))
    
    print("\n" + "=" * 60)
    print("📦 Step 4: Creating Optimized INT8 Model")
    print("=" * 60)
    
    optimized_output = output_dir / "realplksr_web_optimized.onnx"
    if int8_output.exists():
        shutil.copy2(int8_output, optimized_output)
        print(f"  ✓ Copied INT8 model to {optimized_output}")
    else:
        print("  ⚠️ INT8 model not found, skipping copy")
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print("\n Generated Models:")
    
    for model_file in sorted(output_dir.glob("*.onnx")):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  • {model_file.name:<35} {size_mb:>6.1f} MB")
    
    print("\n Model Guide:")
    print("  • FP32: Best quality, most compatible (use for WebGPU if others fail)")
    print("  • FP16: Good balance of quality and speed (recommended for WebGPU)")
    print("  • INT8: Fastest, may have compatibility issues with some ops")
    print("  • Optimized: Alias for INT8 (for backward compatibility)")
    
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()