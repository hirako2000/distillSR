"""
Simple ONNX web converter
"""

import onnx
from pathlib import Path

def convert_for_web(input_path: str, output_path: str):
    """
    Simple conversion: load model and save with external data embedded
    """
    print(f"\n🔄 Converting {input_path} for web...")
    
    try:
        print("  Loading model...")
        model = onnx.load(input_path)
        print(f"  ✓ Model loaded: {model.graph.name}")
        
        external_data = False
        for initializer in model.graph.initializer:
            if initializer.HasField('data_location') and initializer.data_location == onnx.TensorProto.EXTERNAL:
                external_data = True
                break
        
        if external_data:
            print("  ⚠️ Model has external data, will embed all tensors")
        
        print(f"  Saving to {output_path}...")
        onnx.save(
            model,
            output_path,
            save_as_external_data=False,
            all_tensors_to_one_file=True
        )
        
        print("  Verifying model...")
        onnx.checker.check_model(output_path)
        
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ✓ Conversion successful! Size: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    output_dir = Path('web_models')
    output_dir.mkdir(exist_ok=True)
    
    models = [
        ('realplksr_4x.onnx', 'realplksr_web_fp32.onnx'),
        ('realplksr_4x_fp16.onnx', 'realplksr_web_fp16.onnx'),
        ('realplksr_4x_int8.onnx', 'realplksr_web_int8.onnx')
    ]
    
    success_count = 0
    for input_name, output_name in models:
        input_path = Path(input_name)
        if not input_path.exists():
            print(f"\n⚠️ Skipping {input_name} - file not found")
            continue
            
        output_path = output_dir / output_name
        if convert_for_web(str(input_path), str(output_path)):
            success_count += 1
    
    print(f"\n✅ Converted {success_count} models to {output_dir}/")
    print("\nNow serve these files with your web app!")


if __name__ == '__main__':
    main()