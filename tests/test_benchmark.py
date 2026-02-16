# tests/test_benchmark.py
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def pytest_addoption(parser):
    parser.addoption(
        "--model-name",
        action="store",
        default="4xNomos2_realplksr_dysample",
        help="Name of the model to test"
    )
    parser.addoption(
        "--models-dir",
        action="store",
        default="weights",
        help="Directory containing models"
    )
    parser.addoption(
        "--images-dir",
        action="store",
        default="tests/images",
        help="Directory with test images"
    )
    parser.addoption(
        "--output-dir",
        action="store",
        default="tests/images-output",
        help="Directory for output images"
    )

@pytest.fixture
def model_name(request) -> str:
    return request.config.getoption("--model-name")

@pytest.fixture
def models_dir(request) -> Path:
    return Path(request.config.getoption("--models-dir"))

@pytest.fixture
def images_dir(request) -> Path:
    return Path(request.config.getoption("--images-dir"))

@pytest.fixture
def output_dir(request) -> Path:
    return Path(request.config.getoption("--output-dir"))

@pytest.fixture
def model_path(models_dir, model_name) -> Path:
    """Find the model file"""
    model_path = models_dir / f"{model_name}.pth"
    if not model_path.exists():
        # Try alternative names
        alternatives = [
            models_dir / model_name,
            models_dir / f"{model_name}.pt",
            models_dir / "4xNomos2_realplksr_dysample.pth"
        ]
        for alt in alternatives:
            if alt.exists():
                return alt
        pytest.skip(f"Model not found: {model_path}")
    return model_path

@pytest.fixture
def test_images(images_dir) -> list:
    """Find all test images"""
    if not images_dir.exists():
        pytest.skip(f"Images directory not found: {images_dir}")

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(images_dir.glob(ext))
        images.extend(images_dir.glob(ext.upper()))

    if not images:
        pytest.skip(f"No test images found in {images_dir}")

    return sorted(images)

def run_cli_inference(model_path: Path, image_path: Path, output_path: Path) -> dict:
    """Run inference via CLI and extract timing"""

    cmd = [
        "uv", "run", "cli.py",
        "infer",
        str(model_path),
        str(image_path),
        "--output", str(output_path),
        "--benchmark"
    ]

    # Run and time
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_time = (time.perf_counter() - start) * 1000

    # Parse output for inference time
    inference_time = None
    for line in result.stdout.split('\n'):
        if "Inference time:" in line:
            match = re.search(r"Inference time: ([\d.]+)s", line)
            if match:
                inference_time = float(match.group(1)) * 1000

    return {
        'success': result.returncode == 0,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'total_time_ms': total_time,
        'inference_time_ms': inference_time,
        'output_path': output_path if output_path.exists() else None
    }

@pytest.mark.benchmark
def test_model_benchmark(
    model_path,
    test_images,
    output_dir,
    tmp_path
):
    """Benchmark model using the CLI"""

    print(f"\n{'='*60}")
    print(f"🚀 Benchmarking: {model_path.name}")
    print(f"{'='*60}")
    print(f"Python: {platform.python_version()}")
    print(f"System: {platform.system()} {platform.release()}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': str(model_path),
        'model_size_mb': model_path.stat().st_size / (1024 * 1024),
        'system': {
            'python_version': platform.python_version(),
            'system': platform.system(),
            'release': platform.release()
        },
        'images': []
    }

    # Test each image
    for img_path in test_images:
        print(f"\n📸 Processing: {img_path.name}")

        # Create output path
        out_path = tmp_path / f"out_{img_path.name}"
        final_path = output_dir / f"{img_path.stem}_sr{img_path.suffix}"

        # Run inference
        result = run_cli_inference(model_path, img_path, out_path)

        if result['success'] and result['output_path']:
            # Copy to final location
            import shutil
            shutil.copy2(result['output_path'], final_path)

            results['images'].append({
                'image': img_path.name,
                'total_time_ms': result['total_time_ms'],
                'inference_time_ms': result['inference_time_ms'],
                'output': str(final_path)
            })

            print("  ✅ Success")
            print(f"  ⏱️  Total: {result['total_time_ms']:.2f}ms")
            if result['inference_time_ms']:
                print(f"  ⏱️  Inference: {result['inference_time_ms']:.2f}ms")
        else:
            print(f"  ❌ Failed: {result['stderr']}")

    # Calculate statistics
    if results['images']:
        times = [img['inference_time_ms'] or img['total_time_ms']
                for img in results['images']]
        results['statistics'] = {
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'std_ms': float(np.std(times))
        }

        print("\n📊 Statistics:")
        print(f"  Mean: {results['statistics']['mean_ms']:.2f}ms")
        print(f"  Median: {results['statistics']['median_ms']:.2f}ms")
        print(f"  Min: {results['statistics']['min_ms']:.2f}ms")
        print(f"  Max: {results['statistics']['max_ms']:.2f}ms")

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to {results_path}")

    # Basic assertion
    assert len(results['images']) > 0, "No images processed successfully"
