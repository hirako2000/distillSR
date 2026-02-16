# tests/conftest.py
from pathlib import Path
from typing import Dict, List

import pytest

from archs.realplksr import realplksr


def pytest_addoption(parser):
    parser.addoption(
        "--model-name",
        action="store",
        default="realplksr_4x",
        help="Base name of the model to test (e.g., realplksr_4x)"
    )
    parser.addoption(
        "--models-dir",
        action="store",
        default="exports",
        help="Directory containing quantized models"
    )
    parser.addoption(
        "--pytorch-models-dir",
        action="store",
        default="weights",
        help="Directory containing PyTorch models"
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
    """Get model name from command line"""
    return request.config.getoption("--model-name")

@pytest.fixture
def models_dir(request) -> Path:
    """Get models directory for ONNX models"""
    return Path(request.config.getoption("--models-dir"))

@pytest.fixture
def pytorch_models_dir(request) -> Path:
    """Get models directory for PyTorch models"""
    return Path(request.config.getoption("--pytorch-models-dir"))

@pytest.fixture
def images_dir(request) -> Path:
    """Get images directory"""
    return Path(request.config.getoption("--images-dir"))

@pytest.fixture
def output_dir(request) -> Path:
    """Get output directory"""
    return Path(request.config.getoption("--output-dir"))

@pytest.fixture
def test_images(images_dir) -> List[Path]:
    """Find all test images"""
    if not images_dir.exists():
        pytest.skip(f"Images directory not found: {images_dir}")

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    images = []
    for ext in extensions:
        images.extend(images_dir.glob(ext))
        images.extend(images_dir.glob(ext.upper()))

    if not images:
        pytest.skip(f"No test images found in {images_dir}")

    return sorted(images)

@pytest.fixture
def quantized_models(models_dir, model_name) -> Dict[str, Path]:
    """Find all quantized versions of a model"""
    if not models_dir.exists():
        pytest.skip(f"Models directory not found: {models_dir}")

    # Define model variants to look for
    variants = {
        'fp32': f"{model_name}.onnx",
        'fp16': f"{model_name}_fp16.onnx",
        'int8': f"{model_name}_int8.onnx",
    }

    found_models = {}
    for variant, filename in variants.items():
        model_path = models_dir / filename
        if model_path.exists():
            found_models[variant] = model_path
        # Also check for external data format
        elif (models_dir / filename).exists():
            found_models[variant] = models_dir / filename

    if not found_models:
        pytest.skip(f"No quantized models found for {model_name} in {models_dir}")

    return found_models

@pytest.fixture
def model_info(quantized_models) -> Dict:
    """Get model information for ONNX models"""
    info = {}
    for variant, path in quantized_models.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        info[variant] = {
            'path': path,
            'size_mb': size_mb,
            'exists': True
        }
    return info

@pytest.fixture
def pytorch_models(pytorch_models_dir, model_name) -> Dict[str, Path]:
    """Find all PyTorch model variants"""
    if not pytorch_models_dir.exists():
        pytest.skip(f"PyTorch models directory not found: {pytorch_models_dir}")

    # Look for common PyTorch checkpoint names
    possible_names = [
        f"{model_name}.pth",
        f"{model_name}.pt",
        f"{model_name}.pth.tar",
        "model_best.pt",
        "4xNomos2_realplksr_dysample.pth",
        "4x-RealPLKSR_dysample_pretrain.pth",
        "4xNomosWebPhoto_RealPLKSR.pth",
        "4xPurePhoto-RealPLSKR.pth"
    ]

    found_models = {}
    for name in possible_names:
        model_path = pytorch_models_dir / name
        if model_path.exists():
            found_models['fp32'] = model_path
            print(f"  Found PyTorch model: {model_path.name}")
            break

    if not found_models:
        # If model_name doesn't match, try to find any .pth file
        all_pth = list(pytorch_models_dir.glob("*.pth"))
        if all_pth:
            found_models['fp32'] = all_pth[0]
            print(f"  Using first found PyTorch model: {all_pth[0].name}")
        else:
            pytest.skip(f"No PyTorch models found in {pytorch_models_dir}")

    return found_models

@pytest.fixture
def pytorch_model_info(pytorch_models) -> Dict:
    """Get model information for PyTorch models"""
    info = {}
    for variant, path in pytorch_models.items():
        size_mb = path.stat().st_size / (1024 * 1024)

        model = realplksr(
            in_ch=3,
            out_ch=3,
            dim=64,
            n_blocks=28,
            upscaling_factor=4,
            kernel_size=17,
            split_ratio=0.25,
            use_ea=True,
            norm_groups=4,
            dropout=0.1,
            dysample=True
        )

        params = sum(p.numel() for p in model.parameters())

        info[variant] = {
            'path': path,
            'size_mb': size_mb,
            'params': params,
            'exists': True
        }
    return info
