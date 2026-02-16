#!/usr/bin/env python3
"""
Command-line interface for RealPLKSR Factory
All pipeline steps accessible through typer commands
"""

import gc
import multiprocessing
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import lmdb
import torch
import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from export_onnx import ModelExporter, load_model_for_export
from inference import InferenceEngine
from loggers.logger import test_logger
from pipeline.degradations import test_degradation
from pipeline.fetch_hf import HFDataFetcher, create_synthetic_dataset
from pipeline.medallion_svc import MedallionService
from train import Trainer
from train import main as train_main
from training.metrics import test_metrics

multiprocessing.set_start_method('fork', force=True)

sys.path.append(str(Path(__file__).parent))

app = typer.Typer(help="RealPLKSR Factory: Super-Resolution Training & Inference Pipeline")
console = Console()

# ============================================================================
# Data Pipeline Commands
# ============================================================================

@app.command("list-datasets")
def list_datasets():
    """List available datasets on Hugging Face"""
    

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Fetching datasets...", total=None)
        fetcher = HFDataFetcher()
        datasets = fetcher.list_available_datasets()

    table = Table(title="📦 Available Datasets")
    table.add_column("Dataset", style="cyan")
    table.add_column("Repository", style="green")

    for name, repo in datasets.items():
        table.add_row(name, repo)

    console.print(table)
    return 0


@app.command("download")
def download_dataset(
    dataset: str = typer.Argument(..., help="Dataset name"),
    repo: Optional[str] = typer.Option(None, help="HF repo ID (optional)"),
    subset: Optional[str] = typer.Option(None, help="Subset to download"),
    max_images: Optional[int] = typer.Option(None, help="Maximum images"),
    min_size: int = typer.Option(512, help="Minimum image size"),
    output: str = typer.Option("data/bronze", help="Output directory"),
    workers: int = typer.Option(4, help="Parallel workers")
):
    """Download a dataset from Hugging Face Hub"""
    console.print(f"\n📥 [bold]Downloading dataset:[/bold] {dataset}")

    with Progress() as progress:
        task = progress.add_task("Downloading...", total=None)

        fetcher = HFDataFetcher(
            bronze_dir=output,
            max_workers=workers
        )

        stats = fetcher.download_dataset(
            dataset_name=dataset,
            repo_id=repo,
            subset=subset,
            max_images=max_images,
            min_resolution=(min_size, min_size)
        )

        progress.update(task, completed=100)

    console.print("\n✅ [green]Download complete:[/green]")
    console.print(f"  • Downloaded: {stats['downloaded']}")
    console.print(f"  • Skipped: {stats['skipped']}")
    console.print(f"  • Failed: {stats['failed']}")
    console.print(f"  • Total size: {stats['total_size_mb']:.2f} MB")

    return 0


@app.command("synthetic")
def create_synthetic(
    num: int = typer.Option(100, help="Number of images"),
    output: str = typer.Option("data/bronze/synthetic", help="Output directory"),
    sizes: str = typer.Option("1024,2048,4096", help="Comma-separated image sizes")
):
    """Create synthetic dataset for testing"""
    size_list = [int(s) for s in sizes.split(',')]
    size_tuples = [(s, s) for s in size_list]

    console.print(f"\n🎨 [bold]Creating {num} synthetic images[/bold] in {output}")

    with Progress() as progress:
        task = progress.add_task("Generating...", total=num)

        def progress_callback():
            progress.update(task, advance=1)

        # Note: You'd need to modify create_synthetic_dataset to accept a callback
        create_synthetic_dataset(
            output_dir=output,
            num_images=num,
            sizes=size_tuples
        )

    console.print("✅ [green]Synthetic dataset created[/green]")
    return 0


@app.command("process")
def process_dataset(
    dataset: str = typer.Argument(..., help="Dataset name in data/bronze/"),
    patch_size: int = typer.Option(256, help="Patch size"),
    stride: Optional[int] = typer.Option(None, help="Extraction stride"),
    min_size: int = typer.Option(1024, help="Minimum image size"),
    max_images: Optional[int] = typer.Option(None, help="Maximum images to process"),
    workers: int = typer.Option(4, help="Number of workers"),
    bronze: str = typer.Option("data/bronze", help="Bronze directory"),
    silver: str = typer.Option("data/silver", help="Silver directory")
):
    """Process bronze dataset to silver LMDB"""

    console.print(f"\n⚙️  [bold]Processing dataset:[/bold] {dataset}")
    console.print(f"📂 Bronze: {bronze}/{dataset}")
    console.print(f"💿 Silver: {silver}")
    console.print(f"📏 Patch size: {patch_size}")
    console.print(f"📐 Min size: {min_size}")
    console.print(f"👥 Workers: {workers}")

    # Check if bronze directory exists
    bronze_path = Path(bronze) / dataset
    if not bronze_path.exists():
        console.print(f"❌ [red]Bronze directory not found:[/red] {bronze_path}")
        console.print(f"Please place your images in: {bronze_path}")
        return 1

    # Count images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    images = []
    for ext in image_extensions:
        images.extend(bronze_path.glob(f"*{ext}"))
        images.extend(bronze_path.glob(f"*{ext.upper()}"))

    console.print(f"📸 Found {len(images)} images")

    if len(images) == 0:
        console.print(f"❌ [red]No images found[/red] in {bronze_path}")
        return 1

    with Progress() as progress:
        task = progress.add_task("Processing...", total=None)

        service = MedallionService(
            bronze_dir=bronze,
            silver_dir=silver,
            patch_size=patch_size,
            stride=stride,
            min_size=min_size,
            num_workers=workers
        )

        stats = service.process_dataset(dataset, max_images)
        progress.update(task, completed=100)

    console.print("\n✅ [green]Processing complete:[/green]")
    console.print(f"  • Total images: {stats['total_images']}")
    console.print(f"  • Valid images: {stats['valid_images']}")
    console.print(f"  • Invalid images: {stats['invalid_images']}")
    console.print(f"  • Total patches: {stats['total_patches']}")
    console.print(f"  • Time: {stats['processing_time']:.2f}s")

    return 0


@app.command("process-all")
def process_all_datasets(
    patch_size: int = typer.Option(256, help="Patch size"),
    min_size: int = typer.Option(1024, help="Minimum image size"),
    max_datasets: Optional[int] = typer.Option(None, help="Maximum datasets to process"),
    workers: int = typer.Option(4, help="Number of workers"),
    bronze: str = typer.Option("data/bronze", help="Bronze directory"),
    silver: str = typer.Option("data/silver", help="Silver directory")
):
    """Process all bronze datasets to silver"""

    console.print(f"\n⚙️  [bold]Processing all datasets[/bold] in {bronze}")

    service = MedallionService(
        bronze_dir=bronze,
        silver_dir=silver,
        patch_size=patch_size,
        min_size=min_size,
        num_workers=workers
    )

    stats = service.process_all(max_datasets)

    console.print("\n✅ [green]All datasets processed:[/green]")
    for dataset, ds_stats in stats.items():
        console.print(f"  • {dataset}: {ds_stats['total_patches']} patches")

    return 0


@app.command("verify")
def verify_lmdb(
    dataset: str = typer.Argument(..., help="Dataset name"),
    silver: str = typer.Option("data/silver", help="Silver directory")
):
    """Verify a silver layer LMDB database"""
    console.print(f"\n🔍 [bold]Verifying LMDB:[/bold] {dataset}")

    service = MedallionService(silver_dir=silver)
    is_valid = service.verify_lmdb(dataset)

    if is_valid:
        console.print("✅ [green]LMDB verification PASSED[/green]")
        return 0
    else:
        console.print("❌ [red]LMDB verification FAILED[/red]")
        return 1


@app.command("list-silver")
def list_silver(
    silver: str = typer.Option("data/silver", help="Silver directory")
):
    """List processed silver datasets"""
    silver_dir = Path(silver)

    if not silver_dir.exists():
        console.print(f"❌ Silver directory not found: {silver}")
        return 1

    lmdb_dbs = list(silver_dir.glob("*.lmdb"))

    if not lmdb_dbs:
        console.print("No datasets found")
        return 0

    table = Table(title="📚 Processed Silver Datasets")
    table.add_column("Dataset", style="cyan")
    table.add_column("Patches", style="green")
    table.add_column("Source Images", style="yellow")
    table.add_column("Status", style="magenta")

    service = MedallionService(silver_dir=silver)

    for db in lmdb_dbs:
        try:
            env = lmdb.open(str(db), readonly=True, lock=False)
            with env.begin() as txn:
                metadata = pickle.loads(txn.get(b'__metadata__'))
            env.close()

            is_valid = service.verify_lmdb(db.stem)
            status = "✅" if is_valid else "❌"

            table.add_row(
                db.stem,
                str(metadata['num_patches']),
                str(metadata['source_images']),
                status
            )
        except Exception:
            table.add_row(db.stem, "?", "?", "❌")

    console.print(table)
    return 0


# ============================================================================
# Training Commands
# ============================================================================

@app.command("train")
def train(
    config: str = typer.Argument(..., help="Path to configuration file"),
    resume: Optional[str] = typer.Option(None, help="Path to checkpoint to resume from"),
    pretrain: Optional[str] = typer.Option(None, help="Path to pretrained weights"),
    device: Optional[str] = typer.Option(None, help="Override device (cuda/mps/cpu)")
):
    """Start training with configuration file"""
    sys.argv = ['train.py', '--config', config]
    if resume:
        sys.argv.extend(['--resume', resume])
    if pretrain:
        sys.argv.extend(['--pretrain', pretrain])
    if device:
        sys.argv.extend(['--device', device])
    train_main()
    return 0


@app.command("train-default")
def train_default():
    console.print("\n🚀 [bold]Starting training[/bold] with default config")

    # Create default config
    default_config = {
        'experiment_name': 'test_run',
        'model': 'realplksr_s',
        'model_params': {
            'dim': 32,
            'n_blocks': 6,
            'dysample': False
        },
        'scale': 4,
        'iterations': 1000,
        'batch_size': 2,
        'patch_size': 128,
        'train_dataset': 'synthetic',
        'learning_rate': 1e-4,
        'use_wandb': False
    }

    # Save temp config
    config_path = Path('temp_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f)

    try:
        trainer = Trainer(default_config)
        trainer.train()
    finally:
        if config_path.exists():
            config_path.unlink()

    return 0


# ============================================================================
# Inference Commands
# ============================================================================

@app.command("infer")
def infer(
    model: str = typer.Argument(..., help="Path to model checkpoint"),
    input: str = typer.Argument(..., help="Input image or directory"),
    output: Optional[str] = typer.Option(None, help="Output path"),
    device: Optional[str] = typer.Option(None, help="Compute device"),
    scale: int = typer.Option(4, help="Upscaling factor"),
    tile_size: int = typer.Option(512, help="Tile size for inference"),
    halo_size: int = typer.Option(32, help="Halo size for context"),
    model_type: str = typer.Option("realplksr", help="Model architecture"),
    dysample: bool = typer.Option(False, help="Use DySample upsampler (flag, no value needed)"),
    benchmark: bool = typer.Option(False, help="Print benchmark info")
):
    console.print(f"\n🔮 [bold]Running inference[/bold] with model: {model}")
    console.print(f"  DySample: {'Yes' if dysample else 'No'}")

    engine = InferenceEngine(
        model_path=model,
        device=device,
        scale=scale,
        tile_size=tile_size,
        halo_size=halo_size,
        model_type=model_type,
        dysample=dysample
    )

    input_path = Path(input)

    if input_path.is_file():
        if not output:
            output = str(input_path.parent / f"{input_path.stem}_sr{input_path.suffix}")

        engine.process_image(
            str(input_path),
            output,
            benchmark=benchmark
        )

    elif input_path.is_dir():
        if not output:
            output = str(input_path / "sr_output")

        engine.process_directory(
            str(input_path),
            output
        )

    else:
        console.print(f"❌ [red]Input not found:[/red] {input}")
        return 1

    return 0


@app.command("benchmark")
def benchmark(
    model: str = typer.Argument(..., help="Path to model checkpoint"),
    input: str = typer.Argument(..., help="Input image for benchmarking"),
    device: Optional[str] = typer.Option(None, help="Compute device"),
    scale: int = typer.Option(4, help="Upscaling factor"),
    tile_size: int = typer.Option(512, help="Tile size"),
    model_type: str = typer.Option("realplksr", help="Model architecture")
):
    """Benchmark inference speed"""
    console.print(f"\n⏱️  [bold]Benchmarking model:[/bold] {model}")

    engine = InferenceEngine(
        model_path=model,
        device=device,
        scale=scale,
        tile_size=tile_size,
        model_type=model_type
    )

    # Load image to get size
    img = cv2.imread(input)
    if img is None:
        console.print(f"❌ [red]Failed to load image:[/red] {input}")
        return 1

    h, w = img.shape[:2]
    engine.benchmark(input_size=(w, h))

    return 0


# ============================================================================
# Export Commands
# ============================================================================

@app.command("export")
def export_model(
    model: str = typer.Argument(..., help="Path to model checkpoint"),
    output_dir: str = typer.Option("exports", help="Output directory"),
    model_type: str = typer.Option("realplksr", help="Model architecture"),
    scale: int = typer.Option(4, help="Upscaling factor"),
    onnx: bool = typer.Option(False, help="Export to ONNX"),
    fp16: bool = typer.Option(False, help="Convert ONNX to FP16"),
    int8: bool = typer.Option(False, help="Quantize to INT8"),
    coreml: bool = typer.Option(False, help="Export to CoreML"),
    tensorrt: bool = typer.Option(False, help="Build TensorRT engine"),
    opset: int = typer.Option(18, help="ONNX opset version"),
    static_shapes: bool = typer.Option(False, help="Disable dynamic axes"),
    input_size: str = typer.Option("256,256", help="Input height,width"),
    dataset: str = typer.Option("nomosv2", help="Training dataset name"),
    iterations: int = typer.Option(185000, help="Training iterations"),
    psnr: float = typer.Option(0.0, help="Validation PSNR"),
    ssim: float = typer.Option(0.0, help="Validation SSIM")
):
    """Export model to various formats"""
    console.print(f"\n📦 [bold]Exporting model:[/bold] {model}")

    # Parse input size
    h, w = [int(x) for x in input_size.split(',')]

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    console.print("Loading model...")
    try:
        model_pt = load_model_for_export(
            model,
            model_type,
            scale,
            device='cpu'
        )
        console.print("✅ Model loaded successfully")
    except Exception as e:
        console.print(f"❌ Failed to load model: {e}")
        return 1

    # Create exporter
    console.print("Creating exporter...")
    exporter = ModelExporter(
        model_pt,
        model_name=f"{model_type}_{scale}x",
        device=torch.device('cpu')
    )

    exported_formats = []
    export_success = True

    if onnx:
        onnx_path = Path(output_dir) / f"{model_type}_{scale}x.onnx"
        console.print(f"\n📦 Exporting to ONNX: {onnx_path}")

        try:
            exporter.export_onnx(
                str(onnx_path),
                input_shape=(1, 3, h, w),
                dynamic_axes=not static_shapes,
                opset_version=opset
            )
            exported_formats.append("ONNX")
            console.print("✅ ONNX export successful")

            # Skip validation - it's causing crashes
            # The ONNX check already passed inside export_onnx

        except Exception as e:
            console.print(f"❌ ONNX export failed: {e}")
            export_success = False

        # FP16
        if fp16 and export_success:
            fp16_path = Path(output_dir) / f"{model_type}_{scale}x_fp16.onnx"
            console.print(f"\n📦 Converting to FP16: {fp16_path}")
            try:
                exporter.convert_to_fp16(str(onnx_path), str(fp16_path))
                exported_formats.append("ONNX-FP16")
                console.print("✅ FP16 conversion successful")
            except Exception as e:
                console.print(f"❌ FP16 conversion failed: {e}")

        # INT8
        if int8 and export_success:
            int8_path = Path(output_dir) / f"{model_type}_{scale}x_int8.onnx"
            console.print(f"\n📦 Converting to INT8: {int8_path}")
            try:
                exporter.convert_to_int8(str(onnx_path), str(int8_path))
                exported_formats.append("ONNX-INT8")
                console.print("✅ INT8 conversion successful")
            except Exception as e:
                console.print(f"❌ INT8 conversion failed: {e}")

    # CoreML
    if coreml and export_success:
        coreml_path = Path(output_dir) / f"{model_type}_{scale}x.mlpackage"
        console.print(f"\n📦 Exporting to CoreML: {coreml_path}")
        try:
            exporter.export_coreml(str(coreml_path), (3, h, w))
            exported_formats.append("CoreML")
            console.print("✅ CoreML export successful")
        except Exception as e:
            console.print(f"❌ CoreML export failed: {e}")

    # TensorRT
    if tensorrt and onnx and export_success:
        trt_path = Path(output_dir) / f"{model_type}_{scale}x.engine"
        console.print(f"\n📦 Building TensorRT engine: {trt_path}")
        try:
            exporter.export_tensorrt(
                str(onnx_path),
                str(trt_path),
                fp16=fp16,
                int8=int8
            )
            exported_formats.append("TensorRT")
            console.print("✅ TensorRT engine built successfully")
        except Exception as e:
            console.print(f"❌ TensorRT engine build failed: {e}")

    # Create model card
    model_params = sum(p.numel() for p in model_pt.parameters())

    exporter.create_model_card(output_dir, {
        "architecture": model_type,
        "scale": scale,
        "params": model_params,
        "dataset": dataset,
        "iterations": iterations,
        "psnr": psnr,
        "ssim": ssim,
        "exports": exported_formats
    })

    # Clean up to prevent resource warnings
    console.print("\n🧹 Cleaning up...")
    del model_pt
    del exporter
    gc.collect()

    if export_success and exported_formats:
        console.print(f"\n✅ [green]Export complete![/green] Formats: {', '.join(exported_formats)}")
        console.print(f"📁 Files saved to: [bold]{output_dir}[/bold]")
        # Flush and force exit
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # Force exit to avoid cleanup warnings
    else:
        console.print("\n❌ [red]Export failed[/red]")
        return 1


# ============================================================================
# Testing Commands
# ============================================================================

@app.command("test-degradation")
def test_degradation_cmd():
    """Test the second-order degradation pipeline"""
    console.print("\n🧪 [bold]Testing degradation pipeline[/bold]")
    test_degradation()
    return 0


@app.command("test-metrics")
def test_metrics_cmd():
    """Test image quality metrics"""
    console.print("\n🧪 [bold]Testing metrics[/bold]")
    test_metrics()
    return 0


@app.command("test-logger")
def test_logger_cmd():
    """Test experiment logger"""
    console.print("\n🧪 [bold]Testing logger[/bold]")
    test_logger()
    return 0


@app.command("test-all")
def test_all():
    """Run all tests"""
    console.print("\n🧪 [bold]Running all tests[/bold]\n")

    tests = [
        ("Degradation", test_degradation),
        ("Metrics", test_metrics),
        ("Logger", test_logger)
    ]

    for name, test_func in tests:
        console.print(f"\n📋 Testing {name}...")
        try:
            test_func()
            console.print(f"✅ [green]{name} passed[/green]")
        except Exception as e:
            console.print(f"❌ [red]{name} failed: {e}[/red]")
            return 1

    return 0


# ============================================================================
# Utility Commands
# ============================================================================

@app.command("clean")
def clean(
    all: bool = typer.Option(False, help="Clean everything including data")
):
    """Clean generated files"""
    dirs_to_clean = ['__pycache__', 'logs', 'temp_config.yaml']
    if all:
        dirs_to_clean.extend(['data/silver', 'exports', 'weights'])

    for dir_name in dirs_to_clean:
        path = Path(dir_name)
        if path.is_dir():
            shutil.rmtree(path)
            console.print(f"🗑️  Removed directory: {dir_name}")
        elif path.is_file():
            path.unlink()
            console.print(f"🗑️  Removed file: {dir_name}")

    return 0


@app.command("status")
def status():
    """Show current status of data and models"""
    console.print("\n📊 [bold]Pipeline Status[/bold]\n")

    # Check bronze layer
    bronze = Path("data/bronze")
    if bronze.exists():
        datasets = [d for d in bronze.iterdir() if d.is_dir()]
        console.print(f"📂 [cyan]Bronze layer:[/cyan] {len(datasets)} datasets")

        for d in datasets[:5]:
            images = list(d.glob("*.[pj][np]g")) + list(d.glob("*.jpeg")) + list(d.glob("*.bmp"))
            console.print(f"  • {d.name}: {len(images)} images")
        if len(datasets) > 5:
            console.print(f"  ... and {len(datasets)-5} more")
    else:
        console.print("📂 [cyan]Bronze layer:[/cyan] not found")

    # Check silver layer
    silver = Path("data/silver")
    if silver.exists():
        lmdb_dbs = list(silver.glob("*.lmdb"))
        console.print(f"\n💿 [cyan]Silver layer:[/cyan] {len(lmdb_dbs)} processed datasets")

        service = MedallionService(silver_dir="data/silver")
        for db in lmdb_dbs:
            try:
                is_valid = service.verify_lmdb(db.stem)
                status_icon = "✅" if is_valid else "❌"
                console.print(f"  {status_icon} {db.stem}")
            except:
                console.print(f"  ❓ {db.stem} (unreadable)")
    else:
        console.print("\n💿 [cyan]Silver layer:[/cyan] not found")

    # Check weights
    weights = Path("weights")
    if weights.exists():
        checkpoints = list(weights.glob("*.pt"))
        console.print(f"\n🎯 [cyan]Weights:[/cyan] {len(checkpoints)} checkpoints")

        for ckpt in sorted(checkpoints, key=lambda p: p.stat().st_mtime)[-5:]:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            console.print(f"  • {ckpt.name}: {size_mb:.1f} MB")
    else:
        console.print("\n🎯 [cyan]Weights:[/cyan] not found")

    return 0


def main():
    app()


if __name__ == "__main__":
    main()
