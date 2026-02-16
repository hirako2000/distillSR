# justfile - RealPLKSR Factory Command Runner
# Usage: just <command> [arguments]

# Help
default:
    @just --list

# Install dependencies and setup environment
setup:
    uv pip install -r requirements.txt
    mkdir -p data/bronze data/silver weights logs exports

# ============================================================================
# Data Pipeline
# ============================================================================

# List available datasets on Hugging Face
list-datasets:
    uv run cli.py list-datasets

# Download dataset from Hugging Face
download dataset="nomosv2" min_size="512" output="data/bronze" workers="4":
    uv run cli.py download {{dataset}} --min-size {{min_size}} --output {{output}} --workers {{workers}}

# Create synthetic dataset for testing
synthetic num="100" output="data/bronze/synthetic":
    uv run cli.py synthetic --num {{num}} --output {{output}}

# Process dataset to silver LMDB
process dataset='nomos_uni_gt' patch_size='256' min_size='256' workers='4':
    uv run cli.py process {{dataset}} --patch-size {{patch_size}} --min-size {{min_size}} --workers {{workers}}

# Process all datasets to silver
process-all patch_size="256" min_size="1024" workers="4":
    uv run cli.py process-all --patch-size {{patch_size}} --min-size {{min_size}} --workers {{workers}}

# Verify LMDB database
verify dataset:
    uv run cli.py verify {{dataset}}

# List processed silver datasets
list-silver:
    uv run cli.py list-silver

# Aliases for convenience
inject-bronze:
    just download

inject-silver:
    just process

inject-gold:
    @echo "Gold tier is generated on-the-fly during training"

# Quick download and process
get dataset="nomosv2":
    just download {{dataset}}
    just process {{dataset}}

# ============================================================================
# Training
# ============================================================================

# Train with config file
train config="configs/train_config.yaml" resume="" pretrain="":
    @if [ -n "{{resume}}" ]; then \
        uv run cli.py train {{config}} --resume {{resume}}; \
    elif [ -n "{{pretrain}}" ]; then \
        uv run cli.py train {{config}} --pretrain {{pretrain}}; \
    else \
        uv run cli.py train {{config}}; \
    fi

# Train with fast config file
train-fast config="configs/train_fast_config.yaml" resume="" pretrain="":
    @if [ -n "{{resume}}" ]; then \
        uv run cli.py train {{config}} --resume {{resume}}; \
    elif [ -n "{{pretrain}}" ]; then \
        uv run cli.py train {{config}} --pretrain {{pretrain}}; \
    else \
        uv run cli.py train {{config}}; \
    fi

# Train with default config (quick test)
train-default:
    uv run cli.py train-default

# Resume training from checkpoint
resume checkpoint config="configs/train_config.yaml":
    uv run cli.py train --config {{config}} --resume {{checkpoint}}

# Fine-tune from pretrained
finetune pretrain config="configs/train_config.yaml":
    uv run cli.py train {{config}} --pretrain {{pretrain}}

# ============================================================================
# Inference
# ============================================================================

# Run inference on image
infer model input output="" tile_size="512" device="":
    @if [ -n "{{output}}" ] && [ -n "{{device}}" ]; then \
        uv run cli.py infer {{model}} {{input}} --output {{output}} --tile-size {{tile_size}} --device {{device}} --dysample; \
    elif [ -n "{{output}}" ]; then \
        uv run cli.py infer {{model}} {{input}} --output {{output}} --tile-size {{tile_size}} --dysample; \
    elif [ -n "{{device}}" ]; then \
        uv run cli.py infer {{model}} {{input}} --tile-size {{tile_size}} --device {{device}} --dysample; \
    else \
        uv run cli.py infer {{model}} {{input}} --tile-size {{tile_size}} --dysample; \
    fi

infer-pixel model input output="" tile_size="512" device="":
    @if [ -n "{{output}}" ] && [ -n "{{device}}" ]; then \
        uv run cli.py infer {{model}} {{input}} --output {{output}} --tile-size {{tile_size}} --device {{device}}; \
    elif [ -n "{{output}}" ]; then \
        uv run cli.py infer {{model}} {{input}} --output {{output}} --tile-size {{tile_size}}; \
    elif [ -n "{{device}}" ]; then \
        uv run cli.py infer {{model}} {{input}} --tile-size {{tile_size}} --device {{device}}; \
    else \
        uv run cli.py infer {{model}} {{input}} --tile-size {{tile_size}}; \
    fi

# Run inference on directory
infer-dir model input_dir output_dir="sr_output" tile_size="512":
    uv run cli.py infer --model {{model}} --input {{input_dir}} --output {{output_dir}} --tile-size {{tile_size}}

# Benchmark inference speed
benchmark model input device="":
    uv run cli.py benchmark --model {{model}} --input {{input}} --device {{device}}

# Quick upscale (default model)
upscale input output="" model="weights/model_best.pt":
    just infer {{model}} {{input}} {{output}}

# ============================================================================
# Model Export
# ============================================================================

# Export to ONNX with options
export-onnx model output_dir="exports" static="false" input_size="256,256" fp16="false" int8="false" coreml="false" tensorrt="false":
    @if [ "{{static}}" = "true" ]; then \
        uv run cli.py export {{model}} --output-dir {{output_dir}} --onnx --static-shapes --input-size {{input_size}} \
        {% if fp16 == "true" %}--fp16{% endif %} \
        {% if int8 == "true" %}--int8{% endif %} \
        {% if coreml == "true" %}--coreml{% endif %} \
        {% if tensorrt == "true" %}--tensorrt{% endif %}; \
    else \
        uv run cli.py export {{model}} --output-dir {{output_dir}} --onnx \
        {% if fp16 == "true" %}--fp16{% endif %} \
        {% if int8 == "true" %}--int8{% endif %} \
        {% if coreml == "true" %}--coreml{% endif %} \
        {% if tensorrt == "true" %}--tensorrt{% endif %}; \
    fi

# Export all formats
export-all model:
    uv run cli.py export {{model}} --onnx --fp16 --int8 --coreml

# Export only INT8
export-int8 model:
    uv run cli.py export {{model}} --onnx --int8

# Export to CoreML
export-coreml model output_dir="exports":
    uv run cli.py export --model {{model}} --output-dir {{output_dir}} --coreml

# Export to TensorRT
export-tensorrt model output_dir="exports":
    uv run cli.py export --model {{model}} --output-dir {{output_dir}} --onnx --tensorrt --fp16

# Create model card only
model-card model output_dir="exports" dataset="nomosv2" psnr="0" ssim="0":
    uv run cli.py export --model {{model}} --output-dir {{output_dir}} --dataset {{dataset}} --psnr {{psnr}} --ssim {{ssim}}

# ============================================================================
# Testing / Linting
# ============================================================================

# Run all checks
check: lint analysis test

# Fix all auto-fixable linting and formatting issues
fix:
    uv run ruff check --fix .

# Check linting and formatting without fixing 
lint:
    uv run ruff check .

# Run static type analysis
analysis:
    uv run pyright

# Run tests with coverage
test:
    pytest tests/ \
        --cov=. \
        --cov-report=term-missing \
        --cov-report=html:coverage_html \
        --cov-report=xml:coverage.xml \
        -v

# Run tests with coverage and generate report
test-coverage:
    pytest tests/ \
        --cov=. \
        --cov-report=term-missing \
        --cov-report=html:coverage_html \
        --cov-report=xml:coverage.xml \
        -v --cov-fail-under=80

# Run specific test file
test-file f:
    pytest tests/test_{{f}}.py -v --cov=. --cov-append

# Test degradation pipeline
test-degradation:
    uv run cli.py test-degradation

# Test metrics
test-metrics:
    uv run cli.py test-metrics

# Test logger
test-logger:
    uv run cli.py test-logger

# Run all tests
test-all:
    uv run cli.py test-all

benchmark-mps:
    just benchmark device="mps"

benchmark-cpu:
    just benchmark device="cpu"

# Benchmarks (speed)
benchmark-auto  model-name="4xNomos2_realplksr_dysample" models-dir="weights" images-dir="tests/images" output-dir="tests/images-output":
    @mkdir -p {{images-dir}} {{output-dir}}
    @uv run pytest tests/test_benchmark.py  \
        --model-name {{model-name}} \
        --models-dir {{models-dir}} \
        --images-dir {{images-dir}} \
        --output-dir {{output-dir}} \
        -v

# ============================================================================
# Utilities
# ============================================================================

# Clean generated files
clean all="false":
    @if [ "{{all}}" = "true" ]; then \
        uv run cli.py clean --all; \
    else \
        uv run cli.py clean; \
    fi

# Clean everything (warning: wipes data and weights)
clean-all:
    uv run cli.py clean --all

# Show pipeline status
status:
    uv run cli.py status

# Show help
help:
    @just --list

# Quick test pipeline
quick-test:
    just synthetic num=10
    just process synthetic patch_size=128 min_size=256
    just train-default