The project includes pytest-based test suites for validating model performance and benchmarking inference speed.

## Structure

Tests are organized in the `tests/` directory with conftest.py providing fixtures and test_benchmark.py containing the main benchmarking logic. The test framework accepts command-line parameters for model name, model directories, image paths, and output locations, making it adaptable to different testing scenarios.

## Benchmark

The model_name fixture retrieves the model identifier from command-line. The models_dir and pytorch_models_dir fixtures locate directories containing ONNX and PyTorch models respectively. The test_images fixture discovers all images in the specified directory with common extensions including jpg, png, etc.

## Timing execution

The benchmark test runs the model through the CLI inference command for each test image, measuring total execution time. Results include success status, output paths, and timing information for each image processed. After processing all images, the test calculates statistics including mean, median, minimum, maximum, and standard deviation of inference times.

## Result Aggregation

All benchmark results are saved to a JSON file in the output directory, containing timestamp, model information, system details, per-image results, and aggregate statistics. This structured output enables automated analysis and comparison across different model versions or hardware configurations.