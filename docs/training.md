# Training

Training is an iterative optimization process using supervised learning. The model makes a prediction and we measure its error against ground truth, then update weights to reduce that error.

## Loop

Each iteration begins with stochastic degradation where a clean patch from the Silver tier transforms into a Gold sample using the randomized pipeline of blur, noise, resize, sinc filtering, and compression. The model processes this degraded sample in a forward pass to produce a restored version. Loss calculation quantifies the difference between the model output and the original Silver patch using one or more loss functions. Backpropagation sends this error backward through the network to update internal weights. The optimizer can be selected between AdamW and Adan with schedule-free training, the latter incorporating three momentum buffers and linear warmup before cosine annealing.

## Loss functions

To achieve perceptual integrity, the model considers image structure beyond individual pixels through multiple complementary loss objectives. L1 loss measures direct pixel-to-pixel difference, ensuring color accuracy and general brightness. MS-SSIM or multi-scale structural similarity measures similarity in contrast, luminance, and structure, preventing the model from smearing fine details. Consistency loss operating in Oklab color space enforces color and luminance coherence between restored and ground truth images. Local detail loss computes gradient differences to preserve edge sharpness. Frequency domain loss compares Fourier transform magnitudes to maintain spectral characteristics. These losses are composed through a registry pattern, allowing flexible combinations per experiment with configurable weights.

## Optimizer and Scheduling

The optimizer configuration supports both AdamW and Adan with schedule-free training. When Adan is enabled, three momentum buffers with betas 0.98, 0.92, and 0.99 track gradient history, and linear warmup steps precede cosine annealing. Learning rate and weight decay are explicitly cast to float from config values to handle string inputs. The cosine annealing scheduler reduces learning rate to a minimum value over total iterations, stepped per batch rather than per epoch. Gradient clipping at a configurable threshold applies global norm clipping after each backward pass.

## Training or Fine-tuning

The starting point of model weights distinguishes training from fine-tuning. Training from scratch begins with random weights and requires millions of iterations with substantial compute resources. Fine-tuning starts with pre-trained weights and adjusts them to specialize in a new domain such as ultra-heavy JPEG compression or specific illustration styles. This approach is significantly faster and feasible on consumer hardware including Apple Silicon. When fine-tuning, the architecture must match the pretrained model exactly, including the choice of upsampler between DySample and PixelShuffle.

## Platform Adaptations

Training behavior adapts to the underlying hardware for stability and performance. On macOS with MPS, dataloader workers are reduced to zero as multiprocessing with LMDB can cause instability, and pin_memory is disabled as MPS does not support pinned memory transfers. Mixed precision training through the use_amp flag with GradScaler is available for CUDA throughput but disabled by default on MPS. For MPS execution, the DySample forward method can be patched with device-aware implementations that avoid unsupported operations.

## Convergence

Training reaches completion when the loss curve flattens, a state called convergence. To ensure the model has not simply memorized the Silver patches, a validation pass runs at configurable intervals using images the model has never seen during training. Validation employs simple bicubic downsampling rather than the full degradation pipeline to establish a consistent baseline. This validation monitors generalization performance through PSNR and SSIM metrics and triggers checkpoint saving when metrics improve.

## HOWTO

Running train.py launches a supervised learning factory. The script handles Gold degradation in real time, creating a boundless dataset where no two degraded images are identical. This forces the model to learn the physics of restoration rather than memorizing a limited set of examples. Configuration is managed through YAML files that specify model architecture, optimizer settings, loss weights, and dataset paths. The training loop integrates with Weights and Biases for experiment tracking when enabled, and maintains both full checkpoints for resuming and lightweight weight-only exports for deployment.

For more on training, read [inference-tiling](./inference-tiling.md)