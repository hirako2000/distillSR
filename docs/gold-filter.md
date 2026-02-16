# Sinc-Filter Technical Specification

The sinc filter simulates ringing and overshoot artifacts characteristic of digital sharpening and bandwidth-limited imaging systems. Its implementation uses the first-order Bessel function for isotropic 2D convolution with Lanczos windowing.

The 2D isotropic sinc kernel K at coordinates (x, y) is defined as K(x, y) = (ω / 2π√(x² + y²)) J₁(ω√(x² + y²)), where J₁ is the first-order Bessel function and ω is the cutoff frequency. For discrete implementation, kernel size ranges from 15 to 21 pixels, always odd to ensure a central anchor point. Cutoff frequency is sampled between 0.5 and 0.8.

Lanczos windowing mitigates Gibbs phenomenon at kernel edges by applying a sinc-based window function that tapers the kernel smoothly to zero at boundaries. The sinc filter is applied exclusively during the second degradation pass to simulate re-upload ringing artifacts.

# High-Order Degradation

The Gold tier implements a second-order stochastic process where each training iteration generates a unique low-resolution image from a high-resolution ground truth through a degradation function D composed of two sequential passes following the formulation x = D₂(D₁(y)).

The first degradation pass D₁ simulates original encoding damage. Blur is applied through random selection from Gaussian, anisotropic Gaussian, generalized Gaussian, or plateau kernels with parameters sampled per iteration. Downsampling follows using stochastic interpolation randomly selected from nearest, bilinear, bicubic, area, or Lanczos methods. Additive Gaussian or Poisson noise is injected, then JPEG compression applies a quality factor between 75 and 95 with optional chroma subsampling.

The second degradation pass D₂ simulates re-upload damage. A sinc filter introduces ringing with kernel size and cutoff frequency sampled independently. A second blur of any type may be applied. Secondary scaling resizes the image to final target dimensions matching the ground truth divided by scale factor. A second noise injection applies, followed by final JPEG compression with quality factor between 30 and 85. The order of sinc filtering and final JPEG compression is randomized per iteration to increase degradation diversity.

# Silver Layer Storage

The Silver layer stores pre-validated image patches in LMDB format for high-throughput training access, balancing storage efficiency with decompression CPU cost during training.

Images undergo resolution verification requiring minimum 1024 pixels on the shortest side and corruption detection via PIL verification. Invalid images are logged and skipped. Valid images are segmented into 256 by 256 pixel patches with stride equal to patch size for non-overlapping extraction. For each patch, horizontal and vertical flips are generated as augmentation.

Each dataset maps to a separate LMDB environment with 1TB virtual map size. Keys follow the pattern dataset name, relative path, row, column, and optional flip suffix. Patches are serialized as PNG-compressed byte arrays, trading storage efficiency for decompression CPU cost during training. Metadata including patch count, source images, and creation timestamp is stored under a metadata key.

The SilverDataset class delays LMDB environment initialization until first access. Keys are cached after initial read, with fallback to random tensor generation when decode failures occur.

# Data Optimized Model

The model implements a Partial Large Kernel network, balancing receptive field expansion with computational efficiency through channel-wise kernel application.

Partial Large Kernel Convolution applies large kernels of size 13 by 13 or 17 by 17 to only a subset of channels determined by the split ratio defaulting to 0.25. The remaining channels pass through unchanged, preserving global context while reducing parameter count. Optional residual scaling can be applied to the output of each block to improve training stability in deeper networks.

The Doubled Convolutional Channel Mixer begins each PLKBlock by expanding channels by factor two through three by three convolutions, applies Mish activation, then projects back to original dimensions for local texture extraction.

Following large kernel convolution, an Element-wise Attention module applies channel-wise gating through Sigmoid-activated three by three convolutions, providing instance-dependent modulation of features.

Each block incorporates Group Normalization with four groups, applied after refinement convolution and before residual connection. Scale and bias parameters are constant initialized for training stability.

The DySample Upsampler replaces PixelShuffle with content-aware upsampling when enabled. A hyper-network generates sampling offsets delta, allowing the model to learn non-uniform sampling patterns following the formulation I_SR(p) equals sum of I_LR(p plus delta p). For MPS compatibility, the DySample forward method can be patched with a device-aware implementation that avoids unsupported operations. When running on MPS, DySample automatically falls back to PixelShuffle to maintain compatibility while sacrificing some upsampling quality.

Mish activation applies the function x times tanh of softplus of x throughout the channel mixer.

# Training

Training balances multiple loss objectives with platform-specific optimizations for stability across CUDA and MPS backends.

The optimizer can be selected between AdamW and Adan with schedule-free training. The Adan optimizer, when enabled, applies three momentum buffers with betas 0.98, 0.92, and 0.99, and incorporates linear warmup steps before cosine annealing. Learning rate and weight decay are explicitly cast to float from config values to handle string inputs. A cosine annealing scheduler reduces learning rate to a minimum value over total iterations, stepped per batch rather than per epoch.

Gradient clipping at default threshold of one point zero applies global norm clipping after backward pass. Mixed precision training is configurable through the use_amp flag with GradScaler when active, disabled by default for MPS stability and enabled for CUDA throughput.

Loss functions combine multiple objectives. L1 loss with weight one point zero ensures color accuracy and brightness preservation. MS-SSIM loss with weight one point zero when enabled computes as one minus MS-SSIM to maximize structural preservation. Consistency loss operating in Oklab color space enforces color and luminance coherence between restored and ground truth images. Local detail loss computes gradient differences to preserve edge sharpness. Frequency domain loss compares Fourier transform magnitudes to maintain spectral characteristics. These losses are composed through a registry pattern, allowing flexible combinations per experiment.

Platform adaptations include reducing dataloader workers to zero on macOS with MPS as multiprocessing with LMDB can cause instability. Pin_memory is disabled as MPS does not support pinned memory transfers. For MPS execution, the DySample forward method can be replaced with a device-aware implementation that coordinates grid generation explicitly on the MPS device.