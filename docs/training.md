Training is an iterative optimization process using supervised learning. The model makes a prediction and we measure its error against ground truth, then update weights to reduce that error.

## Loop

Each iteration begins with stochastic degradation where a clean patch from the Silver tier transforms into a Gold sample using the randomized pipeline of blur, noise, resize, sinc filtering, and compression. The model processes this degraded sample in a forward pass to produce a restored version. Loss calculation quantifies the difference between the model output and the original Silver patch using one or more loss functions. Backpropagation sends this error backward through the network to update internal weights using the AdamW optimizer.

## Loss functions

To achieve perceptual integrity, the model considers image structure beyond individual pixels. L1 loss or mean absolute error measures direct pixel-to-pixel difference, ensuring color accuracy and general brightness. MS-SSIM or multi-scale structural similarity measures similarity in contrast, luminance, and structure, preventing the model from smearing fine details. Perceptual loss uses a pre-trained network such as VGG to ensure the restored image appears correct to human vision even when pixels do not match exactly.

## Training or Fine-tuning

The starting point of model weights distinguishes training from fine-tuning. Training from scratch begins with random weights and requires millions of iterations with substantial compute resources. Fine-tuning starts with pre-trained weights and adjusts them to specialize in a new domain such as ultra-heavy JPEG compression or specific illustration styles. This approach is significantly faster and feasible on consumer hardware including Apple Silicon.

## Convergence

Training reaches completion when the loss curve flattens, a state called convergence. To ensure the model has not simply memorized the Silver patches, a validation pass runs every five thousand iterations using images the model has never seen during training. This validation monitors generalization performance and triggers checkpoint saving when metrics improve.

## HOWTO

Running train.py launches a supervised learning factory. The script handles Gold degradation in real time, creating an boundless dataset where no two degraded images are identical. This forces the model to learn the physics of restoration rather than memorizing a limited set of examples.

For more on training, read [inference-tiling](./inference-tiling.md)