The training pipeline combines multiple perceptual and structural losses to achieve high-quality restoration. 

Each loss targets different aspects of image quality, from pixel-level accuracy to perceptual coherence, and all are composed through a registry pattern for flexible combinations per experiment.

## L1 Loss

[L1 loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.l1_loss.html) measures mean absolute error between restored and ground truth pixels. This ensures color accuracy and brightness preservation with weight one point zero. Unlike L2 loss, L1 penalizes large errors linearly, preventing excessive smoothing of edges and textures.

## MS-SSIM Loss

A type of [Strutural Similarity measure](https://en.wikipedia.org/wiki/Structural_similarity_index_measure), multi-scale structural similarity operates across image pyramid levels to measure coherence in contrast, luminance, and structure. 

It is here implemented as one minus MS-SSIM, this loss prevents the model from [smearing](https://en.wikipedia.org/wiki/Smear_(optics)) fine details while maintaining overall image structure. Weight is configurable, typically set to one point zero when enabled. It was a trial and error that led to this with the datasets used.

## Consistency Loss

Operating in [Oklab color space](https://en.wikipedia.org/wiki/Oklab_color_space), consistency loss enforces color and luminance coherence independently. The implementation separates luminance through CIE L* conversion and chroma through Oklab transformation, then applies separate loss terms to each component. 

Using L1 by [Luminance loss](https://en.wikipedia.org/wiki/Luminance) while chroma loss combines L1 with cosine similarity to preserve color relationships. An optional Gaussian blur can be applied before conversion to focus on larger-scale color consistency.

## Local detail loss

Local detail loss computes gradient differences between restored and ground truth images. Horizontal and vertical gradients are extracted through finite differences, then L1 loss is applied to each gradient map. This preserves edge sharpness and texture details that might be smoothed by pixel-level losses alone.

## Frequency domain loss

Frequency domain loss operates on [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) magnitudes rather than spatial pixels. 

Both restored and ground truth images are transformed using rfft2 with orthonormal normalization, then L1 loss is applied to the magnitude spectra. This maintains spectral characteristics and helps the model recover correct frequency distributions, particularly important for handling compression artifacts.


## Rational

For super-resolution, a combination of L1, MS-SSIM, and consistency losses provides a strong baseline. Adding local detail and frequency domain losses improves edge sharpness and spectral accuracy at the cost of slower training. The Adan optimizer described in [adan-optimizer](./adan-optimizer.md) pairs well with multiple losses due to its stability with complex gradient signals.