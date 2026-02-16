# Stochastic Blurring

Blurring in the Gold tier is not static. To ensure the model can restore images from various camera sensors, we implement a stochastic selection of blur kernels with parameters sampled per iteration.

Isotropic Gaussian blur applies equal sigma in all directions with values sampled from 0.5 to 4.0. This models general out-of-focus optics.

Anisotropic Gaussian blur applies independent sigma values for horizontal and vertical axes, sampled from 0.5 to 6.0. This simulates directional motion blur or lens astigmatism.

Generalized Gaussian blur adjusts the distribution peak sharpness through shape parameter beta from 0.5 to 2.0. Beta below one produces heavier tails, modeling complex optical diffraction patterns.

Plateau blur creates a flat central region with Gaussian falloff. Plateau ratio between 0.3 and 0.7 determines flat portion size relative to sigma sampled from 1.0 to 5.0. This simulates sensor blooming or over-exposure artifacts.

# Multi-Modal Noising

Noise is injected into clean Silver patches to simulate the electrical interference found in low-light digital photography. Two noise models are applied stochastically with parameters randomized per batch.

## Gaussian Noise

Johann Carl Friedrich Gauss is everywhere. Gaussian noise simulates the standard electronic hiss of a digital sensor. It is additive and follows a normal distribution. In the Gold tier, standard deviation is randomized between 0.01 and 0.1 for every batch, ensuring the model can handle both subtle grain and heavy static. This models sensor read noise and electronic interference.

## Poisson Noise

Siméon Denis Poisson provides the method for simulating shot noise, a quantum effect of light. Unlike Gaussian noise, Poisson noise is signal-dependent and more prominent in darker image regions. The implementation scales the image by a factor between 10 and 100 to simulate photon counts, applies Poisson sampling, then rescales. When Poisson sampling fails due to numerical instability, the implementation falls back to Gaussian noise. This is useful for training the model to clean shadows without destroying texture in well-lit areas.

We also do resize compression. Read about it [here](./gold-resize-compress.md).