## Resizing and Compression

In the Gold tier, resizing and compression act as the primary downsamplers. Resizing reduces raw pixel count while compression simulates information loss caused by digital encoders. Training on these artifacts helps the model recover sharp edges from blurry blocks and distinguish between natural textures and encoding noise.

### Multi-Interpolation Resizing Logic

To prevent the model from overfitting to a single downsampling algorithm such as bicubic, the pipeline randomly selects from a pool of interpolation methods for every batch. Nearest neighbor sampling with constant time complexity simulates low-quality scaling and pixelation. Bilinear interpolation with quadratic complexity simulates standard soft scaling found in older mobile devices. Bicubic interpolation with cubic complexity serves as the industry standard, teaching the model high-fidelity reconstruction. Area resampling with complexity proportional to pixel area simulates browser-based resizing and prevents aliasing in the Gold output. The scale factor is sampled from a continuous range between one and four.

### JPEG Compression

JPEG compression is a lossy process based on the Discrete Cosine Transform. It introduces three typical artifacts that the model is specifically designed to remove.

Blocking artifacts appear when JPEG divides images into eight by eight pixel blocks. At low quality factors, boundaries between these blocks become visible. The model learns to treat these as discontinuity noise rather than image edges.

Ringing and mosquito noise manifest as shimmering or halos around high-contrast edges such as text or fine lines. This is caused by quantization of high-frequency DCT coefficients. Mosquito noise appears near sharp edges in busy textures while ringing produces oscillations parallel to main edges.

Color smearing occurs because JPEG uses YCbCr color space with chroma subsampling typically in four two zero format. This causes color information to have half the resolution of brightness information. The Gold tier simulates this by randomly applying chroma subsampling before the final JPEG pass.

### Quality Curve

The quality factor is sampled stochastically to cover the widest possible range of internet damage. Values between 90 and 100 represent high quality for fine-tuning subtle detail retention. Values between 50 and 90 represent standard web quality for removal of mosquito noise and slight blurring. Values between 10 and 50 represent heavy compression for reconstruction from severe blockiness and color loss.

### Second-Order Loop

In the final pipeline, compression is applied twice following the formulation Output equals JPEG of Sinc of Resize of JPEG of Clean with quality factor Q1 and scale factor r, then quality factor Q2. The first pass applies higher quality between 75 and 95 to simulate original encoding. The second pass applies lower quality between 30 and 85 to simulate degradation caused by re-uploading or screenshotting.

Once you've digested all of this data peraration processes, time to read on the actual [training](./training.md) aspects.