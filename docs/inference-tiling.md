# inference tiling

During the restoration of high-resolution images such as 4K or 8K, the memory required for the model's feature maps exceeds the available Unified Memory or VRAM found on consumer hardware. Tiling allows the model to process arbitrarily large images by breaking them into smaller segments, restoring them individually, and reassembling them into a seamless result.

### Workflow

To maintain structural continuity, tiles are not processed in isolation. An overlapping window strategy preserves context across tile boundaries.

| Phase | Action | Purpose |
| --- | --- | --- |
| **Slicing** | Divide input into tiles of size `S x S `. | Fits within hardware memory limits. |
| **Padding** | Add a border of `P` pixels to each tile. | Provides the model with context for edge pixels. |
| **Inference** | Run the trained model on each padded tile. | Executes the restoration logic. |
| **Cropping** | Remove the extra `P` pixels from the output. | Discards edge artifacts created by the model. |
| **Blending** | Average overlapping regions with weight normalization. | Ensures smooth transitions between tiles. |

### Halo and Receptive Field

Every convolution-based model has a receptive field, the maximum area of input pixels that influence a single output pixel. If a tile is processed without its surrounding context, the model lacks sufficient information at the edges, leading to visible grid lines or seams.

To prevent this, a halo or ghost region provides surrounding context. The relationship follows:

```math
\text{Tile}_{\text{input}} = \text{Target}_{\text{area}} + \text{Halo}_{\text{context}}
```

For a model with deep architecture, a halo of 32 to 64 pixels typically ensures the central target area is restored with full receptive field coverage.

### Blending

When two adjacent tiles overlap in a region, simple averaging of accumulated outputs divided by weight count produces seamless transitions. The implementation accumulates pixel values from each tile into an output buffer while tracking a weight buffer that counts contributions per pixel. After all tiles are processed, the output buffer is divided element-wise by the weight buffer. This averaging approach, while simpler than gradient-based blending, effectively neutralizes variations between tile passes without introducing additional artifacts.

### Resource Requirements

Tile size optimization balances memory constraints against throughput. The default tile size of 512 pixels with 32-pixel halo processes approximately 600 by 600 pixel regions per forward pass, limiting peak memory usage to roughly two gigabytes for FP32 inference on a batch size of one.

| Memory Pool | Recommended Tile Size | Rationale |
| --- | --- | --- |
| **8GB to 16GB** | `512 x 512` | Prevents system-wide slowdowns during inference. |
| **24GB to 32GB** | `1024 x 1024` | Optimal balance between speed and memory overhead. |
| **64GB and above** | `2048 x 2048` | Maximum efficiency for large-scale batch processing. |

### Implementation details

The tiling implementation calculates tile coordinates based on image dimensions and configured tile size. Edge cases near image boundaries use reflect padding to maintain consistency. A reconstruction buffer accumulates results with associated weights, and after all tiles are processed, the buffer is normalized by the weight map to produce the final output. For MPS execution, all operations remain on device to avoid costly CPU-GPU transfers during tile assembly.