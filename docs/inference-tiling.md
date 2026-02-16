# inference tiling

During the restoration of high-resolution images (e.g., 4K or 8K), the memory required for the model's feature maps exceeds the available Unified Memory (or VRAM) found on consumer hardware. Tiling allows the model to process arbitrarily large images by breaking them into smaller segments, restoring them individually, and reassembling them into a seamless result.

### Workflow

To maintain structural continuity, tiles are not processed in isolation. We use an **overlapping window** strategy.

| Phase | Action | Purpose |
| --- | --- | --- |
| **Slicing** | Divide input into tiles of size `S x S `. | Fits within hardware memory limits. |
| **Padding** | Add a border (Halo) of `P` pixels to each tile. | Provides the model with "context" for edge pixels. |
| **Inference** | Run the trained model on each padded tile. | Executes the restoration logic. |
| **Cropping** | Remove the extra `P` pixels from the output. | Discards edge artifacts created by the model. |
| **Blending** | Apply a linear or Gaussian ramp to overlapping edges. | Ensures a seamless transition between tiles. |

### Halo (and receptive field)

Every convolution-based model has a "Receptive Field"—the maximum area of input pixels that influence a single output pixel. If a tile is processed without its surrounding context, the model "guesses" at the edges, leading to visible grid lines (seams).

To prevent this, we define a **Halo** or Ghost Region technique.

```math
\text{Tile}_{\text{input}} = \text{Target}_{\text{area}} + \text{Halo}_{\text{context}}
```

For a model with a deep architecture, a halo of `32` to `64` pixels is typically sufficient to ensure the central target area is restored with 100% accuracy.

### Blending Equation

When two adjacent tiles `T_1` and `T_2` overlap in a region of width `W`, we don't just "cut and paste." We blend them using a weight matrix `M`:

```math
\text{Result} = T_1 \cdot (1 - M) + T_2 \cdot M

```

Where `M` is a gradient (it's just a ramp/cliff) from `0.0` to `1.0` across the overlap. This ensures that any slight variation in color or texture between tile passes is visually neutralized.

### Resource requirements

Optimize the tile size based on the specific memory constraint to maximize throughput.

| Memory Pool | Recommended Tile Size | Rationale |
| --- | --- | --- |
| **8GB - 16GB** | `512 x 512` | Prevents system-wide slowdowns during inference. |
| **24GB - 32GB** | `1024 x 1024` | Optimal balance between speed and memory overhead. |
| **64GB+** | `2048 x 2048` | Maximum efficiency for large-scale batch processing. |



### Other aspects

Considered, may have been removed..

1. **Coordinate Mapper:** Calculate the top-left indices for all tiles based on image dimensions.
2. **Padding Logic:** Handle edge-cases where tiles are near the image boundary using "Reflect Padding."
3. **Reconstruction Buffer:** Initialize an empty tensor to accumulate the results.
4. **Feathering:** Apply a 10% overlap feathering to the edges of the reconstruction.
