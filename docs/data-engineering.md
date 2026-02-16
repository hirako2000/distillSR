The data engineering pipeline transforms high-resolution imagery into a machine-learning-ready format while maintaining structural integrity through a tiered dataset approach that isolates raw ingestion from optimized training data.

## Medallion Tier Architecture

The Bronze tier handles raw ingestion, storing original images downloaded from source repositories without modification in standard formats including PNG, JPG, and RAW. This preserves the original source of truth for reproducibility.

The Silver tier contains processed patches. Images are validated, filtered, and cropped into uniform tiles stored in LMDB format for high-performance input-output during training.

The Gold tier exists as a virtual state during training, where clean Silver patches are transformed in real-time through stochastic degradation into the training stream as memory tensors.

## Bronze Ingestion

The ingestion layer synchronizes datasets from Hugging Face Hub to local storage using a dedicated fetcher utility. The implementation scans repository files, filters for supported image extensions, and downloads in parallel with configurable worker count. Downloaded images undergo verification including minimum resolution requirements and corruption detection before acceptance. No filtering occurs at this stage beyond basic verification, preserving the original source of truth. Storage volume must accommodate high-resolution datasets such as Nomos-v2 or LSDIR which can reach hundreds of gigabytes.

## Silver Transformation

The transition from Bronze to Silver is critical for training performance, particularly on Apple Silicon. Images first pass through a validation filter removing any files with resolution below 1024 pixels on the shortest side. Validated images are then segmented into uniform patches of 256 by 256 pixels with stride equal to patch size for non-overlapping extraction. For each patch, horizontal and vertical flips are generated as augmentation when stride equals patch size. Segmenting images into smaller patches increases the variety of local textures available for learning and prevents memory spikes on devices with Unified Memory.

The resulting patches are serialized into a Lightning Memory-Mapped Database. Each dataset maps to a separate LMDB environment with one terabyte virtual map size. Keys follow the pattern dataset name, relative path, row, column, and optional flip suffix. Patches are stored as PNG-compressed byte arrays, trading storage efficiency for decompression CPU cost during training. The database provides memory-mapped reading where the training script accesses image data directly from the system cache, eliminating individual file open and close operations. This cached storage approach is essential for maintaining high iterations per second on Mac hardware.

Metadata including patch count, source images, and creation timestamp is stored under a dedicated metadata key.

## Gold Stream

The Gold layer exists only during training, implementing the second-order degradation logic. When the training loop requests a batch, the data loader pulls clean patches from the Silver LMDB through a SilverDataset class that delays environment initialization until first access. Keys are cached after initial read with fallback to random tensor generation when decode failures occur.

Before patches reach the model, they pass through the degradation engine which applies sequential blur, noise, resize, sinc filtering, and compression operations. Because degradation happens on-the-fly with randomly sampled parameters per iteration, the model sees a different version of the data in every epoch even when the underlying Silver patches remain unchanged. This prevents overfitting and ensures restoration capabilities generalize across varied real-world image damage.

## Infrastructure Requirements

The pipeline is tested with SSD storage, NVMe preferred, for rapid LMDB mapping. The one terabyte virtual memory map prevents database expansion crashes during dataset ingestion. Patch sizes between 128 and 256 pixels are optimized for the receptive field of the PLKSR architecture.