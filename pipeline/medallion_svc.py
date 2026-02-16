# pipeline/medallion_svc.py
"""
Medallion Service: Bronze → Silver transformation
Validates images, extracts patches, and stores in LMDB for zero-copy access
"""

import os
import sys
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import io
import pickle
import time

import cv2
import numpy as np
from PIL import Image
import lmdb
from tqdm import tqdm
import torch
from torchvision import transforms

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedallionService:
    """Bronze → Silver transformation service with LMDB storage"""
    
    def __init__(
        self,
        bronze_dir: str = "data/bronze",
        silver_dir: str = "data/silver",
        patch_size: int = 256,
        stride: Optional[int] = None,
        min_size: int = 1024,
        quality_threshold: float = 0.95,  # Minimum SSIM-like quality for validation
        num_workers: int = 4,
        map_size: int = 1099511627776,  # 1TB virtual map size
    ):
        """
        Args:
            bronze_dir: Directory containing raw images
            silver_dir: Directory for LMDB database
            patch_size: Size of extracted patches (e.g., 256)
            stride: Stride for patch extraction (if None, uses patch_size for non-overlapping)
            min_size: Minimum image dimension (short side) for validation
            quality_threshold: Threshold for image quality validation
            num_workers: Number of parallel workers
            map_size: LMDB map size (default 1TB virtual)
        """
        self.bronze_dir = Path(bronze_dir)
        self.silver_dir = Path(silver_dir)
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.min_size = min_size
        self.quality_threshold = quality_threshold
        self.num_workers = num_workers
        self.map_size = map_size
        
        # Create directories
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'total_patches': 0,
            'skipped_patches': 0,
            'processing_time': 0
        }
        
    # In pipeline/medallion_svc.py, let's fix the scan_bronze method:

    def scan_bronze(self) -> List[Path]:
        """Scan bronze directory for all image files"""
        image_files = []
        
        # Make sure the dataset directory exists
        dataset_path = self.bronze_dir
        if not dataset_path.exists():
            logger.error(f"Bronze directory not found: {dataset_path}")
            return []
        
        # Recursively find all images
        for ext in self.image_extensions:
            # Case-insensitive glob
            image_files.extend(dataset_path.rglob(f"*{ext}"))
            image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = sorted(set(image_files))
        
        logger.info(f"Found {len(image_files)} images in {dataset_path}")
        if len(image_files) > 0:
            logger.info(f"First few: {[f.name for f in image_files[:5]]}")
        
        return image_files
    
    def validate_image(self, image_path: Path) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Validate image meets minimum requirements"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Check minimum size
                if min(width, height) < self.min_size:
                    logger.debug(f"Image {image_path} too small: {width}x{height} < {self.min_size}")
                    return False, None
                
                # Check if image can be loaded properly
                img.load()
                
                # Additional quality checks could go here
                # e.g., check for excessive compression artifacts
                
                return True, (height, width)
                
        except Exception as e:
            logger.debug(f"Image validation failed for {image_path}: {e}")
            return False, None
    
    def extract_patches(
        self,
        image_path: Path,
        dataset_name: str
    ) -> List[Tuple[str, np.ndarray]]:
        """Extract patches from a single image"""
        patches = []
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to load {image_path}")
                return patches
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w = img.shape[:2]
            
            # Calculate number of patches
            n_h = max(1, (h - self.patch_size) // self.stride + 1)
            n_w = max(1, (w - self.patch_size) // self.stride + 1)
            
            # Extract patches
            for i in range(n_h):
                for j in range(n_w):
                    y = i * self.stride
                    x = j * self.stride
                    
                    # Ensure patch is within bounds
                    if y + self.patch_size <= h and x + self.patch_size <= w:
                        patch = img[y:y + self.patch_size, x:x + self.patch_size]
                        
                        # Create unique key
                        rel_path = image_path.relative_to(self.bronze_dir)
                        key = f"{dataset_name}/{rel_path}_{i:04d}_{j:04d}"
                        
                        patches.append((key, patch))
                        
                        # Optional: extract flipped/rotated versions for augmentation
                        if self.stride == self.patch_size:  # Non-overlapping only
                            # Horizontal flip
                            key_flip = f"{key}_flip"
                            patches.append((key_flip, np.fliplr(patch).copy()))
                            
                            # Vertical flip
                            key_flip_v = f"{key}_flipv"
                            patches.append((key_flip_v, np.flipud(patch).copy()))
            
        except Exception as e:
            logger.error(f"Error extracting patches from {image_path}: {e}")
        
        return patches
    
    def process_dataset(
        self,
        dataset_name: str,
        max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a single dataset from bronze to silver"""
        start_time = time.time()
        
        # Find all images in dataset
        dataset_path = self.bronze_dir / dataset_name
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"Scanning for images in: {dataset_path}")
        
        image_files = []
        for ext in self.image_extensions:
            # Use glob to find all images
            found = list(dataset_path.glob(f"*{ext}"))
            image_files.extend(found)
            logger.info(f"Found {len(found)} images with extension {ext}")
        
        # Also check for uppercase extensions
        for ext in self.image_extensions:
            found = list(dataset_path.glob(f"*{ext.upper()}"))
            image_files.extend(found)
        
        image_files = sorted(set(image_files))
        
        logger.info(f"Total images found: {len(image_files)}")
        
        if max_images:
            image_files = image_files[:max_images]
            logger.info(f"Limited to {max_images} images")
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {dataset_path}")
            return {
                'total_images': 0,
                'valid_images': 0,
                'invalid_images': 0,
                'total_patches': 0,
                'skipped_patches': 0,
                'processing_time': time.time() - start_time
            }
        
        logger.info(f"Processing dataset '{dataset_name}' with {len(image_files)} images")
        
        # Validate images
        valid_images = []
        for img_path in tqdm(image_files, desc="Validating images"):
            is_valid, dims = self.validate_image(img_path)
            if is_valid:
                valid_images.append(img_path)
                self.stats['valid_images'] += 1
            else:
                self.stats['invalid_images'] += 1
        
        self.stats['total_images'] = len(image_files)
        
        logger.info(f"Valid images: {len(valid_images)}/{len(image_files)}")
        
        if len(valid_images) == 0:
            logger.warning("No valid images found")
            return self.stats
        
        # Setup LMDB
        db_path = self.silver_dir / f"{dataset_name}.lmdb"
        db_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating LMDB at: {db_path}")
        
        env = lmdb.open(
            str(db_path),
            map_size=self.map_size,
            subdir=True,
            readonly=False,
            meminit=False,
            map_async=True
        )
        
        # Process images
        all_patches = []
        
        with tqdm(total=len(valid_images), desc="Extracting patches") as pbar:
            for img_path in valid_images:
                patches = self.extract_patches(img_path, dataset_name)
                all_patches.extend(patches)
                pbar.update(1)
                pbar.set_postfix(patches=len(patches))
        
        self.stats['total_patches'] = len(all_patches)
        
        logger.info(f"Extracted {len(all_patches)} patches")
        
        # Write to LMDB
        logger.info(f"Writing {len(all_patches)} patches to LMDB...")
        
        with env.begin(write=True) as txn:
            # Store metadata
            metadata = {
                'dataset': dataset_name,
                'num_patches': len(all_patches),
                'patch_size': self.patch_size,
                'stride': self.stride,
                'source_images': len(valid_images),
                'created': time.time()
            }
            txn.put(b'__metadata__', pickle.dumps(metadata))
            
            # Write patches in batches
            batch_size = 1000
            for i in tqdm(range(0, len(all_patches), batch_size), desc="Writing to LMDB"):
                batch = all_patches[i:i + batch_size]
                
                for key, patch in batch:
                    # Encode patch as PNG for compression
                    success, encoded = cv2.imencode('.png', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                    if success:
                        txn.put(key.encode('ascii'), encoded.tobytes())
                    else:
                        logger.warning(f"Failed to encode patch {key}")
        
        env.sync()
        env.close()
        
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(
            f"Dataset '{dataset_name}' processed: "
            f"{self.stats['valid_images']}/{self.stats['total_images']} valid, "
            f"{self.stats['total_patches']} patches in {self.stats['processing_time']:.2f}s"
        )
        
        return self.stats.copy()
    
    def process_all(self, max_datasets: Optional[int] = None) -> Dict[str, Any]:
        """Process all datasets in bronze layer"""
        # Find all dataset directories
        datasets = [d for d in self.bronze_dir.iterdir() if d.is_dir()]
        
        if max_datasets:
            datasets = datasets[:max_datasets]
        
        logger.info(f"Found {len(datasets)} datasets to process")
        
        all_stats = {}
        for dataset_path in datasets:
            stats = self.process_dataset(dataset_path.name)
            all_stats[dataset_path.name] = stats
        
        return all_stats
    
    def verify_lmdb(self, dataset_name: str) -> bool:
        """Verify LMDB database integrity"""
        db_path = self.silver_dir / f"{dataset_name}.lmdb"
        
        if not db_path.exists():
            logger.error(f"LMDB not found: {db_path}")
            return False
        
        try:
            env = lmdb.open(str(db_path), readonly=True, lock=False)
            
            with env.begin() as txn:
                # Check metadata
                metadata = pickle.loads(txn.get(b'__metadata__'))
                logger.info(f"Dataset metadata: {metadata}")
                
                # Count patches
                cursor = txn.cursor()
                cursor.set_key(b'__metadata__')
                
                patch_count = 0
                for key, value in cursor:
                    if key != b'__metadata__':
                        patch_count += 1
                        
                        # Try to decode first patch
                        if patch_count == 1:
                            img_array = np.frombuffer(value, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            logger.info(f"Sample patch shape: {img.shape}")
            
            env.close()
            
            logger.info(f"Verified {patch_count} patches in {dataset_name}.lmdb")
            return patch_count == metadata['num_patches']
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False


# In pipeline/medallion_svc.py, update the SilverDataset class:

class SilverDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Silver LMDB"""
    
    def __init__(
        self,
        silver_dir: str,
        dataset_name: str,
        transform: Optional[callable] = None
    ):
        """
        Args:
            silver_dir: Silver layer directory
            dataset_name: Name of dataset (e.g., 'nomosv2')
            transform: Optional torchvision transforms
        """
        self.silver_dir = Path(silver_dir)
        self.dataset_name = dataset_name
        self.transform = transform
        self.db_path = self.silver_dir / f"{dataset_name}.lmdb"
        
        # Don't open LMDB here - do it lazily
        self._env = None
        self._keys = None
        self.metadata = None
        
        # Load metadata without opening full environment
        self._load_metadata()
        
        logger.info(f"Loaded SilverDataset '{dataset_name}' with {len(self)} patches")
    
    def _load_metadata(self):
        """Load metadata without keeping environment open"""
        env = lmdb.open(
            str(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        
        with env.begin() as txn:
            self.metadata = pickle.loads(txn.get(b'__metadata__'))
            self.num_patches = self.metadata['num_patches']
        
        env.close()
    
    def _init_db(self):
        """Initialize LMDB environment (called lazily)"""
        if self._env is None:
            self._env = lmdb.open(
                str(self.db_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            
            # Load keys if not already loaded
            if self._keys is None:
                with self._env.begin() as txn:
                    self._keys = []
                    cursor = txn.cursor()
                    # Skip metadata key
                    if cursor.first():
                        while cursor.next():
                            key = cursor.key()
                            if key != b'__metadata__':
                                self._keys.append(key)
    
    def __len__(self) -> int:
        return self.num_patches
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single patch as tensor"""
        # Initialize DB if needed
        self._init_db()
        
        # Get key
        if self._keys is None:
            raise RuntimeError("Dataset not properly initialized")
        
        # Handle index wrapping
        idx = idx % len(self._keys)
        key = self._keys[idx]
        
        # Read from LMDB
        with self._env.begin() as txn:
            img_data = txn.get(key)
        
        if img_data is None:
            logger.warning(f"Could not read key {key}, returning random tensor")
            # Return a random tensor as fallback
            dummy = torch.rand(3, 256, 256)
            if self.transform:
                dummy = self.transform(dummy)
            return dummy
        
        # Decode image
        try:
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Error decoding image: {e}, returning random tensor")
            dummy = torch.rand(3, 256, 256)
            if self.transform:
                dummy = self.transform(dummy)
            return dummy
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor
    
    def close(self):
        """Close LMDB environment"""
        if self._env is not None:
            self._env.close()
            self._env = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Medallion Service: Bronze → Silver processing")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process dataset
    process_parser = subparsers.add_parser('process', help='Process a dataset')
    process_parser.add_argument('dataset', help='Dataset name in bronze/ directory')
    process_parser.add_argument('--patch-size', type=int, default=256, help='Patch size')
    process_parser.add_argument('--stride', type=int, help='Extraction stride (default: patch_size)')
    process_parser.add_argument('--min-size', type=int, default=1024, help='Minimum image size')
    process_parser.add_argument('--max-images', type=int, help='Maximum images to process')
    process_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    process_parser.add_argument('--bronze', default='data/bronze', help='Bronze directory')
    process_parser.add_argument('--silver', default='data/silver', help='Silver directory')
    
    # Process all datasets
    all_parser = subparsers.add_parser('process-all', help='Process all datasets')
    all_parser.add_argument('--patch-size', type=int, default=256, help='Patch size')
    all_parser.add_argument('--min-size', type=int, default=1024, help='Minimum image size')
    all_parser.add_argument('--max-datasets', type=int, help='Maximum datasets to process')
    all_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    all_parser.add_argument('--bronze', default='data/bronze', help='Bronze directory')
    all_parser.add_argument('--silver', default='data/silver', help='Silver directory')
    
    # Verify LMDB
    verify_parser = subparsers.add_parser('verify', help='Verify LMDB database')
    verify_parser.add_argument('dataset', help='Dataset name')
    verify_parser.add_argument('--silver', default='data/silver', help='Silver directory')
    
    # List datasets
    list_parser = subparsers.add_parser('list', help='List processed datasets')
    list_parser.add_argument('--silver', default='data/silver', help='Silver directory')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        service = MedallionService(
            bronze_dir=args.bronze,
            silver_dir=args.silver,
            patch_size=args.patch_size,
            stride=args.stride,
            min_size=args.min_size,
            num_workers=args.workers
        )
        stats = service.process_dataset(args.dataset, args.max_images)
        print(f"\nProcessing complete: {stats}")
        
    elif args.command == 'process-all':
        service = MedallionService(
            bronze_dir=args.bronze,
            silver_dir=args.silver,
            patch_size=args.patch_size,
            min_size=args.min_size,
            num_workers=args.workers
        )
        stats = service.process_all(args.max_datasets)
        print(f"\nAll datasets processed: {stats}")
        
    elif args.command == 'verify':
        service = MedallionService(silver_dir=args.silver)
        is_valid = service.verify_lmdb(args.dataset)
        print(f"\nLMDB verification {'PASSED' if is_valid else 'FAILED'}")
        
    elif args.command == 'list':
        silver_dir = Path(args.silver)
        if silver_dir.exists():
            lmdb_dbs = list(silver_dir.glob("*.lmdb"))
            print(f"\nProcessed datasets in {silver_dir}:")
            for db in lmdb_dbs:
                # Try to read metadata
                try:
                    env = lmdb.open(str(db), readonly=True, lock=False)
                    with env.begin() as txn:
                        metadata = pickle.loads(txn.get(b'__metadata__'))
                    env.close()
                    print(f"  {db.stem}: {metadata['num_patches']} patches from {metadata['source_images']} images")
                except:
                    print(f"  {db.stem}: (corrupted or empty)")
        else:
            print(f"Silver directory not found: {silver_dir}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()