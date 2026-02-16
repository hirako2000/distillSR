"""
Hugging Face dataset downloader for Bronze layer ingestion
Downloads high-resolution images from HF Hub datasets
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HFDataFetcher:
    """Fetch and download datasets from Hugging Face Hub"""

    KNOWN_DATASETS = {
        'nomosv2': 'philiphoffmann/nomosv2',
        'lsdir': 'philiphoffmann/LSDIR',
        'div2k': 'philiphoffmann/DIV2K',
        'flickr2k': 'philiphoffmann/Flickr2K',
        'bsd500': 'philiphoffmann/BSD500',
        'urban100': 'philiphoffmann/Urban100',
        'manga109': 'philiphoffmann/Manga109',
    }

    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    def __init__(
        self,
        bronze_dir: str = "data/bronze",
        max_workers: int = 4,
        timeout: int = 30
    ):
        """
        Args:
            bronze_dir: Directory to store downloaded images
            max_workers: Number of parallel download threads
            timeout: HTTP timeout in seconds
        """
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.timeout = timeout
        self.api = HfApi()

        self.stats = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'total_size_mb': 0
        }

    def list_available_datasets(self) -> dict:
        """List all known datasets and their HF paths"""
        return self.KNOWN_DATASETS.copy()

    def download_dataset(
        self,
        dataset_name: str,
        repo_id: Optional[str] = None,
        subset: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        max_images: Optional[int] = None,
        min_resolution: Tuple[int, int] = (512, 512)
    ) -> dict:
        """
        Download an entire dataset from HF Hub
        
        Args:
            dataset_name: Name to save dataset as (e.g., 'nomosv2')
            repo_id: HF repo ID (if None, tries KNOWN_DATASETS)
            subset: Subdirectory within repo to download
            extensions: List of file extensions to download
            max_images: Maximum number of images to download
            min_resolution: Minimum (height, width) resolution
            
        Returns:
            Statistics dictionary
        """
        if repo_id is None:
            if dataset_name in self.KNOWN_DATASETS:
                repo_id = self.KNOWN_DATASETS[dataset_name]
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset_name}'. "
                    f"Known: {list(self.KNOWN_DATASETS.keys())}"
                )

        if extensions is None:
            extensions = list(self.IMAGE_EXTENSIONS)

        dataset_dir = self.bronze_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        logger.info(f"Downloading {dataset_name} from {repo_id} to {dataset_dir}")

        try:
            files = list_repo_files(repo_id, repo_type="dataset")

            image_files = []
            for f in files:
                ext = Path(f).suffix.lower()
                if ext not in extensions:
                    continue

                if subset and not f.startswith(subset):
                    continue

                image_files.append(f)

            logger.info(f"Found {len(image_files)} images in repository")

            if max_images:
                image_files = image_files[:max_images]
                logger.info(f"Limited to {max_images} images")

            self.stats = self._download_files(
                repo_id,
                image_files,
                dataset_dir,
                min_resolution
            )

            logger.info(
                f"Download complete: {self.stats['downloaded']} downloaded, "
                f"{self.stats['skipped']} skipped, {self.stats['failed']} failed, "
                f"{self.stats['total_size_mb']:.2f} MB"
            )

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

        return self.stats

    def _download_files(
        self,
        repo_id: str,
        files: List[str],
        dataset_dir: Path,
        min_resolution: Tuple[int, int]
    ) -> dict:
        """Download multiple files in parallel"""
        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'total_size_mb': 0}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {}
            for f in files:
                local_path = dataset_dir / f

                if local_path.exists():
                    stats['skipped'] += 1
                    continue

                local_path.parent.mkdir(parents=True, exist_ok=True)

                future = executor.submit(
                    self._download_single_file,
                    repo_id,
                    f,
                    local_path,
                    min_resolution
                )
                future_to_file[future] = f

            with tqdm(total=len(future_to_file), desc="Downloading") as pbar:
                for future in as_completed(future_to_file):
                    f = future_to_file[future]
                    try:
                        success, size = future.result(timeout=self.timeout)
                        if success:
                            stats['downloaded'] += 1
                            stats['total_size_mb'] += size / (1024 * 1024)
                        else:
                            stats['failed'] += 1
                    except Exception as e:
                        logger.error(f"Failed to download {f}: {e}")
                        stats['failed'] += 1

                    pbar.update(1)

        return stats

    def _download_single_file(
        self,
        repo_id: str,
        file_path: str,
        local_path: Path,
        min_resolution: Tuple[int, int]
    ) -> Tuple[bool, int]:
        """Download and validate a single file"""
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=local_path.parent,
                local_dir_use_symlinks=False,
                resume=True
            )

            if not self._verify_image(downloaded_path, min_resolution):
                os.remove(downloaded_path)
                return False, 0

            size = os.path.getsize(downloaded_path)

            return True, size

        except Exception as e:
            logger.debug(f"Download failed for {file_path}: {e}")
            return False, 0

    def _verify_image(
        self,
        path: str,
        min_resolution: Tuple[int, int]
    ) -> bool:
        """Verify image meets resolution requirements"""
        try:
            with Image.open(path) as img:
                width, height = img.size

                if height < min_resolution[0] or width < min_resolution[1]:
                    logger.debug(f"Image {path} too small: {width}x{height}")
                    return False

                img.verify()

                with Image.open(path) as img:
                    img.load()

                return True

        except Exception as e:
            logger.debug(f"Image verification failed for {path}: {e}")
            return False

    def download_urls(
        self,
        urls: List[str],
        dataset_name: str = "custom",
        min_resolution: Tuple[int, int] = (512, 512)
    ) -> dict:
        """Download images from direct URLs"""
        dataset_dir = self.bronze_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'total_size_mb': 0}

        with tqdm(total=len(urls), desc="Downloading URLs") as pbar:
            for url in urls:
                try:
                    filename = url.split('/')[-1].split('?')[0]
                    if not any(filename.lower().endswith(ext) for ext in self.IMAGE_EXTENSIONS):
                        filename += '.jpg'

                    local_path = dataset_dir / filename

                    if local_path.exists():
                        stats['skipped'] += 1
                        pbar.update(1)
                        continue

                    response = requests.get(url, timeout=self.timeout, stream=True)
                    response.raise_for_status()

                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    if self._verify_image(str(local_path), min_resolution):
                        size = os.path.getsize(local_path)
                        stats['downloaded'] += 1
                        stats['total_size_mb'] += size / (1024 * 1024)
                    else:
                        os.remove(local_path)
                        stats['failed'] += 1

                except Exception as e:
                    logger.error(f"Failed to download {url}: {e}")
                    stats['failed'] += 1

                pbar.update(1)

        return stats

    def get_stats(self) -> dict:
        """Get current download statistics"""
        return self.stats.copy()


def create_synthetic_dataset(
    output_dir: str = "data/bronze/synthetic",
    num_images: int = 100,
    sizes: List[Tuple[int, int]] = [(1024, 1024), (2048, 2048), (4096, 4096)]
) -> None:
    """Create synthetic images for testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {num_images} synthetic images in {output_dir}")

    for i in tqdm(range(num_images), desc="Generating"):
        size = sizes[np.random.randint(len(sizes))]

        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

        cv2.rectangle(img, (100, 100), (size[0]-100, size[1]-100),
                     (255, 255, 255), 2)
        cv2.circle(img, (size[0]//2, size[1]//2), 200,
                  (128, 128, 128), -1)

        cv2.imwrite(str(output_path / f"synthetic_{i:04d}.png"), img)

    logger.info(f"Generated {num_images} synthetic images")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    subparsers.add_parser('list', help='List available datasets')

    download_parser = subparsers.add_parser('download', help='Download dataset')
    download_parser.add_argument('dataset', help='Dataset name')
    download_parser.add_argument('--repo', help='HF repo ID (optional)')
    download_parser.add_argument('--subset', help='Subset to download')
    download_parser.add_argument('--max-images', type=int, help='Maximum images')
    download_parser.add_argument('--min-size', type=int, default=512,
                                help='Minimum image size')
    download_parser.add_argument('--output', default='data/bronze',
                                help='Output directory')
    download_parser.add_argument('--workers', type=int, default=4,
                                help='Parallel workers')

    synthetic_parser = subparsers.add_parser('synthetic', help='Create synthetic dataset')
    synthetic_parser.add_argument('--num', type=int, default=100,
                                 help='Number of images')
    synthetic_parser.add_argument('--output', default='data/bronze/synthetic',
                                 help='Output directory')

    args = parser.parse_args()

    if args.command == 'list':
        fetcher = HFDataFetcher()
        datasets = fetcher.list_available_datasets()
        print("\nAvailable datasets:")
        for name, repo in datasets.items():
            print(f"  {name}: {repo}")

    elif args.command == 'download':
        fetcher = HFDataFetcher(
            bronze_dir=args.output,
            max_workers=args.workers
        )
        stats = fetcher.download_dataset(
            dataset_name=args.dataset,
            repo_id=args.repo,
            subset=args.subset,
            max_images=args.max_images,
            min_resolution=(args.min_size, args.min_size)
        )
        print(f"\nDownload complete: {stats}")

    elif args.command == 'synthetic':
        create_synthetic_dataset(
            output_dir=args.output,
            num_images=args.num
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
