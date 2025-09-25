#!/usr/bin/env python3
"""
PyTorch Lightning DataModule for Hand Prior Dataset

Loads the hand-focused dataset splits and provides PyTorch dataloaders
"""

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class HandPriorDataset(Dataset):
    """Dataset class for hand prior data"""

    def __init__(self, data_dir: str, samples_dict: Dict[str, List[int]],
                 transform=None):
        """
        Args:
            data_dir: Directory containing NPZ files
            samples_dict: Dictionary mapping dataset names to frame indices
            transform: Optional transform to apply to data
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Build index mapping
        self.samples = []  # List of (dataset_name, frame_idx) tuples
        self.loaded_datasets = {}  # Cache for loaded NPZ files

        print(f"Loading dataset indices...")
        for dataset_name, indices in samples_dict.items():
            for idx in indices:
                self.samples.append((dataset_name, idx))

        print(f"Total samples: {len(self.samples):,}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load and cache NPZ dataset"""
        if dataset_name not in self.loaded_datasets:
            npz_path = self.data_dir / f"{dataset_name}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {npz_path}")

            try:
                data = np.load(npz_path, allow_pickle=True)
                # Convert to dict for easier access - only load hand poses
                self.loaded_datasets[dataset_name] = {
                    'lhand_pose': data['lhand_pose'],
                    'rhand_pose': data['rhand_pose'],
                    'image_ids': data['image_ids'] if 'image_ids' in data else None
                }
            except Exception as e:
                raise RuntimeError(f"Error loading {npz_path}: {e}")

        return self.loaded_datasets[dataset_name]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        dataset_name, frame_idx = self.samples[idx]

        # Load dataset if not cached
        dataset = self._load_dataset(dataset_name)

        # Extract sample data - only hand poses
        sample = {
            'lhand_pose': torch.tensor(dataset['lhand_pose'][frame_idx], dtype=torch.float32),
            'rhand_pose': torch.tensor(dataset['rhand_pose'][frame_idx], dtype=torch.float32),
            'dataset_name': dataset_name,
            'frame_idx': frame_idx
        }

        # Add image_id if available
        if dataset['image_ids'] is not None:
            sample['image_id'] = str(dataset['image_ids'][frame_idx])

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample


class HandPriorDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Hand Prior Dataset"""

    def __init__(
        self,
        data_dir: str = "../data",
        splits_dir: str = "dataset_splits",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform=None
    ):
        """
        Args:
            data_dir: Directory containing NPZ files
            splits_dir: Directory containing dataset split files
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for dataloaders
            pin_memory: Whether to pin memory in dataloaders
            transform: Optional transform to apply to data
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.splits_dir = Path(splits_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform

        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.metadata = None

    def prepare_data(self):
        """Download/prepare data if needed (not implemented)"""
        # Check that data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Check that splits directory exists
        if not self.splits_dir.exists():
            raise FileNotFoundError(
                f"Splits directory not found: {self.splits_dir}\\n"
                f"Please run create_datamodule.py first to generate dataset splits."
            )

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""

        # Load metadata
        metadata_path = self.splits_dir / "dataset_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"📊 Dataset Info:")
        # Handle both test and regular dataset metadata formats
        if 'dataset_creation_info' in self.metadata:
            creation_info = self.metadata['dataset_creation_info']
        elif 'test_dataset_info' in self.metadata:
            creation_info = self.metadata['test_dataset_info']
        else:
            raise KeyError("No dataset creation info found in metadata")

        print(f"  Total samples: {creation_info['total_samples']:,}")
        for split_name, count in creation_info['splits'].items():
            print(f"  {split_name}: {count:,}")

        # Setup train dataset
        if stage == "fit" or stage is None:
            train_path = self.splits_dir / "train_samples.json"
            with open(train_path, 'r') as f:
                train_samples = json.load(f)

            self.train_dataset = HandPriorDataset(
                self.data_dir, train_samples, self.transform
            )
            print(f"✅ Train dataset: {len(self.train_dataset):,} samples")

        # Setup validation dataset
        if stage == "fit" or stage == "validate" or stage is None:
            val_path = self.splits_dir / "val_samples.json"
            with open(val_path, 'r') as f:
                val_samples = json.load(f)

            self.val_dataset = HandPriorDataset(
                self.data_dir, val_samples, self.transform
            )
            print(f"✅ Val dataset: {len(self.val_dataset):,} samples")

        # Setup test dataset
        if stage == "test" or stage is None:
            test_path = self.splits_dir / "test_samples.json"
            with open(test_path, 'r') as f:
                test_samples = json.load(f)

            self.test_dataset = HandPriorDataset(
                self.data_dir, test_samples, self.transform
            )
            print(f"✅ Test dataset: {len(self.test_dataset):,} samples")

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def get_dataset_info(self) -> Dict:
        """Get dataset metadata"""
        if self.metadata is None:
            raise RuntimeError("DataModule not setup yet. Call setup() first.")
        return self.metadata


# Example transforms
class HandPriorTransform:
    """Example transform for hand prior data"""

    def __init__(self, normalize_poses: bool = True, add_noise: bool = False,
                 noise_std: float = 0.01):
        self.normalize_poses = normalize_poses
        self.add_noise = add_noise
        self.noise_std = noise_std

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transforms to sample"""

        if self.normalize_poses:
            # Normalize hand pose parameters to [-1, 1] range
            for pose_key in ['lhand_pose', 'rhand_pose']:
                pose = sample[pose_key]
                # Simple normalization - you might want to use dataset statistics
                sample[pose_key] = torch.tanh(pose)

        if self.add_noise:
            # Add small amount of noise for regularization
            for pose_key in ['lhand_pose', 'rhand_pose']:
                noise = torch.randn_like(sample[pose_key]) * self.noise_std
                sample[pose_key] = sample[pose_key] + noise

        return sample


if __name__ == "__main__":
    # Example usage
    print("🤖 Testing Hand Prior DataModule")

    # Create datamodule
    transform = HandPriorTransform(normalize_poses=True, add_noise=False)
    dm = HandPriorDataModule(
        data_dir="../data",
        splits_dir="test_dataset_splits",  # Use test splits
        batch_size=16,
        num_workers=0,  # Set to 0 for testing
        transform=transform
    )

    # Setup
    dm.prepare_data()
    dm.setup()

    # Test train dataloader
    print("\\n🔄 Testing train dataloader...")
    train_loader = dm.train_dataloader()

    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  lhand_pose shape: {batch['lhand_pose'].shape}")
        print(f"  rhand_pose shape: {batch['rhand_pose'].shape}")
        print(f"  datasets in batch: {set(batch['dataset_name'])}")

        if i >= 2:  # Test first few batches
            break

    print("\\n✅ DataModule test complete!")