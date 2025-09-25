#!/usr/bin/env python3
"""
Production-ready Hand VAE DataModule

Optimized for VAE training with:
- Memory-mapped NPZ loading for efficiency
- Concatenated 90D hand poses [lhand + rhand]
- Standardization with train set statistics
- Flexible return format (tensor or dict)
- Multi-dataset support from JSON splits
"""

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import warnings


def wrap_to_pi(x):
    """Keep axis-angle in [-pi, pi] per component (optional but often helpful)"""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class HandPoseDataset(Dataset):
    """Dataset for hand poses with memory mapping and standardization"""

    def __init__(
        self,
        data_dir: str,
        samples_dict: Dict[str, List[int]],
        stats: Optional[Dict[str, np.ndarray]] = None,
        standardize: bool = True,
        return_dict: bool = False,
        mmap: bool = True,
        wrap_angles: bool = False
    ):
        """
        Args:
            data_dir: Directory containing NPZ files
            samples_dict: Dict mapping dataset names to frame indices
            stats: Pre-computed mean/std for standardization
            standardize: Whether to standardize the data
            return_dict: If True, return dict; if False, return tensor
            mmap: Use memory mapping for NPZ files
            wrap_angles: Wrap angles to [-pi, pi]
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.standardize = standardize
        self.return_dict = return_dict
        self.mmap = mmap
        self.wrap_angles = wrap_angles

        # Build sample index
        self.samples = []  # List of (dataset_name, frame_idx) tuples
        self.loaded_datasets = {}  # Cache for loaded NPZ files

        print(f"Building dataset index...")
        for dataset_name, indices in samples_dict.items():
            for idx in indices:
                self.samples.append((dataset_name, idx))

        print(f"Total samples: {len(self.samples):,}")

        # Handle standardization
        if self.standardize:
            if stats is None:
                # Compute stats on this dataset (used for train split)
                print("Computing dataset statistics...")
                self.mean, self.std = self._compute_stats()
            else:
                self.mean = stats["mean"]
                self.std = stats["std"]

            # Ensure std is not too small
            self.std = np.clip(self.std, 1e-6, None)
        else:
            self.mean = None
            self.std = None

    def _load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load and cache NPZ dataset with memory mapping"""
        if dataset_name not in self.loaded_datasets:
            npz_path = self.data_dir / f"{dataset_name}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {npz_path}")

            try:
                # Use memory mapping for efficiency
                mmap_mode = "r" if self.mmap else None
                data = np.load(npz_path, mmap_mode=mmap_mode, allow_pickle=True)

                self.loaded_datasets[dataset_name] = {
                    'lhand_pose': data['lhand_pose'],
                    'rhand_pose': data['rhand_pose'],
                }

                # Optional angle wrapping
                if self.wrap_angles:
                    self.loaded_datasets[dataset_name]['lhand_pose'] = wrap_to_pi(
                        self.loaded_datasets[dataset_name]['lhand_pose']
                    )
                    self.loaded_datasets[dataset_name]['rhand_pose'] = wrap_to_pi(
                        self.loaded_datasets[dataset_name]['rhand_pose']
                    )

            except Exception as e:
                raise RuntimeError(f"Error loading {npz_path}: {e}")

        return self.loaded_datasets[dataset_name]

    def _compute_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std for standardization using streaming approach"""

        # Use streaming computation for large datasets
        n_total = 0
        sum_x = np.zeros(90, dtype=np.float64)
        sum_x2 = np.zeros(90, dtype=np.float64)

        # Process in chunks to manage memory
        chunk_size = min(10000, len(self.samples))

        for i in tqdm(range(0, len(self.samples), chunk_size), desc="Computing stats"):
            chunk_samples = self.samples[i:i + chunk_size]
            chunk_data = []

            for dataset_name, frame_idx in chunk_samples:
                dataset = self._load_dataset(dataset_name)
                lh = dataset['lhand_pose'][frame_idx].astype(np.float32)
                rh = dataset['rhand_pose'][frame_idx].astype(np.float32)
                x90 = np.concatenate([lh, rh], axis=0)  # (90,)
                chunk_data.append(x90)

            if chunk_data:
                chunk_array = np.stack(chunk_data, axis=0).astype(np.float64)  # (chunk_size, 90)
                chunk_n = chunk_array.shape[0]

                sum_x += chunk_array.sum(axis=0)
                sum_x2 += (chunk_array ** 2).sum(axis=0)
                n_total += chunk_n

        # Compute final statistics
        mean = (sum_x / n_total).astype(np.float32)
        var = (sum_x2 / n_total) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)

        print(f"Computed stats: mean range [{mean.min():.3f}, {mean.max():.3f}], "
              f"std range [{std.min():.3f}, {std.max():.3f}]")

        return mean, std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Dict]:
        """Get a single sample"""
        dataset_name, frame_idx = self.samples[idx]

        # Load dataset if not cached
        dataset = self._load_dataset(dataset_name)

        # Extract hand poses
        lh = dataset['lhand_pose'][frame_idx].astype(np.float32)  # (45,)
        rh = dataset['rhand_pose'][frame_idx].astype(np.float32)  # (45,)

        # Concatenate to 90D
        x90 = np.concatenate([lh, rh], axis=0)  # (90,)

        # Standardize if enabled
        if self.standardize:
            x90 = (x90 - self.mean) / self.std

        # Convert to tensor
        x90_tensor = torch.from_numpy(x90)  # (90,)

        if self.return_dict:
            output = {
                "x90": x90_tensor,                              # (90,) standardized
                "lhand_pose": torch.from_numpy(lh),            # (45,) raw left hand
                "rhand_pose": torch.from_numpy(rh),            # (45,) raw right hand
                "dataset_name": dataset_name,
                "frame_idx": frame_idx,
            }

            # Add standardization stats if available
            if self.standardize:
                output["mean"] = torch.from_numpy(self.mean)
                output["std"] = torch.from_numpy(self.std)

            return output
        else:
            # Simple tensor return for VAE training
            return x90_tensor  # (90,)


class HandVAEDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule optimized for Hand VAE training"""

    def __init__(
        self,
        data_dir: str = "../data",
        splits_dir: str = "dataset_splits",
        batch_size: int = 8192,      # Large batches for VAE training
        val_batch_size: int = 131072, # Even larger for validation
        num_workers: int = 8,
        pin_memory: bool = True,
        return_dict: bool = False,   # False = tensor only, True = dict with extras
        mmap: bool = True,          # Use memory mapping
        wrap_angles: bool = False,   # Wrap angles to [-pi, pi]
        standardize: bool = True,    # Standardize with train stats
    ):
        """
        Args:
            data_dir: Directory containing NPZ files
            splits_dir: Directory containing dataset split files
            batch_size: Training batch size
            val_batch_size: Validation batch size (can be larger)
            num_workers: Number of worker processes
            pin_memory: Pin memory for GPU transfer
            return_dict: Return dict (True) or tensor only (False)
            mmap: Use memory mapping for NPZ files
            wrap_angles: Wrap angles to [-pi, pi] range
            standardize: Standardize data with train set statistics
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.splits_dir = Path(splits_dir)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.return_dict = return_dict
        self.mmap = mmap
        self.wrap_angles = wrap_angles
        self.standardize = standardize

        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._stats = None

    def prepare_data(self):
        """Check that required files exist"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if not self.splits_dir.exists():
            raise FileNotFoundError(
                f"Splits directory not found: {self.splits_dir}\\n"
                f"Please run create_datamodule.py first to generate dataset splits."
            )

    def setup(self, stage: Optional[str] = None):
        """Setup datasets and compute statistics"""

        # Load train split first to compute stats
        if stage == "fit" or stage is None:
            train_path = self.splits_dir / "train_samples.json"
            with open(train_path, 'r') as f:
                train_samples = json.load(f)

            print("🔧 Setting up training dataset...")
            self.train_dataset = HandPoseDataset(
                data_dir=self.data_dir,
                samples_dict=train_samples,
                stats=None,  # Compute stats on train set
                standardize=self.standardize,
                return_dict=self.return_dict,
                mmap=self.mmap,
                wrap_angles=self.wrap_angles
            )

            # Save computed stats
            if self.standardize and self.train_dataset.mean is not None:
                self._stats = {
                    "mean": self.train_dataset.mean,
                    "std": self.train_dataset.std
                }

        # Setup validation dataset
        if stage == "fit" or stage == "validate" or stage is None:
            val_path = self.splits_dir / "val_samples.json"
            with open(val_path, 'r') as f:
                val_samples = json.load(f)

            print("🔧 Setting up validation dataset...")
            self.val_dataset = HandPoseDataset(
                data_dir=self.data_dir,
                samples_dict=val_samples,
                stats=self._stats,  # Use train stats
                standardize=self.standardize,
                return_dict=self.return_dict,
                mmap=self.mmap,
                wrap_angles=self.wrap_angles
            )

        # Setup test dataset
        if stage == "test" or stage is None:
            test_path = self.splits_dir / "test_samples.json"
            with open(test_path, 'r') as f:
                test_samples = json.load(f)

            print("🔧 Setting up test dataset...")
            self.test_dataset = HandPoseDataset(
                data_dir=self.data_dir,
                samples_dict=test_samples,
                stats=self._stats,  # Use train stats
                standardize=self.standardize,
                return_dict=self.return_dict,
                mmap=self.mmap,
                wrap_angles=self.wrap_angles
            )

        # Print dataset info
        if self.train_dataset is not None:
            print(f"✅ Train dataset: {len(self.train_dataset):,} samples")
        if self.val_dataset is not None:
            print(f"✅ Val dataset: {len(self.val_dataset):,} samples")
        if self.test_dataset is not None:
            print(f"✅ Test dataset: {len(self.test_dataset):,} samples")

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # Keep all data for VAE training
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )

    @property
    def stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Get dataset statistics for saving in checkpoints"""
        return self._stats

    def get_sample_batch(self, split: str = "train", n_samples: int = 8) -> Union[torch.Tensor, Dict]:
        """Get a small sample batch for debugging/visualization"""
        if split == "train" and self.train_dataset is not None:
            dataset = self.train_dataset
        elif split == "val" and self.val_dataset is not None:
            dataset = self.val_dataset
        elif split == "test" and self.test_dataset is not None:
            dataset = self.test_dataset
        else:
            raise ValueError(f"Dataset {split} not available")

        indices = np.random.choice(len(dataset), n_samples, replace=False)
        samples = [dataset[i] for i in indices]

        if self.return_dict:
            # Stack dict entries
            batch = {}
            for key in samples[0].keys():
                if key in ["dataset_name"]:
                    batch[key] = [s[key] for s in samples]
                elif key in ["frame_idx"]:
                    batch[key] = torch.tensor([s[key] for s in samples])
                else:
                    batch[key] = torch.stack([s[key] for s in samples])
            return batch
        else:
            # Stack tensors
            return torch.stack(samples)


if __name__ == "__main__":
    # Test the VAE DataModule
    print("🧪 Testing Hand VAE DataModule")
    print("=" * 50)

    # Create datamodule
    dm = HandVAEDataModule(
        data_dir="../data",
        splits_dir="test_dataset_splits",
        batch_size=32,
        val_batch_size=64,
        num_workers=0,  # Single process for testing
        return_dict=False,  # Test tensor-only mode first
        standardize=True,
        mmap=True
    )

    # Setup
    dm.prepare_data()
    dm.setup()

    print(f"\\n📊 Dataset Statistics:")
    if dm.stats is not None:
        print(f"  Mean range: [{dm.stats['mean'].min():.3f}, {dm.stats['mean'].max():.3f}]")
        print(f"  Std range: [{dm.stats['std'].min():.3f}, {dm.stats['std'].max():.3f}]")

    # Test train dataloader
    print("\\n🔄 Testing train dataloader (tensor mode)...")
    train_loader = dm.train_dataloader()

    batch = next(iter(train_loader))
    print(f"✅ Batch shape: {batch.shape}")  # Should be (batch_size, 90)
    print(f"  Data type: {batch.dtype}")
    print(f"  Value range: [{batch.min():.3f}, {batch.max():.3f}]")

    # Test dict mode
    print("\\n🔄 Testing dict mode...")
    dm.return_dict = True
    dm.setup()  # Re-setup with dict mode

    sample_batch = dm.get_sample_batch("train", n_samples=4)
    print(f"✅ Dict batch keys: {list(sample_batch.keys())}")
    print(f"  x90 shape: {sample_batch['x90'].shape}")
    print(f"  lhand_pose shape: {sample_batch['lhand_pose'].shape}")
    print(f"  rhand_pose shape: {sample_batch['rhand_pose'].shape}")
    print(f"  Datasets: {set(sample_batch['dataset_name'])}")

    print("\\n✅ All VAE DataModule tests passed!")
    print("\\n💡 Usage for VAE training:")
    print("  - return_dict=False: Get (batch_size, 90) tensors directly")
    print("  - return_dict=True: Get dict with x90, raw poses, and metadata")
    print("  - Large batch sizes (8192+) recommended for VAE training")
    print("  - Statistics automatically computed from train set")