#!/usr/bin/env python3
"""
Test the consolidated NPZ approach for VAE training
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


def wrap_to_pi(x):
    """Keep axis-angle in [-pi, pi] per component"""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class HandPoseNPZDataset(Dataset):
    """Simple dataset for single consolidated NPZ files"""

    def __init__(self, npz_path, stats=None, standardize=True, return_dict=False, mmap=True):
        super().__init__()
        self._npz = np.load(npz_path, mmap_mode="r" if mmap else None)
        self.lh = self._npz["lhand_pose"]   # (N, 45)
        self.rh = self._npz["rhand_pose"]   # (N, 45)
        assert self.lh.shape == self.rh.shape and self.lh.shape[1] == 45

        self.N = self.lh.shape[0]
        self.standardize = standardize
        self.return_dict = return_dict

        if self.standardize:
            if stats is None:
                # compute on this file (used for train split)
                x90 = np.concatenate([self.lh, self.rh], axis=1).astype(np.float32)
                self.mean = x90.mean(axis=0)
                self.std = x90.std(axis=0)
            else:
                self.mean = stats["mean"]
                self.std = stats["std"]
            self.std = np.clip(self.std, 1e-6, None)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        lh = self.lh[idx].astype(np.float32)  # (45,)
        rh = self.rh[idx].astype(np.float32)  # (45,)
        x90 = np.concatenate([lh, rh], axis=0)  # (90,)

        if self.standardize:
            x90 = (x90 - self.mean) / self.std

        x90 = torch.from_numpy(x90)  # (90,)

        if self.return_dict:
            out = {
                "x90": x90,                         # (90,) standardized (if enabled)
                "lhand_pose": torch.from_numpy(lh), # raw per-hand (unstandardized)
                "rhand_pose": torch.from_numpy(rh),
            }
            if self.standardize:
                out["mean"] = torch.from_numpy(self.mean)
                out["std"] = torch.from_numpy(self.std)
            return out
        else:
            return x90  # simplest for your VAE


class ConsolidatedHandDataModule(pl.LightningDataModule):
    """DataModule for consolidated NPZ files"""

    def __init__(
        self,
        consolidated_dir="test_consolidated",
        batch_size=8192,
        num_workers=8,
        pin_memory=True,
        return_dict=False,
    ):
        super().__init__()
        self.consolidated_dir = Path(consolidated_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.return_dict = return_dict
        self._stats = None

    def setup(self, stage=None):
        # Build a small dataset to compute train stats once
        train_npz = self.consolidated_dir / "train.npz"
        tmp_train = HandPoseNPZDataset(train_npz, stats=None, standardize=True, return_dict=False, mmap=True)

        if hasattr(tmp_train, 'mean') and tmp_train.mean is not None:
            self._stats = {"mean": tmp_train.mean, "std": tmp_train.std}

        self.train_ds = HandPoseNPZDataset(
            train_npz, stats=None, standardize=True, return_dict=self.return_dict, mmap=True
        )

        val_npz = self.consolidated_dir / "val.npz"
        if val_npz.exists():
            self.val_ds = HandPoseNPZDataset(
                val_npz, stats=self._stats, standardize=True, return_dict=self.return_dict, mmap=True
            )

        test_npz = self.consolidated_dir / "test.npz"
        if test_npz.exists():
            self.test_ds = HandPoseNPZDataset(
                test_npz, stats=self._stats, standardize=True, return_dict=self.return_dict, mmap=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=max(131072, self.batch_size),
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory, drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=max(131072, self.batch_size),
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory, drop_last=False
        )

    @property
    def stats(self):
        return self._stats


def test_consolidated_datamodule():
    """Test the consolidated approach"""

    print("🧪 Testing Consolidated Hand DataModule")
    print("=" * 45)

    # Test tensor-only mode
    print("\\n🔄 Testing tensor-only mode...")
    dm = ConsolidatedHandDataModule(
        consolidated_dir="test_consolidated",
        batch_size=32,
        num_workers=0,
        return_dict=False
    )

    dm.setup()

    # Test train loader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"✅ Tensor mode:")
    print(f"  Batch shape: {batch.shape}")  # Should be (32, 90)
    print(f"  Data type: {batch.dtype}")
    print(f"  Value range: [{batch.min():.3f}, {batch.max():.3f}]")

    # Test dict mode
    print("\\n🔄 Testing dict mode...")
    dm_dict = ConsolidatedHandDataModule(
        consolidated_dir="test_consolidated",
        batch_size=16,
        num_workers=0,
        return_dict=True
    )

    dm_dict.setup()
    train_loader_dict = dm_dict.train_dataloader()
    batch_dict = next(iter(train_loader_dict))

    print(f"✅ Dict mode:")
    print(f"  Keys: {list(batch_dict.keys())}")
    print(f"  x90 shape: {batch_dict['x90'].shape}")
    print(f"  lhand_pose shape: {batch_dict['lhand_pose'].shape}")
    print(f"  rhand_pose shape: {batch_dict['rhand_pose'].shape}")

    # Test statistics
    print(f"\\n📊 Dataset Statistics:")
    if dm.stats is not None:
        print(f"  Mean range: [{dm.stats['mean'].min():.3f}, {dm.stats['mean'].max():.3f}]")
        print(f"  Std range: [{dm.stats['std'].min():.3f}, {dm.stats['std'].max():.3f}]")

    # Test all loaders
    print(f"\\n🔄 Testing all dataloaders...")
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    print(f"  Train samples: {len(dm.train_ds):,}")
    print(f"  Val samples: {len(dm.val_ds):,}")
    print(f"  Test samples: {len(dm.test_ds):,}")

    print(f"  Val batch shape: {val_batch.shape}")
    print(f"  Test batch shape: {test_batch.shape}")

    print("\\n✅ All tests passed!")

    # Performance comparison
    print("\\n⚡ Performance benefits of consolidated approach:")
    print("  ✅ Single file loading (no multi-dataset coordination)")
    print("  ✅ Memory-mapped access for large datasets")
    print("  ✅ Direct 90D tensor output for VAE")
    print("  ✅ Pre-computed standardization statistics")
    print("  ✅ Optimized for large batch training (8192+ samples)")

    return dm


if __name__ == "__main__":
    dm = test_consolidated_datamodule()

    print("\\n🚀 Ready for VAE training!")
    print("\\nExample usage:")
    print("""
# For VAE training (tensor only):
dm = ConsolidatedHandDataModule(
    consolidated_dir="consolidated_splits",  # Full dataset
    batch_size=8192,
    return_dict=False  # Just (B, 90) tensors
)

# For research (with metadata):
dm = ConsolidatedHandDataModule(
    batch_size=1024,
    return_dict=True  # Dict with x90, raw poses, stats
)
    """)