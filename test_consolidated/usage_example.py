#!/usr/bin/env python3
"""
Example usage of consolidated NPZ files with VAE DataModule
"""

# Option 1: Use the production VAE datamodule with consolidated files
from hand_vae_datamodule import HandVAEDataModule

# Create a simple datamodule that loads single NPZ files
class ConsolidatedHandVAEDataModule(HandVAEDataModule):
    def __init__(self, consolidated_dir="consolidated_splits", **kwargs):
        # Override parent init to use consolidated files
        super().__init__(**kwargs)
        self.consolidated_dir = Path(consolidated_dir)

    def setup(self, stage=None):
        from hand_vae_datamodule import HandPoseNPZDataset

        if stage == "fit" or stage is None:
            train_npz = self.consolidated_dir / "train.npz"
            self.train_dataset = HandPoseNPZDataset(
                train_npz, stats=None, standardize=True,
                return_dict=self.return_dict, mmap=True
            )
            if hasattr(self.train_dataset, 'mean'):
                self._stats = {"mean": self.train_dataset.mean, "std": self.train_dataset.std}

        if stage == "fit" or stage == "validate" or stage is None:
            val_npz = self.consolidated_dir / "val.npz"
            self.val_dataset = HandPoseNPZDataset(
                val_npz, stats=self._stats, standardize=True,
                return_dict=self.return_dict, mmap=True
            )

# Usage:
# dm = ConsolidatedHandVAEDataModule(batch_size=8192, return_dict=False)
# dm.setup()
# train_loader = dm.train_dataloader()

# Option 2: Direct NPZ usage
import numpy as np
import torch

train_data = np.load("consolidated_splits/train.npz")
lhand = train_data["lhand_pose"]  # (N, 45)
rhand = train_data["rhand_pose"]  # (N, 45)

# Concatenate to 90D for VAE
x90 = np.concatenate([lhand, rhand], axis=1)  # (N, 90)

# Standardize
mean = x90.mean(axis=0)
std = x90.std(axis=0)
x90_std = (x90 - mean) / np.clip(std, 1e-6, None)

# Convert to tensor
x90_tensor = torch.from_numpy(x90_std.astype(np.float32))
print(f"Ready for VAE: {x90_tensor.shape}")
