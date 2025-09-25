#!/usr/bin/env python3
"""
Comprehensive SMPL-X Pose DataModule

Loads all SMPL-X pose parameters:
- Global orientation (3 params)
- Body pose (63 params)
- Jaw pose (3 params)
- Left hand pose (45 params)
- Right hand pose (45 params)

Returns axis-angle format (B, 159) -> reshaped to (B, 53, 3)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import warnings


class ComprehensivePoseDataset(Dataset):
    """Dataset for comprehensive SMPL-X pose data"""

    def __init__(
        self,
        npz_files: list,
        split: str = 'train',
        return_dict: bool = False,
        standardize: bool = True,
        stats: Optional[Dict] = None
    ):
        self.npz_files = npz_files
        self.split = split
        self.return_dict = return_dict
        self.standardize = standardize
        self.stats = stats

        # Load all data
        self.data = self._load_all_data()
        print(f"✅ Loaded {len(self.data)} samples for {split}")

    def _load_all_data(self) -> np.ndarray:
        """Load and concatenate all pose data"""
        all_poses = []

        for npz_file in self.npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)

                # Extract pose components - all should be axis-angle format
                poses_per_file = []

                # Global orientation (3,)
                if 'global_orient' in data:
                    global_orient = data['global_orient']  # Shape: (N, 3)
                    poses_per_file.append(global_orient)
                else:
                    # Fallback to zeros if not present
                    n_samples = len(data[list(data.keys())[0]])
                    poses_per_file.append(np.zeros((n_samples, 3)))

                # Body pose (21 joints * 3 = 63)
                if 'body_pose' in data:
                    body_pose = data['body_pose']  # Shape: (N, 63)
                    poses_per_file.append(body_pose)
                else:
                    n_samples = len(data[list(data.keys())[0]])
                    poses_per_file.append(np.zeros((n_samples, 63)))

                # Jaw pose (3,)
                if 'jaw_pose' in data:
                    jaw_pose = data['jaw_pose']  # Shape: (N, 3)
                    poses_per_file.append(jaw_pose)
                else:
                    n_samples = len(data[list(data.keys())[0]])
                    poses_per_file.append(np.zeros((n_samples, 3)))

                # Left hand pose (15 joints * 3 = 45)
                if 'lhand_pose' in data:
                    lhand_pose = data['lhand_pose']  # Shape: (N, 45)
                    poses_per_file.append(lhand_pose)
                else:
                    n_samples = len(data[list(data.keys())[0]])
                    poses_per_file.append(np.zeros((n_samples, 45)))

                # Right hand pose (15 joints * 3 = 45)
                if 'rhand_pose' in data:
                    rhand_pose = data['rhand_pose']  # Shape: (N, 45)
                    poses_per_file.append(rhand_pose)
                else:
                    n_samples = len(data[list(data.keys())[0]])
                    poses_per_file.append(np.zeros((n_samples, 45)))

                # Concatenate all pose components: 3 + 63 + 3 + 45 + 45 = 159
                file_poses = np.concatenate(poses_per_file, axis=1)
                all_poses.append(file_poses)

                print(f"  📁 {Path(npz_file).name}: {file_poses.shape} -> Total 159D poses")

            except Exception as e:
                print(f"❌ Error loading {npz_file}: {e}")
                continue

        if not all_poses:
            raise ValueError("No valid pose data found!")

        # Concatenate all files
        combined_poses = np.concatenate(all_poses, axis=0)
        print(f"📊 Combined shape: {combined_poses.shape} (should be N x 159)")

        return combined_poses

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        pose_159d = self.data[idx]  # Shape: (159,)

        # Apply standardization if available
        if self.standardize and self.stats is not None:
            pose_159d = (pose_159d - self.stats['mean']) / (self.stats['std'] + 1e-8)

        # Convert to tensor
        pose_tensor = torch.from_numpy(pose_159d).float()

        if self.return_dict:
            # Reshape to (53, 3) and split into components
            pose_53x3 = pose_tensor.view(53, 3)

            return {
                'global_orient': pose_53x3[0:1],      # (1, 3) - first joint
                'body_pose': pose_53x3[1:22],         # (21, 3) - body joints
                'jaw_pose': pose_53x3[22:23],         # (1, 3) - jaw joint
                'lhand_pose': pose_53x3[23:38],       # (15, 3) - left hand joints
                'rhand_pose': pose_53x3[38:53],       # (15, 3) - right hand joints
                'full_pose': pose_53x3                # (53, 3) - everything
            }
        else:
            # Return as (53, 3) tensor
            return pose_tensor.view(53, 3)


class ComprehensivePoseDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for comprehensive SMPL-X pose data"""

    def __init__(
        self,
        data_dir: str = "../data",
        splits_dir: str = "dataset_splits",
        batch_size: int = 64,
        num_workers: int = 4,
        return_dict: bool = False,
        standardize: bool = True,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.splits_dir = Path(splits_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_dict = return_dict
        self.standardize = standardize
        self.pin_memory = pin_memory

        # Will store computed statistics
        self.stats = None

    def prepare_data(self):
        """Ensure data splits exist"""
        if not self.splits_dir.exists():
            raise FileNotFoundError(f"Splits directory not found: {self.splits_dir}")

        required_files = ['train_files.txt', 'val_files.txt', 'test_files.txt']
        for file in required_files:
            if not (self.splits_dir / file).exists():
                raise FileNotFoundError(f"Required split file not found: {self.splits_dir / file}")

    def _load_split_files(self, split: str) -> list:
        """Load NPZ file paths for a given split"""
        split_file = self.splits_dir / f"{split}_files.txt"

        with open(split_file, 'r') as f:
            file_paths = [line.strip() for line in f if line.strip()]

        # Convert to absolute paths
        npz_files = []
        for file_path in file_paths:
            if Path(file_path).is_absolute():
                npz_files.append(file_path)
            else:
                npz_files.append(str(self.data_dir / file_path))

        print(f"📁 {split.upper()} split: {len(npz_files)} files")
        return npz_files

    def _compute_statistics(self) -> Dict[str, np.ndarray]:
        """Compute mean and std from training data"""
        print("📊 Computing dataset statistics...")

        train_files = self._load_split_files('train')

        # Load a subset for stats (memory efficient)
        all_data = []
        max_samples_per_file = 1000  # Limit samples per file for stats

        for npz_file in train_files[:10]:  # Use first 10 files for stats
            try:
                data = np.load(npz_file, allow_pickle=True)

                # Extract and concatenate pose components
                poses = []
                n_samples = min(len(data[list(data.keys())[0]]), max_samples_per_file)

                # Global orientation (3,)
                global_orient = data.get('global_orient', np.zeros((n_samples, 3)))[:n_samples]
                poses.append(global_orient)

                # Body pose (63,)
                body_pose = data.get('body_pose', np.zeros((n_samples, 63)))[:n_samples]
                poses.append(body_pose)

                # Jaw pose (3,)
                jaw_pose = data.get('jaw_pose', np.zeros((n_samples, 3)))[:n_samples]
                poses.append(jaw_pose)

                # Left hand pose (45,)
                lhand_pose = data.get('lhand_pose', np.zeros((n_samples, 45)))[:n_samples]
                poses.append(lhand_pose)

                # Right hand pose (45,)
                rhand_pose = data.get('rhand_pose', np.zeros((n_samples, 45)))[:n_samples]
                poses.append(rhand_pose)

                # Combine: 3 + 63 + 3 + 45 + 45 = 159
                file_data = np.concatenate(poses, axis=1)
                all_data.append(file_data)

            except Exception as e:
                print(f"⚠️  Skipping {npz_file} for stats: {e}")
                continue

        if not all_data:
            raise ValueError("Could not load any data for statistics computation!")

        # Compute statistics
        combined_data = np.concatenate(all_data, axis=0)
        print(f"📈 Statistics computed from {combined_data.shape} samples")

        stats = {
            'mean': combined_data.mean(axis=0),
            'std': combined_data.std(axis=0),
            'min': combined_data.min(axis=0),
            'max': combined_data.max(axis=0)
        }

        # Ensure no zero std
        stats['std'] = np.maximum(stats['std'], 1e-6)

        print(f"✅ Stats: mean range [{stats['mean'].min():.3f}, {stats['mean'].max():.3f}]")
        print(f"✅ Stats: std range [{stats['std'].min():.3f}, {stats['std'].max():.3f}]")

        return stats

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage"""

        # Compute statistics if needed
        if self.standardize and self.stats is None:
            self.stats = self._compute_statistics()

        if stage == "fit" or stage is None:
            train_files = self._load_split_files('train')
            val_files = self._load_split_files('val')

            self.train_dataset = ComprehensivePoseDataset(
                train_files, 'train', self.return_dict, self.standardize, self.stats
            )
            self.val_dataset = ComprehensivePoseDataset(
                val_files, 'val', self.return_dict, self.standardize, self.stats
            )

        if stage == "test" or stage is None:
            test_files = self._load_split_files('test')
            self.test_dataset = ComprehensivePoseDataset(
                test_files, 'test', self.return_dict, self.standardize, self.stats
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )


def main():
    """Test the comprehensive pose datamodule"""

    print("🧪 Testing Comprehensive Pose DataModule")
    print("=" * 50)

    # Create datamodule
    dm = ComprehensivePoseDataModule(
        data_dir="../data",
        splits_dir="test_dataset_splits",
        batch_size=8,
        return_dict=True,  # Test dict format
        standardize=True
    )

    try:
        dm.prepare_data()
        dm.setup()
        print("✅ DataModule setup successful")

        # Test train loader
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        print(f"\n📊 Batch format test:")
        if isinstance(batch, dict):
            for key, value in batch.items():
                print(f"  {key}: {value.shape}")
        else:
            print(f"  Tensor shape: {batch.shape}")

        print(f"\n🎯 Expected format: (B, 53, 3)")
        print(f"  Global: (B, 1, 3)")
        print(f"  Body: (B, 21, 3)")
        print(f"  Jaw: (B, 1, 3)")
        print(f"  L.Hand: (B, 15, 3)")
        print(f"  R.Hand: (B, 15, 3)")
        print(f"  Total: 1+21+1+15+15 = 53 joints")

        # Test tensor format
        print(f"\n🔧 Testing tensor format...")
        dm_tensor = ComprehensivePoseDataModule(
            data_dir="../data",
            splits_dir="test_dataset_splits",
            batch_size=8,
            return_dict=False  # Tensor format
        )
        dm_tensor.setup()

        tensor_loader = dm_tensor.train_dataloader()
        tensor_batch = next(iter(tensor_loader))
        print(f"  Tensor batch: {tensor_batch.shape}")

        print(f"\n🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)