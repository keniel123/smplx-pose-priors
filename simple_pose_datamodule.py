#!/usr/bin/env python3
"""
Simple Pose DataModule for Consolidated Datasets

Uses the consolidated train.npz, val.npz, test.npz files created by
create_consolidated_dataset_splits.py

Much simpler and more efficient than loading individual datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union


class SimplePoseDataset(Dataset):
    """Dataset for consolidated pose data"""

    def __init__(
        self,
        npz_file: str,
        split: str = 'train',
        return_dict: bool = False,
        standardize: bool = True,
        stats: Optional[Dict] = None
    ):
        self.npz_file = npz_file
        self.split = split
        self.return_dict = return_dict
        self.standardize = standardize
        self.stats = stats

        # Load consolidated data
        self.data = self._load_consolidated_data()
        print(f"✅ Loaded {len(self.data)} samples for {split} from {Path(npz_file).name}")

    def _load_consolidated_data(self) -> np.ndarray:
        """Load consolidated pose data from NPZ file"""

        try:
            data = np.load(self.npz_file, allow_pickle=True)

            # Extract pose components
            pose_components = []

            # Global orientation (3,)
            global_orient = data.get('global_orient', np.array([]).reshape(0, 3))
            if global_orient.size > 0:
                pose_components.append(global_orient)
            else:
                raise ValueError(f"No global_orient data in {self.npz_file}")

            # Body pose (63,) - 21 joints * 3
            body_pose = data.get('body_pose', np.array([]).reshape(0, 63))
            if body_pose.size > 0:
                pose_components.append(body_pose)
            else:
                n_samples = len(global_orient)
                pose_components.append(np.zeros((n_samples, 63)))

            # Jaw pose (3,)
            jaw_pose = data.get('jaw_pose', np.array([]).reshape(0, 3))
            if jaw_pose.size > 0:
                pose_components.append(jaw_pose)
            else:
                n_samples = len(global_orient)
                pose_components.append(np.zeros((n_samples, 3)))

            # Left hand pose (45,) - 15 joints * 3
            lhand_pose = data.get('lhand_pose', np.array([]).reshape(0, 45))
            if lhand_pose.size > 0:
                pose_components.append(lhand_pose)
            else:
                n_samples = len(global_orient)
                pose_components.append(np.zeros((n_samples, 45)))

            # Right hand pose (45,) - 15 joints * 3
            rhand_pose = data.get('rhand_pose', np.array([]).reshape(0, 45))
            if rhand_pose.size > 0:
                pose_components.append(rhand_pose)
            else:
                n_samples = len(global_orient)
                pose_components.append(np.zeros((n_samples, 45)))

            # Concatenate all components: 3 + 63 + 3 + 45 + 45 = 159
            poses_159 = np.concatenate(pose_components, axis=1)

            # Convert to 55 joints (165D) by inserting eye joints as zeros
            n_samples = poses_159.shape[0]
            poses_165 = np.zeros((n_samples, 165))  # 55 * 3 = 165

            # Reshape to process joints
            poses_53x3 = poses_159.reshape(n_samples, 53, 3)  # Current: 53 joints
            poses_55x3 = np.zeros((n_samples, 55, 3))         # Target: 55 joints

            # Copy joints in correct SMPL-X order:
            # Source: [global(1), body(21), jaw(1), lhand(15), rhand(15)] = 53 joints
            # Target: [global(1), body(21), jaw(1), leye(1), reye(1), lhand(15), rhand(15)] = 55 joints

            # Copy global orient (joint 0)
            poses_55x3[:, 0, :] = poses_53x3[:, 0, :]

            # Copy body pose (joints 1-21)
            poses_55x3[:, 1:22, :] = poses_53x3[:, 1:22, :]

            # Copy jaw pose (joint 22)
            poses_55x3[:, 22, :] = poses_53x3[:, 22, :]

            # Joints 23, 24 (eye joints) stay as zeros - leye_pose, reye_pose

            # Copy left hand pose (joints 25-39) from source joints 23-37
            poses_55x3[:, 25:40, :] = poses_53x3[:, 23:38, :]

            # Copy right hand pose (joints 40-54) from source joints 38-52
            poses_55x3[:, 40:55, :] = poses_53x3[:, 38:53, :]

            # Flatten back to (N, 165)
            poses_165 = poses_55x3.reshape(n_samples, 165)

            print(f"  📊 Converted {poses_159.shape} -> {poses_165.shape} (55 joints with eye zeros)")

            return poses_165

        except Exception as e:
            raise ValueError(f"Error loading {self.npz_file}: {e}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        pose_165d = self.data[idx]  # Shape: (165,)

        # Apply standardization if available
        if self.standardize and self.stats is not None:
            pose_165d = (pose_165d - self.stats['mean']) / (self.stats['std'] + 1e-8)

        # Convert to tensor
        pose_tensor = torch.from_numpy(pose_165d).float()

        if self.return_dict:
            # Reshape to (55, 3) and split into components
            pose_55x3 = pose_tensor.view(55, 3)

            return {
                'global_orient': pose_55x3[0:1],      # (1, 3) - global orientation
                'body_pose': pose_55x3[1:22],         # (21, 3) - body joints
                'jaw_pose': pose_55x3[22:23],         # (1, 3) - jaw joint
                'eye_pose': pose_55x3[23:25],         # (2, 3) - eye joints (zeros)
                'lhand_pose': pose_55x3[25:40],       # (15, 3) - left hand joints
                'rhand_pose': pose_55x3[40:55],       # (15, 3) - right hand joints
                'full_pose': pose_55x3                # (55, 3) - everything
            }
        else:
            # Return as (55, 3) tensor
            return pose_tensor.view(55, 3)


class SimplePoseDataModule(pl.LightningDataModule):
    """Simple PyTorch Lightning DataModule for consolidated pose data"""

    def __init__(
        self,
        splits_dir: str = "dataset_splits",
        batch_size: int = 64,
        num_workers: int = 4,
        return_dict: bool = False,
        standardize: bool = False,
        pin_memory: bool = True
    ):
        super().__init__()
        self.splits_dir = Path(splits_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_dict = return_dict
        self.standardize = standardize
        self.pin_memory = pin_memory

        # Will store computed statistics
        self.stats = None

    def prepare_data(self):
        """Ensure consolidated data files exist"""
        required_files = ['train.npz', 'val.npz', 'test.npz']

        for file in required_files:
            file_path = self.splits_dir / file
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required consolidated file not found: {file_path}\n"
                    f"Please run create_consolidated_dataset_splits.py first!"
                )

    def _compute_statistics(self) -> Dict[str, np.ndarray]:
        """Compute mean and std from training data"""
        print("📊 Computing dataset statistics from consolidated train.npz...")

        train_file = self.splits_dir / "train.npz"

        try:
            data = np.load(train_file, allow_pickle=True)

            # Extract and concatenate pose components (same as dataset loading)
            pose_components = []

            global_orient = data.get('global_orient', np.array([]))
            body_pose = data.get('body_pose', np.array([]))
            jaw_pose = data.get('jaw_pose', np.array([]))
            lhand_pose = data.get('lhand_pose', np.array([]))
            rhand_pose = data.get('rhand_pose', np.array([]))

            if global_orient.size == 0:
                raise ValueError("No global_orient data for statistics")

            n_samples = len(global_orient)

            # Use actual data or zeros
            pose_components.append(global_orient)
            pose_components.append(body_pose if body_pose.size > 0 else np.zeros((n_samples, 63)))
            pose_components.append(jaw_pose if jaw_pose.size > 0 else np.zeros((n_samples, 3)))
            pose_components.append(lhand_pose if lhand_pose.size > 0 else np.zeros((n_samples, 45)))
            pose_components.append(rhand_pose if rhand_pose.size > 0 else np.zeros((n_samples, 45)))

            # Combine and convert to 55 joints format (same as dataset)
            poses_159 = np.concatenate(pose_components, axis=1)

            # Convert to 165D (55 joints) - same logic as dataset loading
            poses_53x3 = poses_159.reshape(n_samples, 53, 3)
            poses_55x3 = np.zeros((n_samples, 55, 3))

            # Copy joints in correct SMPL-X order:
            poses_55x3[:, 0, :] = poses_53x3[:, 0, :]      # global orient
            poses_55x3[:, 1:22, :] = poses_53x3[:, 1:22, :] # body pose
            poses_55x3[:, 22, :] = poses_53x3[:, 22, :]     # jaw pose
            # Eye joints 23, 24 stay as zeros
            poses_55x3[:, 25:40, :] = poses_53x3[:, 23:38, :] # left hand
            poses_55x3[:, 40:55, :] = poses_53x3[:, 38:53, :] # right hand

            poses_165 = poses_55x3.reshape(n_samples, 165)

            print(f"📈 Computing statistics from {poses_165.shape} samples")

            # Use subset for efficiency if dataset is very large
            if len(poses_165) > 100000:
                subset_indices = np.random.choice(len(poses_165), 100000, replace=False)
                poses_subset = poses_165[subset_indices]
                print(f"  Using subset of {len(poses_subset)} samples for statistics")
            else:
                poses_subset = poses_165

            # Compute statistics
            stats = {
                'mean': poses_subset.mean(axis=0),
                'std': poses_subset.std(axis=0),
                'min': poses_subset.min(axis=0),
                'max': poses_subset.max(axis=0)
            }

            # Ensure no zero std
            stats['std'] = np.maximum(stats['std'], 1e-6)

            print(f"✅ Stats computed - mean range: [{stats['mean'].min():.3f}, {stats['mean'].max():.3f}]")
            print(f"✅ Stats computed - std range: [{stats['std'].min():.3f}, {stats['std'].max():.3f}]")

            return stats

        except Exception as e:
            raise ValueError(f"Error computing statistics: {e}")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage"""

        # Compute statistics if needed
        if self.standardize and self.stats is None:
            self.stats = self._compute_statistics()

        if stage == "fit" or stage is None:
            self.train_dataset = SimplePoseDataset(
                str(self.splits_dir / "train.npz"),
                'train', self.return_dict, self.standardize, self.stats
            )
            self.val_dataset = SimplePoseDataset(
                str(self.splits_dir / "val.npz"),
                'val', self.return_dict, self.standardize, self.stats
            )

        if stage == "test" or stage is None:
            self.test_dataset = SimplePoseDataset(
                str(self.splits_dir / "test.npz"),
                'test', self.return_dict, self.standardize, self.stats
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
    """Test the simple pose datamodule"""

    print("🧪 Testing Simple Pose DataModule")
    print("=" * 50)

    # Create datamodule
    dm = SimplePoseDataModule(
        splits_dir="dataset_splits",
        batch_size=8,
        return_dict=True,
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

        print(f"\n🎯 Expected format: (B, 55, 3)")
        print(f"  Global: (B, 1, 3)")
        print(f"  Body: (B, 21, 3)")
        print(f"  Jaw: (B, 1, 3)")
        print(f"  Eyes: (B, 2, 3) - zeros")
        print(f"  L.Hand: (B, 15, 3)")
        print(f"  R.Hand: (B, 15, 3)")
        print(f"  Total: 1+21+1+2+15+15 = 55 joints")

        # Test tensor format
        print(f"\n🔧 Testing tensor format...")
        dm_tensor = SimplePoseDataModule(
            splits_dir="dataset_splits",
            batch_size=8,
            return_dict=False
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