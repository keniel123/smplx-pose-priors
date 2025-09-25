#!/usr/bin/env python3
"""
Create consolidated NPZ files from dataset splits for efficient VAE training

This script takes the JSON split files and creates single consolidated NPZ files
for train/val/test, which is more efficient for VAE training than loading from
multiple source files.
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def create_consolidated_splits(
    data_dir: str = "../data",
    splits_dir: str = "dataset_splits",
    output_dir: str = "consolidated_splits",
):
    """Create consolidated NPZ files from JSON splits"""

    data_dir = Path(data_dir)
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"🔄 Creating consolidated NPZ files...")
    print(f"  Input data: {data_dir}")
    print(f"  Input splits: {splits_dir}")
    print(f"  Output dir: {output_dir}")

    # Process each split
    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}_samples.json"

        if not split_file.exists():
            print(f"⚠️  Skipping {split_name} - file not found: {split_file}")
            continue

        print(f"\\n📝 Processing {split_name} split...")

        # Load split samples
        with open(split_file, 'r') as f:
            samples_dict = json.load(f)

        # Count total samples
        total_samples = sum(len(indices) for indices in samples_dict.values())
        print(f"  Total samples: {total_samples:,}")

        # Collect all hand poses
        lhand_poses = []
        rhand_poses = []
        metadata = []

        sample_count = 0

        for dataset_name, indices in tqdm(samples_dict.items(), desc=f"Loading {split_name}"):
            # Load dataset NPZ
            npz_path = data_dir / f"{dataset_name}.npz"

            if not npz_path.exists():
                print(f"⚠️  Dataset not found: {npz_path}")
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)

                # Extract samples for this dataset
                dataset_lhand = data['lhand_pose'][indices]  # (n_samples, 45)
                dataset_rhand = data['rhand_pose'][indices]  # (n_samples, 45)

                lhand_poses.append(dataset_lhand)
                rhand_poses.append(dataset_rhand)

                # Create metadata
                for i, idx in enumerate(indices):
                    metadata.append({
                        'dataset_name': dataset_name,
                        'original_idx': idx,
                        'consolidated_idx': sample_count + i
                    })

                sample_count += len(indices)
                print(f"    {dataset_name}: {len(indices):,} samples")

            except Exception as e:
                print(f"❌ Error loading {dataset_name}: {e}")

        if not lhand_poses:
            print(f"⚠️  No valid data found for {split_name}")
            continue

        # Concatenate all poses
        print(f"  Consolidating {len(lhand_poses)} datasets...")
        lhand_consolidated = np.concatenate(lhand_poses, axis=0)  # (total_samples, 45)
        rhand_consolidated = np.concatenate(rhand_poses, axis=0)  # (total_samples, 45)

        print(f"  Final shapes: lhand={lhand_consolidated.shape}, rhand={rhand_consolidated.shape}")

        # Save consolidated NPZ
        output_npz = output_dir / f"{split_name}.npz"

        np.savez_compressed(
            output_npz,
            lhand_pose=lhand_consolidated,
            rhand_pose=rhand_consolidated,
        )

        # Save metadata separately
        metadata_file = output_dir / f"{split_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        file_size_mb = output_npz.stat().st_size / (1024 * 1024)
        print(f"✅ Saved {split_name}: {total_samples:,} samples, {file_size_mb:.1f} MB")
        print(f"   NPZ: {output_npz}")
        print(f"   Metadata: {metadata_file}")

    # Create usage example
    example_file = output_dir / "usage_example.py"
    example_code = '''#!/usr/bin/env python3
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
'''

    with open(example_file, 'w') as f:
        f.write(example_code)

    print(f"\\n✅ Consolidation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"💡 Usage example: {example_file}")

    # Print summary
    print(f"\\n📊 Summary:")
    for split_name in ["train", "val", "test"]:
        npz_file = output_dir / f"{split_name}.npz"
        if npz_file.exists():
            # Quick check of file
            data = np.load(npz_file)
            n_samples = data['lhand_pose'].shape[0]
            file_size_mb = npz_file.stat().st_size / (1024 * 1024)
            print(f"  {split_name}: {n_samples:,} samples, {file_size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create consolidated NPZ files from dataset splits")
    parser.add_argument("--data_dir", default="../data", help="Directory containing source NPZ files")
    parser.add_argument("--splits_dir", default="dataset_splits", help="Directory containing JSON split files")
    parser.add_argument("--output_dir", default="consolidated_splits", help="Output directory for consolidated files")

    args = parser.parse_args()

    create_consolidated_splits(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir
    )