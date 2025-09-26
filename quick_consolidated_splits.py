#!/usr/bin/env python3
"""
Quick Consolidated Dataset Creation

Create train.npz, val.npz, test.npz from existing sample JSON files
Much faster approach - just loads the specific samples listed in the JSON files.
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


def create_consolidated_from_json():
    """Create consolidated splits from existing sample JSON files"""
    print("🚀 Creating consolidated splits from sample JSONs...")

    splits_dir = Path("dataset_splits")
    data_dir = Path("../data")

    # Check if sample JSON files exist
    json_files = {
        'train': splits_dir / 'train_samples.json',
        'val': splits_dir / 'val_samples.json',
        'test': splits_dir / 'test_samples.json'
    }

    for split_name, json_file in json_files.items():
        if not json_file.exists():
            print(f"❌ {json_file} not found. Run create_fullbody_dataset_splits.py first!")
            return

    # Process each split
    for split_name, json_file in json_files.items():
        print(f"\n📊 Processing {split_name} split...")

        # Load sample information
        with open(json_file, 'r') as f:
            samples = json.load(f)

        print(f"  📋 {len(samples)} samples to process")

        # Group samples by dataset file
        file_samples = {}
        for sample in samples:
            file_path = sample['file']
            if file_path not in file_samples:
                file_samples[file_path] = []
            file_samples[file_path].append(sample['index'])

        print(f"  📁 Loading from {len(file_samples)} dataset files")

        # Storage for consolidated data
        all_global_orient = []
        all_body_pose = []
        all_jaw_pose = []
        all_lhand_pose = []
        all_rhand_pose = []

        # Process each file
        for file_path, indices in tqdm(file_samples.items(), desc=f"Loading {split_name}"):
            try:
                # Load NPZ file
                data = np.load(file_path, allow_pickle=True)

                # Extract specific samples by index
                indices = sorted(indices)  # Ensure sorted for efficiency

                # Global orientation (3,)
                if 'global_orient' in data:
                    global_orient = data['global_orient'][indices]
                else:
                    global_orient = np.zeros((len(indices), 3))

                # Body pose (63,) - 21 joints * 3
                if 'body_pose' in data:
                    body_pose = data['body_pose'][indices]
                else:
                    body_pose = np.zeros((len(indices), 63))

                # Jaw pose (3,)
                if 'jaw_pose' in data:
                    jaw_pose = data['jaw_pose'][indices]
                else:
                    jaw_pose = np.zeros((len(indices), 3))

                # Left hand pose (45,) - 15 joints * 3
                if 'lhand_pose' in data:
                    lhand_pose = data['lhand_pose'][indices]
                else:
                    lhand_pose = np.zeros((len(indices), 45))

                # Right hand pose (45,) - 15 joints * 3
                if 'rhand_pose' in data:
                    rhand_pose = data['rhand_pose'][indices]
                else:
                    rhand_pose = np.zeros((len(indices), 45))

                # Add to consolidated arrays
                all_global_orient.append(global_orient)
                all_body_pose.append(body_pose)
                all_jaw_pose.append(jaw_pose)
                all_lhand_pose.append(lhand_pose)
                all_rhand_pose.append(rhand_pose)

            except Exception as e:
                print(f"  ⚠️  Error loading {file_path}: {e}")
                continue

        # Concatenate all data
        print(f"  🔗 Consolidating {split_name} data...")

        consolidated_data = {}

        if all_global_orient:
            consolidated_data['global_orient'] = np.concatenate(all_global_orient, axis=0)
            consolidated_data['body_pose'] = np.concatenate(all_body_pose, axis=0)
            consolidated_data['jaw_pose'] = np.concatenate(all_jaw_pose, axis=0)
            consolidated_data['lhand_pose'] = np.concatenate(all_lhand_pose, axis=0)
            consolidated_data['rhand_pose'] = np.concatenate(all_rhand_pose, axis=0)

            print(f"  📊 Final shapes:")
            for key, value in consolidated_data.items():
                print(f"    {key}: {value.shape}")

        # Save consolidated split
        output_file = splits_dir / f"{split_name}.npz"
        np.savez_compressed(output_file, **consolidated_data)

        file_size = output_file.stat().st_size / (1024*1024)  # MB
        print(f"  💾 Saved {output_file} ({file_size:.1f} MB)")

    print(f"\n✅ All consolidated splits created!")
    print(f"📁 Files: train.npz, val.npz, test.npz")
    print(f"💡 Now you can use SimplePoseDataModule instead of ComprehensivePoseDataModule")


def test_consolidated_datamodule():
    """Test the new datamodule with consolidated data"""
    print(f"\n🧪 Testing SimplePoseDataModule...")

    try:
        from simple_pose_datamodule import SimplePoseDataModule

        # Create datamodule
        dm = SimplePoseDataModule(
            splits_dir="dataset_splits",
            batch_size=16,
            return_dict=False,  # Return as tensors
            standardize=True
        )

        # Setup
        dm.prepare_data()
        dm.setup()

        print("✅ SimplePoseDataModule setup successful")

        # Test a batch
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        print(f"📊 Batch shape: {batch.shape}")
        print(f"💡 Expected: (batch_size, 55, 3) = ({batch.shape[0]}, 55, 3)")

        if batch.shape[1:] == (55, 3):
            print("✅ Batch format is correct!")
        else:
            print(f"❌ Unexpected batch format: {batch.shape}")

        # Test dict format
        dm_dict = SimplePoseDataModule(
            splits_dir="dataset_splits",
            batch_size=8,
            return_dict=True
        )
        dm_dict.setup()

        dict_loader = dm_dict.train_dataloader()
        dict_batch = next(iter(dict_loader))

        print(f"\n📊 Dict format test:")
        for key, value in dict_batch.items():
            print(f"  {key}: {value.shape}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Step 1: Create consolidated files
    create_consolidated_from_json()

    # Step 2: Test the datamodule
    test_consolidated_datamodule()