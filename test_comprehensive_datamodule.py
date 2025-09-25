#!/usr/bin/env python3
"""
Quick test for the comprehensive pose datamodule using existing hand VAE setup
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from comprehensive_pose_datamodule import ComprehensivePoseDataset


def test_comprehensive_loading():
    """Test comprehensive pose loading with sample data"""

    print("🧪 Testing Comprehensive Pose Loading")
    print("=" * 40)

    # Get some NPZ files from data directory
    data_dir = Path("../data")
    npz_files = list(data_dir.glob("*.npz"))[:3]  # Use first 3 files for testing

    if not npz_files:
        print("❌ No NPZ files found in ../data/")
        return False

    print(f"📁 Testing with {len(npz_files)} files:")
    for f in npz_files:
        print(f"  - {f.name}")

    # Test dataset creation
    try:
        dataset = ComprehensivePoseDataset(
            npz_files=[str(f) for f in npz_files],
            split='test',
            return_dict=False,  # Test tensor format first
            standardize=False   # No stats for this test
        )

        print(f"✅ Dataset created: {len(dataset)} samples")

        # Test single sample
        sample = dataset[0]
        print(f"✅ Sample shape: {sample.shape}")
        print(f"  Expected: (53, 3) for full pose")

        if sample.shape == (53, 3):
            print("✅ Shape is correct!")

            # Break down the components
            global_orient = sample[0:1]      # (1, 3)
            body_pose = sample[1:22]         # (21, 3)
            jaw_pose = sample[22:23]         # (1, 3)
            lhand_pose = sample[23:38]       # (15, 3)
            rhand_pose = sample[38:53]       # (15, 3)

            print(f"  Global Orient: {global_orient.shape}")
            print(f"  Body Pose: {body_pose.shape}")
            print(f"  Jaw Pose: {jaw_pose.shape}")
            print(f"  L Hand Pose: {lhand_pose.shape}")
            print(f"  R Hand Pose: {rhand_pose.shape}")
            print(f"  Total: {1+21+1+15+15} = 53 joints ✅")

        else:
            print(f"❌ Wrong shape! Expected (53, 3), got {sample.shape}")
            return False

    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return False

    # Test dict format
    print(f"\n🔧 Testing dict format...")
    try:
        dataset_dict = ComprehensivePoseDataset(
            npz_files=[str(f) for f in npz_files],
            split='test',
            return_dict=True,   # Dict format
            standardize=False
        )

        sample_dict = dataset_dict[0]

        print(f"✅ Dict format test:")
        for key, value in sample_dict.items():
            print(f"  {key}: {value.shape}")

        # Verify total dimensions
        full_pose = sample_dict['full_pose']
        if full_pose.shape == (53, 3):
            print("✅ Dict format correct!")
        else:
            print(f"❌ Dict format wrong shape: {full_pose.shape}")
            return False

    except Exception as e:
        print(f"❌ Dict format test failed: {e}")
        return False

    # Test batch processing
    print(f"\n🔧 Testing batch processing...")
    try:
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        print(f"✅ Batch shape: {batch.shape}")
        print(f"  Expected: (4, 53, 3)")

        if batch.shape == (4, 53, 3):
            print("✅ Batch processing works!")
        else:
            print(f"❌ Wrong batch shape: {batch.shape}")
            return False

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False

    print(f"\n🎉 All comprehensive datamodule tests passed!")
    print(f"\n💡 Summary:")
    print(f"  • Loads full SMPL-X pose: global + body + jaw + hands")
    print(f"  • Returns (B, 53, 3) axis-angle format")
    print(f"  • Supports both tensor and dict return formats")
    print(f"  • Compatible with PyTorch DataLoader")
    print(f"  • Ready for comprehensive pose modeling!")

    return True


if __name__ == "__main__":
    success = test_comprehensive_loading()
    exit(0 if success else 1)