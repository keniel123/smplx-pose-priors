#!/usr/bin/env python3
"""
Test the PyTorch DataModule with the test dataset splits
"""

import sys
import os
from pathlib import Path

# Add code directory to path
sys.path.append(str(Path(__file__).parent))

from hand_prior_datamodule import HandPriorDataModule, HandPriorTransform


def test_datamodule():
    """Test the PyTorch Lightning DataModule"""

    print("🧪 Testing Hand Prior DataModule")
    print("=" * 40)

    # Create transform
    transform = HandPriorTransform(normalize_poses=True, add_noise=False)

    # Create datamodule with test splits
    dm = HandPriorDataModule(
        data_dir="../data",
        splits_dir="test_dataset_splits",  # Use test splits
        batch_size=8,  # Small batch for testing
        num_workers=0,  # No multiprocessing for testing
        transform=transform
    )

    print("📋 Preparing data...")
    try:
        dm.prepare_data()
        print("✅ Data preparation successful")
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False

    print("🔧 Setting up datasets...")
    try:
        dm.setup()
        print("✅ Dataset setup successful")

        # Print dataset info
        metadata = dm.get_dataset_info()
        print(f"📊 Dataset info:")
        creation_info = metadata['test_dataset_info']
        print(f"  Total samples: {creation_info['total_samples']:,}")
        for split_name, count in creation_info['splits'].items():
            print(f"  {split_name}: {count:,}")

    except Exception as e:
        print(f"❌ Dataset setup failed: {e}")
        return False

    # Test train dataloader
    print("\n🔄 Testing train dataloader...")
    try:
        train_loader = dm.train_dataloader()
        print(f"Train loader created with {len(train_loader)} batches")

        # Test first batch
        batch = next(iter(train_loader))
        print(f"✅ First batch loaded successfully:")
        print(f"  Batch size: {len(batch['lhand_pose'])}")
        print(f"  lhand_pose shape: {batch['lhand_pose'].shape}")
        print(f"  rhand_pose shape: {batch['rhand_pose'].shape}")
        print(f"  Datasets in batch: {set(batch['dataset_name'])}")
        print(f"  Data types: lhand_pose={batch['lhand_pose'].dtype}")

        # Test data ranges
        print(f"  lhand_pose range: [{batch['lhand_pose'].min():.3f}, {batch['lhand_pose'].max():.3f}]")
        print(f"  rhand_pose range: [{batch['rhand_pose'].min():.3f}, {batch['rhand_pose'].max():.3f}]")

    except Exception as e:
        print(f"❌ Train dataloader test failed: {e}")
        return False

    # Test val dataloader
    print("\n🔄 Testing val dataloader...")
    try:
        val_loader = dm.val_dataloader()
        print(f"Val loader created with {len(val_loader)} batches")

        if len(val_loader) > 0:
            batch = next(iter(val_loader))
            print(f"✅ Val batch loaded: {len(batch['lhand_pose'])} samples")
        else:
            print("⚠️  Val loader is empty")

    except Exception as e:
        print(f"❌ Val dataloader test failed: {e}")
        return False

    # Test test dataloader
    print("\n🔄 Testing test dataloader...")
    try:
        test_loader = dm.test_dataloader()
        print(f"Test loader created with {len(test_loader)} batches")

        if len(test_loader) > 0:
            batch = next(iter(test_loader))
            print(f"✅ Test batch loaded: {len(batch['lhand_pose'])} samples")
        else:
            print("⚠️  Test loader is empty")

    except Exception as e:
        print(f"❌ Test dataloader test failed: {e}")
        return False

    print("\n✅ All DataModule tests passed!")
    return True


def test_multiple_batches():
    """Test loading multiple batches to check for memory leaks"""

    print("\n🔄 Testing multiple batch loading...")

    dm = HandPriorDataModule(
        data_dir="../data",
        splits_dir="test_dataset_splits",
        batch_size=4,
        num_workers=0,
    )

    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()

    # Load first few batches
    for i, batch in enumerate(train_loader):
        print(f"  Batch {i+1}: {len(batch['lhand_pose'])} samples from datasets: {set(batch['dataset_name'])}")

        if i >= 4:  # Test first 5 batches
            break

    print("✅ Multiple batch loading successful")


if __name__ == "__main__":
    success = test_datamodule()

    if success:
        test_multiple_batches()
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)