#!/usr/bin/env python3
"""
Full-Body Pose Dataset Creation Script

Creates dataset splits for full-body pose training with proper weighting:
- Motion capture datasets (upweighted): MotionX, Bedlam, MSCOCO, IDEA400, etc.
- Sitting/static datasets (downweighted): SignLanguage, UBody, entertainment, etc.

Train/Val/Test split: 80/10/10
"""

import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm


class FullBodyPoseDatasetCreator:
    def __init__(self, data_dir: str, output_dir: str = "dataset_splits"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Dataset categorization for full-body pose training
        # HIGH WEIGHT: Motion capture, dynamic activities, diverse poses
        self.high_motion_datasets = [
            # Motion capture datasets - high quality poses
            "MotionXmusic",
            "MotionXperform",
            "MotionXsports",
            "MotionXdance",
            "DanceDB",
            "EyesJapan",
            "Bedlam",           # Synthetic with diverse poses
            "MSCOCO",           # Natural pose variations
            "IDEA400",          # Diverse activities
            "HUMAN4D",          # Dynamic sequences
            "GRAB",             # Interaction poses
            "MOYO",             # Multi-person interactions
            "TCDHands",         # Hand-object interactions
            "ACCAD",            # Motion capture
            "CMU",              # Classic mocap
            "KIT",              # KIT motion capture
            "HDM05",            # Heidelberg mocap
            "SFU",              # Simon Fraser mocap
            "TotalCapture",     # Multi-modal capture
            "WEIZMANN",         # Action sequences
            "EKUT",             # Walking sequences
            "BMLrub",           # BioMotionLab
            "BMLmovi",          # Multi-modal
            "humman",           # Human motion
            "arctic",           # Hand-object
            "ham",              # Hand activities
        ]

        # LOW WEIGHT: Static poses, sitting activities, limited motion
        self.low_motion_datasets = [
            # Sitting/static activities
            "signlanguage",     # Often seated signing
            "UBody",            # Upper body focus, often seated
            "entertainment",    # TV/media (often seated)
            "tvshow",           # Television (seated/static)
            "livevlog",         # Streaming (seated)
            "fitness",          # Some exercises (but limited variation)
            "conductmusic",     # Conducting (limited body motion)
            "magic",            # Performance (limited full-body)
            "movie",            # Acting (variable, but often scripted)
            "speech",           # Speaking (minimal body motion)
            "talkshow",         # Talk shows (seated)
            "videoconference",  # Video calls (seated/static)
            "singing",          # Singing (upper body focus)
        ]

        # Target samples for balanced training
        self.target_total = 1_500_000  # 1.5M total samples

        # Weights: High motion gets 75%, Low motion gets 25%
        self.high_motion_weight = 0.75  # 1.125M samples
        self.low_motion_weight = 0.25   # 0.375M samples

        # Split ratios
        self.split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

    def scan_data_directory(self) -> Dict[str, Dict[str, int]]:
        """Scan data directory and categorize datasets"""
        print(f"📂 Scanning data directory: {self.data_dir}")

        dataset_info = {
            'high_motion': {},  # dataset_name -> frame_count
            'low_motion': {},
            'unknown': {}
        }

        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return dataset_info

        # Scan NPZ files
        npz_files = list(self.data_dir.glob("*.npz"))
        print(f"📊 Found {len(npz_files)} NPZ files")

        for npz_file in tqdm(npz_files, desc="Analyzing datasets"):
            try:
                # Load NPZ to get sample count
                data = np.load(npz_file, allow_pickle=True)

                # Get sample count (assuming first key contains the main data)
                keys = list(data.keys())
                if not keys:
                    continue

                sample_count = len(data[keys[0]])
                dataset_name = npz_file.stem

                # Categorize dataset
                category = self._categorize_dataset(dataset_name)
                dataset_info[category][dataset_name] = sample_count

                print(f"  📁 {dataset_name}: {sample_count:,} samples ({category})")

            except Exception as e:
                print(f"⚠️ Error reading {npz_file}: {e}")
                continue

        return dataset_info

    def _categorize_dataset(self, dataset_name: str) -> str:
        """Categorize dataset based on motion type"""
        dataset_lower = dataset_name.lower()

        # Check high motion datasets
        for high_dataset in self.high_motion_datasets:
            if high_dataset.lower() in dataset_lower:
                return 'high_motion'

        # Check low motion datasets
        for low_dataset in self.low_motion_datasets:
            if low_dataset.lower() in dataset_lower:
                return 'low_motion'

        # Default to unknown (will be treated as medium priority)
        return 'unknown'

    def calculate_sampling_rates(self, dataset_info: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        """Calculate sampling rates for each dataset"""
        print(f"\n🔢 Calculating sampling rates...")

        # Calculate total samples per category
        high_motion_total = sum(dataset_info['high_motion'].values())
        low_motion_total = sum(dataset_info['low_motion'].values())
        unknown_total = sum(dataset_info['unknown'].values())

        print(f"📊 Category totals:")
        print(f"  High motion: {high_motion_total:,} samples")
        print(f"  Low motion: {low_motion_total:,} samples")
        print(f"  Unknown: {unknown_total:,} samples")

        # Target samples per category
        high_motion_target = int(self.target_total * self.high_motion_weight)
        low_motion_target = int(self.target_total * self.low_motion_weight)

        # Unknown datasets get remaining allocation
        remaining_target = self.target_total - high_motion_target - low_motion_target

        print(f"\n🎯 Target samples:")
        print(f"  High motion: {high_motion_target:,} samples ({self.high_motion_weight:.1%})")
        print(f"  Low motion: {low_motion_target:,} samples ({self.low_motion_weight:.1%})")
        print(f"  Unknown: {remaining_target:,} samples")

        # Calculate sampling rates
        sampling_rates = {
            'high_motion': {},
            'low_motion': {},
            'unknown': {}
        }

        # High motion datasets - higher sampling rate
        if high_motion_total > 0:
            high_motion_rate = high_motion_target / high_motion_total
            for dataset in dataset_info['high_motion']:
                sampling_rates['high_motion'][dataset] = min(1.0, high_motion_rate)

        # Low motion datasets - lower sampling rate
        if low_motion_total > 0:
            low_motion_rate = low_motion_target / low_motion_total
            for dataset in dataset_info['low_motion']:
                sampling_rates['low_motion'][dataset] = min(1.0, low_motion_rate)

        # Unknown datasets - medium sampling rate
        if unknown_total > 0:
            unknown_rate = remaining_target / unknown_total
            for dataset in dataset_info['unknown']:
                sampling_rates['unknown'][dataset] = min(1.0, unknown_rate)

        return sampling_rates

    def create_dataset_splits(self, dataset_info: Dict, sampling_rates: Dict) -> Dict[str, List]:
        """Create train/val/test splits with sampling"""
        print(f"\n📝 Creating dataset splits...")

        all_samples = []

        # Process each category
        for category in ['high_motion', 'low_motion', 'unknown']:
            category_datasets = dataset_info[category]
            category_rates = sampling_rates[category]

            print(f"\n🔄 Processing {category} datasets:")

            for dataset_name, total_samples in category_datasets.items():
                sampling_rate = category_rates[dataset_name]
                target_samples = int(total_samples * sampling_rate)

                print(f"  📁 {dataset_name}: {total_samples:,} → {target_samples:,} samples (rate: {sampling_rate:.3f})")

                # Create file path
                npz_file = str(self.data_dir / f"{dataset_name}.npz")

                # Sample indices
                if sampling_rate >= 1.0:
                    # Use all samples
                    sample_indices = list(range(total_samples))
                else:
                    # Random sampling
                    sample_indices = sorted(random.sample(range(total_samples), target_samples))

                # Add to all samples
                for idx in sample_indices:
                    all_samples.append({
                        'file': npz_file,
                        'index': idx,
                        'dataset': dataset_name,
                        'category': category
                    })

        print(f"\n📊 Total samples collected: {len(all_samples):,}")

        # Shuffle all samples
        random.shuffle(all_samples)

        # Create splits
        total_samples = len(all_samples)
        train_end = int(total_samples * self.split_ratios['train'])
        val_end = train_end + int(total_samples * self.split_ratios['val'])

        splits = {
            'train': all_samples[:train_end],
            'val': all_samples[train_end:val_end],
            'test': all_samples[val_end:]
        }

        print(f"\n✅ Split summary:")
        for split_name, samples in splits.items():
            print(f"  {split_name}: {len(samples):,} samples")

        return splits

    def save_splits(self, splits: Dict[str, List]):
        """Save splits to files"""
        print(f"\n💾 Saving splits to {self.output_dir}")

        # Save file lists (compatible with comprehensive_pose_datamodule)
        for split_name, samples in splits.items():
            # Create file list
            file_list = []
            for sample in samples:
                file_list.append(sample['file'])

            # Remove duplicates and sort
            unique_files = sorted(list(set(file_list)))

            # Save file list
            output_file = self.output_dir / f"{split_name}_files.txt"
            with open(output_file, 'w') as f:
                for file_path in unique_files:
                    f.write(f"{file_path}\n")

            print(f"  📝 {split_name}_files.txt: {len(unique_files)} unique files")

        # Save detailed sample info
        for split_name, samples in splits.items():
            output_file = self.output_dir / f"{split_name}_samples.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)

            print(f"  📝 {split_name}_samples.json: {len(samples)} samples with indices")

        # Save metadata
        metadata = {
            'total_samples': sum(len(samples) for samples in splits.values()),
            'splits': {name: len(samples) for name, samples in splits.items()},
            'target_total': self.target_total,
            'high_motion_weight': self.high_motion_weight,
            'low_motion_weight': self.low_motion_weight,
            'split_ratios': self.split_ratios,
            'high_motion_datasets': self.high_motion_datasets,
            'low_motion_datasets': self.low_motion_datasets
        }

        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  📝 dataset_metadata.json: Saved configuration")


def main():
    """Main function"""
    print("🚀 Creating Full-Body Pose Dataset Splits")
    print("=" * 60)

    # Configuration
    data_dir = "../data"  # Update this path

    # Create dataset creator
    creator = FullBodyPoseDatasetCreator(data_dir)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Scan data directory
    dataset_info = creator.scan_data_directory()

    if not any(dataset_info.values()):
        print("❌ No valid datasets found!")
        return

    # Calculate sampling rates
    sampling_rates = creator.calculate_sampling_rates(dataset_info)

    # Create splits
    splits = creator.create_dataset_splits(dataset_info, sampling_rates)

    # Save splits
    creator.save_splits(splits)

    print(f"\n🎉 Dataset splits created successfully!")
    print(f"📁 Output directory: {creator.output_dir}")
    print(f"📊 Use these files with comprehensive_pose_datamodule.py")


if __name__ == "__main__":
    main()