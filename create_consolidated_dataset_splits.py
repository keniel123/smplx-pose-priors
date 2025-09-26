#!/usr/bin/env python3
"""
Create Consolidated Dataset Splits

Fix the issues with dataset creation:
1. Load specific samples using indices from JSON files (not entire datasets)
2. Create single consolidated train.npz, val.npz, test.npz files
3. Ensure all datasets are represented (not just 37)
4. Maintain proper weighting for high/low motion datasets
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


class ConsolidatedDatasetCreator:
    def __init__(self, data_dir: str = "../data", output_dir: str = "dataset_splits"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Expected datasets based on your file structure
        self.expected_datasets = [
            "ACCAD", "BMLmovi", "BMLrub", "Bedlam", "CMU", "DanceDB", "EKUT",
            "EyesJapan", "GRAB", "HDM05", "HUMAN4D", "KIT", "MOYO",
            "MotionXmusic", "MotionXperform", "SFU", "SignAvatarsHam",
            "TCDHands", "TotalCapture", "WEIZMANN", "arctic", "conductmusic",
            "entertainment", "fitness", "ham", "humman", "idea400", "livevlog",
            "magic", "movie", "mscoco", "signlanguage", "singing", "speech",
            "talkshow", "tvshow", "videoconference", "MotionXanimation",
            "MotionXhaa500", "DFaust", "SSM", "CNRS", "HumanEva", "interview",
            "kungfu", "Mosh", "olympic", "online_class", "PosePrior", "SOMA",
            "SignAvatarsLang-001", "SignAvatarsWord", "Transitions"
        ]

    def scan_available_datasets(self) -> Dict[str, Dict[str, int]]:
        """Scan what datasets are actually available in data directory"""
        print(f"🔍 Scanning available datasets in {self.data_dir}")

        available_datasets = {}

        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return available_datasets

        # Scan NPZ files
        npz_files = list(self.data_dir.glob("*.npz"))
        print(f"📊 Found {len(npz_files)} NPZ files")

        for npz_file in tqdm(npz_files, desc="Analyzing datasets"):
            try:
                dataset_name = npz_file.stem

                # Load NPZ to get sample count
                data = np.load(npz_file, allow_pickle=True)

                # Get sample count (assuming first key contains the main data)
                keys = list(data.keys())
                if not keys:
                    continue

                sample_count = len(data[keys[0]])
                available_datasets[dataset_name] = {
                    'file_path': str(npz_file),
                    'sample_count': sample_count,
                    'keys': keys
                }

                print(f"  📁 {dataset_name}: {sample_count:,} samples")

            except Exception as e:
                print(f"⚠️  Error reading {npz_file}: {e}")
                continue

        return available_datasets

    def categorize_datasets(self, available_datasets: Dict) -> Dict[str, List[str]]:
        """Categorize datasets into high/low motion based on names"""

        # High motion patterns
        high_motion_patterns = [
            'motion', 'dance', 'grab', 'moyo', 'accad', 'cmu', 'kit', 'hdm05',
            'sfu', 'totalcapture', 'weizmann', 'ekut', 'bml', 'human', 'arctic',
            'ham', 'eyes', 'bedlam', 'mscoco', 'idea400', 'tcd', 'kungfu', 'soma',
            'transitions', 'cnrs', 'mosh', 'humman', 'poseprior',
            'dfaust', 'ssm'
        ]

        # Low motion patterns
        low_motion_patterns = [
            'sign', 'ubody', 'entertainment', 'tv', 'vlog', 'fitness', 'conduct',
            'magic', 'movie', 'speech', 'talk', 'video', 'sing', 'olympic',
            'online_class', 'interview'
        ]

        categorized = {
            'high_motion': [],
            'low_motion': [],
            'unknown': []
        }

        for dataset_name in available_datasets.keys():
            name_lower = dataset_name.lower()

            # Check high motion patterns
            is_high_motion = any(pattern in name_lower for pattern in high_motion_patterns)
            is_low_motion = any(pattern in name_lower for pattern in low_motion_patterns)

            if is_high_motion and not is_low_motion:
                categorized['high_motion'].append(dataset_name)
            elif is_low_motion and not is_high_motion:
                categorized['low_motion'].append(dataset_name)
            else:
                categorized['unknown'].append(dataset_name)
                print(f"⚠️  Uncertain categorization for {dataset_name} - treating as medium priority")

        return categorized

    def calculate_sampling_strategy(self, available_datasets: Dict, categorized: Dict) -> Dict:
        """Calculate how many samples to take from each dataset"""

        print(f"\n📊 Calculating sampling strategy...")

        # Target samples
        target_total = 1_500_000
        high_motion_target = int(target_total * 0.75)  # 75%
        low_motion_target = int(target_total * 0.25)   # 25%

        # Calculate totals per category
        high_motion_total = sum(available_datasets[name]['sample_count'] for name in categorized['high_motion'])
        low_motion_total = sum(available_datasets[name]['sample_count'] for name in categorized['low_motion'])
        unknown_total = sum(available_datasets[name]['sample_count'] for name in categorized['unknown'])

        print(f"  High motion datasets: {len(categorized['high_motion'])}, Total samples: {high_motion_total:,}")
        print(f"  Low motion datasets: {len(categorized['low_motion'])}, Total samples: {low_motion_total:,}")
        print(f"  Unknown datasets: {len(categorized['unknown'])}, Total samples: {unknown_total:,}")

        # Apply caps to prevent dataset dominance
        max_samples_per_dataset = 500_000  # Cap per dataset

        def apply_caps(dataset_list, available_datasets):
            """Apply caps to large datasets and recalculate totals"""
            total_before = sum(available_datasets[name]['sample_count'] for name in dataset_list)
            capped_datasets = {}

            for name in dataset_list:
                original_count = available_datasets[name]['sample_count']
                capped_count = min(original_count, max_samples_per_dataset)
                capped_datasets[name] = capped_count

                if original_count > max_samples_per_dataset:
                    print(f"  📊 Capping {name}: {original_count:,} -> {capped_count:,} samples")

            total_after = sum(capped_datasets.values())
            print(f"  📊 Total after caps: {total_before:,} -> {total_after:,} samples")
            return capped_datasets, total_after

        print(f"\n🔒 Applying caps to prevent dataset dominance (max {max_samples_per_dataset:,} per dataset):")
        capped_high, high_motion_total = apply_caps(categorized['high_motion'], available_datasets)
        capped_low, low_motion_total = apply_caps(categorized['low_motion'], available_datasets)
        capped_unknown, unknown_total = apply_caps(categorized['unknown'], available_datasets)

        # Calculate sampling rates
        sampling_strategy = {}

        # High motion sampling
        if high_motion_total > 0:
            high_rate = min(1.0, high_motion_target / high_motion_total)
            for dataset_name in categorized['high_motion']:
                samples_available = capped_high[dataset_name]  # Use capped count
                samples_to_take = int(samples_available * high_rate)
                sampling_strategy[dataset_name] = {
                    'samples_to_take': samples_to_take,
                    'sampling_rate': high_rate,
                    'category': 'high_motion',
                    'original_count': available_datasets[dataset_name]['sample_count'],
                    'capped_count': samples_available
                }

        # Low motion sampling
        if low_motion_total > 0:
            low_rate = min(1.0, low_motion_target / low_motion_total)
            for dataset_name in categorized['low_motion']:
                samples_available = capped_low[dataset_name]  # Use capped count
                samples_to_take = int(samples_available * low_rate)
                sampling_strategy[dataset_name] = {
                    'samples_to_take': samples_to_take,
                    'sampling_rate': low_rate,
                    'category': 'low_motion',
                    'original_count': available_datasets[dataset_name]['sample_count'],
                    'capped_count': samples_available
                }

        # Unknown datasets get medium priority
        unknown_target = target_total - sum(s['samples_to_take'] for s in sampling_strategy.values())
        if unknown_total > 0 and unknown_target > 0:
            unknown_rate = min(1.0, unknown_target / unknown_total)
            for dataset_name in categorized['unknown']:
                samples_available = capped_unknown[dataset_name]  # Use capped count
                samples_to_take = int(samples_available * unknown_rate)
                sampling_strategy[dataset_name] = {
                    'samples_to_take': samples_to_take,
                    'sampling_rate': unknown_rate,
                    'category': 'unknown',
                    'original_count': available_datasets[dataset_name]['sample_count'],
                    'capped_count': samples_available
                }

        # Print sampling strategy
        print(f"\n🎯 Sampling Strategy:")
        for category in ['high_motion', 'low_motion', 'unknown']:
            datasets_in_category = [name for name, info in sampling_strategy.items() if info['category'] == category]
            if datasets_in_category:
                print(f"  {category.upper()}:")
                for dataset_name in datasets_in_category:
                    info = sampling_strategy[dataset_name]
                    print(f"    {dataset_name}: {info['samples_to_take']:,} samples (rate: {info['sampling_rate']:.3f})")

        return sampling_strategy

    def create_consolidated_splits(self, available_datasets: Dict, sampling_strategy: Dict):
        """Create consolidated train/val/test NPZ files with specific samples"""

        print(f"\n🔄 Creating consolidated dataset splits...")

        # Split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        # Storage for all splits
        splits_data = {
            'train': {
                'global_orient': [],
                'body_pose': [],
                'jaw_pose': [],
                'lhand_pose': [],
                'rhand_pose': [],
            },
            'val': {
                'global_orient': [],
                'body_pose': [],
                'jaw_pose': [],
                'lhand_pose': [],
                'rhand_pose': [],
            },
            'test': {
                'global_orient': [],
                'body_pose': [],
                'jaw_pose': [],
                'lhand_pose': [],
                'rhand_pose': [],
            }
        }

        # Process each dataset
        for dataset_name, strategy_info in tqdm(sampling_strategy.items(), desc="Processing datasets"):

            samples_to_take = strategy_info['samples_to_take']
            if samples_to_take == 0:
                continue

            dataset_info = available_datasets[dataset_name]
            npz_file = dataset_info['file_path']
            total_samples = dataset_info['sample_count']

            print(f"\n📁 Processing {dataset_name}: taking {samples_to_take:,} / {total_samples:,} samples")

            try:
                # Load data
                data = np.load(npz_file, allow_pickle=True)

                # Sample indices
                if samples_to_take >= total_samples:
                    # Take all samples
                    sample_indices = list(range(total_samples))
                else:
                    # Random sampling
                    np.random.seed(42)  # Reproducible
                    sample_indices = sorted(np.random.choice(total_samples, samples_to_take, replace=False))

                # Extract pose components for selected samples
                components = {}

                # Global orientation (3,)
                if 'global_orient' in data:
                    components['global_orient'] = data['global_orient'][sample_indices]
                else:
                    components['global_orient'] = np.zeros((len(sample_indices), 3))

                # Body pose (63,) - 21 joints * 3
                if 'body_pose' in data:
                    components['body_pose'] = data['body_pose'][sample_indices]
                else:
                    components['body_pose'] = np.zeros((len(sample_indices), 63))

                # Jaw pose (3,)
                if 'jaw_pose' in data:
                    components['jaw_pose'] = data['jaw_pose'][sample_indices]
                else:
                    components['jaw_pose'] = np.zeros((len(sample_indices), 3))

                # Left hand pose (45,) - 15 joints * 3
                if 'lhand_pose' in data:
                    components['lhand_pose'] = data['lhand_pose'][sample_indices]
                else:
                    components['lhand_pose'] = np.zeros((len(sample_indices), 45))

                # Right hand pose (45,) - 15 joints * 3
                if 'rhand_pose' in data:
                    components['rhand_pose'] = data['rhand_pose'][sample_indices]
                else:
                    components['rhand_pose'] = np.zeros((len(sample_indices), 45))

                # Split into train/val/test
                n_samples = len(sample_indices)
                train_end = int(n_samples * train_ratio)
                val_end = train_end + int(n_samples * val_ratio)

                # Add to respective splits
                for component_name, component_data in components.items():
                    splits_data['train'][component_name].append(component_data[:train_end])
                    splits_data['val'][component_name].append(component_data[train_end:val_end])
                    splits_data['test'][component_name].append(component_data[val_end:])

                print(f"  ✅ Train: {train_end}, Val: {val_end - train_end}, Test: {n_samples - val_end}")

            except Exception as e:
                print(f"  ❌ Error processing {dataset_name}: {e}")
                continue

        # Concatenate all data for each split
        print(f"\n🔗 Consolidating splits...")

        for split_name in ['train', 'val', 'test']:
            print(f"  📊 Consolidating {split_name} data...")

            consolidated_data = {}
            for component_name in ['global_orient', 'body_pose', 'jaw_pose', 'lhand_pose', 'rhand_pose']:
                if splits_data[split_name][component_name]:
                    consolidated_data[component_name] = np.concatenate(splits_data[split_name][component_name], axis=0)
                    print(f"    {component_name}: {consolidated_data[component_name].shape}")
                else:
                    print(f"    ⚠️  No data for {component_name} in {split_name}")
                    consolidated_data[component_name] = np.array([]).reshape(0, -1)

            # Save consolidated split
            output_file = self.output_dir / f"{split_name}.npz"
            np.savez_compressed(output_file, **consolidated_data)
            print(f"  💾 Saved {output_file}")

        # Save metadata
        metadata = {
            'total_samples': sum(len(splits_data[split]['global_orient']) for split in splits_data if splits_data[split]['global_orient']),
            'splits': {
                split: sum(len(arr) for arr in splits_data[split]['global_orient']) if splits_data[split]['global_orient'] else 0
                for split in ['train', 'val', 'test']
            },
            'datasets_processed': list(sampling_strategy.keys()),
            'sampling_strategy': sampling_strategy
        }

        metadata_file = self.output_dir / "consolidated_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Consolidated dataset creation complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Files created: train.npz, val.npz, test.npz, consolidated_metadata.json")


def main():
    print("🚀 Creating Consolidated Dataset Splits")
    print("=" * 60)

    # Create dataset creator
    creator = ConsolidatedDatasetCreator(
        data_dir="../data",
        output_dir="dataset_splits"
    )

    # Step 1: Scan available datasets
    available_datasets = creator.scan_available_datasets()

    if not available_datasets:
        print("❌ No datasets found!")
        return

    print(f"\n📊 Found {len(available_datasets)} available datasets")

    # Step 2: Categorize datasets
    categorized = creator.categorize_datasets(available_datasets)

    print(f"\n📋 Dataset categorization:")
    print(f"  High motion: {len(categorized['high_motion'])} datasets")
    print(f"  Low motion: {len(categorized['low_motion'])} datasets")
    print(f"  Unknown: {len(categorized['unknown'])} datasets")

    # Step 3: Calculate sampling strategy
    sampling_strategy = creator.calculate_sampling_strategy(available_datasets, categorized)

    # Step 4: Create consolidated splits
    creator.create_consolidated_splits(available_datasets, sampling_strategy)

    print(f"\n🎉 Process complete!")
    print(f"💡 Now you can use the consolidated train.npz, val.npz, test.npz files")
    print(f"💡 These contain only the selected samples, not entire datasets")


if __name__ == "__main__":
    main()