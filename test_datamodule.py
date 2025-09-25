#!/usr/bin/env python3
"""
Test version of dataset creation - creates small dataset for testing
"""

import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm


class TestHandPriorDatasetCreator:
    def __init__(self, data_dir: str, output_dir: str = "test_dataset_splits"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Same categorization as main script
        self.high_priority = [
            'SignAvatarsLang-001', 'SignAvatarsWord', 'SignAvatarsHam', 'signlanguage',
            'DanceDB', 'EyesJapan', 'MotionXmusic', 'MotionXperform',
            'conductmusic', 'entertainment', 'singing',
            'GRAB', 'MOYO', 'TCDHands', 'ham', 'arctic',
            'Bedlam', 'HUMAN4D', 'idea400', 'humman'
        ]

        self.medium_priority = [
            'magic', 'speech', 'talkshow', 'tvshow', 'interview',
            'videoconference', 'online_class', 'livevlog', 'kungfu',
            'fitness', 'movie', 'olympic', 'mscoco'
        ]

        self.low_priority = [
            'CMU', 'KIT', 'HDM05', 'ACCAD', 'SFU', 'TotalCapture',
            'BMLrub', 'BMLmovi', 'WEIZMANN', 'EKUT', 'Mosh',
            'HumanEva', 'PosePrior', 'SOMA', 'Transitions',
            'CNRS', 'MotionXanimation', 'MotionXhaa500', 'DFaust', 'SSM'
        ]

        # Small test targets (1000 samples total)
        self.target_samples = {
            'high': 600,    # 60%
            'medium': 250,  # 25%
            'low': 150      # 15%
        }

        self.split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

    def load_dataset_info(self) -> Dict[str, int]:
        """Load frame counts for first few datasets only"""
        dataset_info = {}

        print("🔍 Scanning NPZ files (test mode - limited)...")

        # Only load first few files for testing
        npz_files = sorted(list(self.data_dir.glob("*.npz")))[:10]

        for npz_file in tqdm(npz_files):
            try:
                data = np.load(npz_file, allow_pickle=True)
                if 'body_pose' in data:
                    frame_count = len(data['body_pose'])
                    dataset_name = npz_file.stem
                    dataset_info[dataset_name] = frame_count
                    print(f"  {dataset_name}: {frame_count:,} frames")
                else:
                    print(f"  ⚠️  {npz_file.name}: No body_pose key found")
            except Exception as e:
                print(f"  ❌ {npz_file.name}: Error - {e}")

        return dataset_info

    def calculate_sampling_rates(self, dataset_info: Dict[str, int]) -> Dict[str, Dict]:
        """Calculate sampling rates for test dataset"""

        priority_totals = {'high': 0, 'medium': 0, 'low': 0}
        priority_datasets = {'high': {}, 'medium': {}, 'low': {}}

        for dataset, frames in dataset_info.items():
            if dataset in self.high_priority:
                priority_totals['high'] += frames
                priority_datasets['high'][dataset] = frames
            elif dataset in self.medium_priority:
                priority_totals['medium'] += frames
                priority_datasets['medium'][dataset] = frames
            elif dataset in self.low_priority:
                priority_totals['low'] += frames
                priority_datasets['low'][dataset] = frames

        sampling_rates = {}
        for priority in ['high', 'medium', 'low']:
            total_available = priority_totals[priority]
            target_samples = self.target_samples[priority]

            if total_available > 0:
                rate = min(1.0, target_samples / total_available)  # Cap at 100%
                sampling_rates[priority] = {
                    'rate': rate,
                    'datasets': priority_datasets[priority],
                    'total_available': total_available,
                    'target_samples': min(target_samples, total_available)
                }
            else:
                sampling_rates[priority] = {
                    'rate': 0.0,
                    'datasets': {},
                    'total_available': 0,
                    'target_samples': 0
                }

        return sampling_rates

    def create_sample_indices(self, dataset_info: Dict[str, int],
                            sampling_rates: Dict[str, Dict]) -> Dict[str, List[Tuple[str, int]]]:
        """Create sample indices for test dataset"""

        all_samples = {'high': [], 'medium': [], 'low': []}

        print("\\n📊 Creating sample indices (test mode)...")

        for priority in ['high', 'medium', 'low']:
            rate_info = sampling_rates[priority]
            sampling_rate = rate_info['rate']

            print(f"\\n{priority.upper()} Priority (sampling rate: {sampling_rate:.4f}):")

            for dataset, total_frames in rate_info['datasets'].items():
                n_samples = min(int(total_frames * sampling_rate), total_frames)

                if n_samples > 0:
                    # Use smaller sample size for testing
                    max_indices = min(total_frames, n_samples)
                    indices = random.sample(range(total_frames), max_indices)
                    samples = [(dataset, idx) for idx in indices]
                    all_samples[priority].extend(samples)
                    print(f"  {dataset}: {n_samples:,} samples from {total_frames:,} frames")

        # Shuffle and create splits
        splits = {'train': [], 'val': [], 'test': []}

        for priority in ['high', 'medium', 'low']:
            samples = all_samples[priority]
            random.shuffle(samples)
            n_samples = len(samples)

            n_train = int(n_samples * self.split_ratios['train'])
            n_val = int(n_samples * self.split_ratios['val'])
            n_test = n_samples - n_train - n_val

            splits['train'].extend(samples[:n_train])
            splits['val'].extend(samples[n_train:n_train + n_val])
            splits['test'].extend(samples[n_train + n_val:])

            print(f"  {priority.upper()} split: Train={n_train}, Val={n_val}, Test={n_test}")

        for split in splits:
            random.shuffle(splits[split])

        return splits

    def save_splits(self, splits: Dict[str, List[Tuple[str, int]]], sampling_info: Dict) -> None:
        """Save test dataset splits"""

        print("\\n💾 Saving test dataset splits...")

        for split_name, samples in splits.items():
            split_file = self.output_dir / f"{split_name}_samples.json"

            samples_dict = {}
            for dataset, idx in samples:
                if dataset not in samples_dict:
                    samples_dict[dataset] = []
                samples_dict[dataset].append(idx)

            with open(split_file, 'w') as f:
                json.dump(samples_dict, f, indent=2)

            print(f"  {split_name}: {len(samples):,} samples -> {split_file}")

        # Save metadata
        metadata = {
            'test_dataset_info': {
                'total_samples': sum(len(samples) for samples in splits.values()),
                'splits': {name: len(samples) for name, samples in splits.items()},
                'priority_targets': self.target_samples,
                'split_ratios': self.split_ratios,
            },
            'sampling_info': sampling_info
        }

        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Metadata -> {metadata_file}")

    def create_test_dataset(self, seed: int = 42) -> None:
        """Create small test dataset"""

        print("🧪 Creating Test Hand-Focused Dataset (1K samples)")
        print("=" * 50)

        random.seed(seed)
        np.random.seed(seed)

        dataset_info = self.load_dataset_info()
        sampling_rates = self.calculate_sampling_rates(dataset_info)

        print("\\n📈 Test Sampling Summary:")
        for priority in ['high', 'medium', 'low']:
            info = sampling_rates[priority]
            print(f"  {priority.upper()}: {info['target_samples']:,} target samples "
                  f"from {info['total_available']:,} available")

        splits = self.create_sample_indices(dataset_info, sampling_rates)
        self.save_splits(splits, sampling_rates)

        print("\\n✅ Test dataset creation complete!")
        print(f"📁 Output: {self.output_dir}")
        print("\\n📊 Final test split sizes:")
        for split_name, samples in splits.items():
            print(f"  {split_name}: {len(samples):,} samples")


if __name__ == "__main__":
    # Test configuration
    DATA_DIR = "../data"
    OUTPUT_DIR = "test_dataset_splits"
    RANDOM_SEED = 42

    creator = TestHandPriorDatasetCreator(DATA_DIR, OUTPUT_DIR)
    creator.create_test_dataset(RANDOM_SEED)