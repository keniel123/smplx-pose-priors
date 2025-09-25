#!/usr/bin/env python3
"""
Hand-Focused Dataset Creation Script

Creates a balanced 1.5M sample dataset prioritizing hand-relevant data:
- Sign language, dance, interaction, and synthetic data (60%)
- Expressive activities (25%)
- Traditional motion capture (15%)

Train/Val/Test split: 80/10/10 (1.2M/150K/150K)
"""

import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm


class HandPriorDatasetCreator:
    def __init__(self, data_dir: str, output_dir: str = "dataset_splits"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Dataset categorization by hand relevance
        self.high_priority = [
            # Sign Language (highest hand activity)
            "SignAvatarsLang-001",
            "SignAvatarsWord",
            "SignAvatarsHam",
            "signlanguage",
            # Dance/Performance (expressive hand movements)
            "DanceDB",
            "EyesJapan",
            "MotionXmusic",
            "MotionXperform",
            "conductmusic",
            "entertainment",
            "singing",
            # Interaction (hand-object/hand-hand)
            "GRAB",
            "MOYO",
            "TCDHands",
            "ham",
            "arctic",
            # Synthetic (likely includes detailed hand data)
            "Bedlam",
            "HUMAN4D",
            "idea400",
            "humman",
        ]

        self.medium_priority = [
            # Expressive activities
            "magic",
            "speech",
            "talkshow",
            "tvshow",
            "interview",
            "videoconference",
            "online_class",
            "livevlog",
            "kungfu",
            "fitness",
            "movie",
            "olympic",
            "mscoco",
        ]

        self.low_priority = [
            # Traditional motion capture (less hand focus)
            "CMU",
            "KIT",
            "HDM05",
            "ACCAD",
            "SFU",
            "TotalCapture",
            "BMLrub",
            "BMLmovi",
            "WEIZMANN",
            "EKUT",
            "Mosh",
            "HumanEva",
            "PosePrior",
            "SOMA",
            "Transitions",
            "CNRS",
            "MotionXanimation",
            "MotionXhaa500",
            "DFaust",
            "SSM",
        ]

        # Target samples
        self.target_samples = {
            "high": 900_000,  # 60%
            "medium": 375_000,  # 25%
            "low": 225_000,  # 15%
        }

        # Train/val/test split ratios
        self.split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    def load_dataset_info(self) -> Dict[str, int]:
        """Load frame counts for all datasets"""
        dataset_info = {}

        print("🔍 Scanning NPZ files...")
        for npz_file in tqdm(sorted(self.data_dir.glob("*.npz"))):
            try:
                data = np.load(npz_file, allow_pickle=True)
                if "body_pose" in data:
                    frame_count = len(data["body_pose"])
                    dataset_name = npz_file.stem
                    dataset_info[dataset_name] = frame_count
                    print(f"  {dataset_name}: {frame_count:,} frames")
                else:
                    print(f"  ⚠️  {npz_file.name}: No body_pose key found")
            except Exception as e:
                print(f"  ❌ {npz_file.name}: Error - {e}")

        return dataset_info

    def calculate_sampling_rates(
        self, dataset_info: Dict[str, int]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate sampling rates for each priority category"""

        # Group datasets by priority
        priority_totals = {"high": 0, "medium": 0, "low": 0}
        priority_datasets = {"high": {}, "medium": {}, "low": {}}

        for dataset, frames in dataset_info.items():
            if dataset in self.high_priority:
                priority_totals["high"] += frames
                priority_datasets["high"][dataset] = frames
            elif dataset in self.medium_priority:
                priority_totals["medium"] += frames
                priority_datasets["medium"][dataset] = frames
            elif dataset in self.low_priority:
                priority_totals["low"] += frames
                priority_datasets["low"][dataset] = frames
            else:
                print(f"⚠️  Unknown dataset: {dataset}")

        # Calculate sampling rates
        sampling_rates = {}
        for priority in ["high", "medium", "low"]:
            total_available = priority_totals[priority]
            target_samples = self.target_samples[priority]

            if total_available > 0:
                rate = target_samples / total_available
                sampling_rates[priority] = {
                    "rate": rate,
                    "datasets": priority_datasets[priority],
                    "total_available": total_available,
                    "target_samples": target_samples,
                }
            else:
                sampling_rates[priority] = {
                    "rate": 0.0,
                    "datasets": {},
                    "total_available": 0,
                    "target_samples": 0,
                }

        return sampling_rates

    def create_sample_indices(
        self, dataset_info: Dict[str, int], sampling_rates: Dict[str, Dict]
    ) -> Dict[str, List[Tuple[str, int]]]:
        """Create sample indices for train/val/test splits"""

        all_samples = {"high": [], "medium": [], "low": []}

        print("\\n📊 Creating sample indices...")

        for priority in ["high", "medium", "low"]:
            rate_info = sampling_rates[priority]
            sampling_rate = rate_info["rate"]

            print(
                f"\\n{priority.upper()} Priority (sampling rate: {sampling_rate:.4f}):"
            )

            for dataset, total_frames in rate_info["datasets"].items():
                n_samples = int(total_frames * sampling_rate)

                # Generate random indices
                if n_samples > 0:
                    indices = random.sample(range(total_frames), n_samples)
                    samples = [(dataset, idx) for idx in indices]
                    all_samples[priority].extend(samples)
                    print(
                        f"  {dataset}: {n_samples:,} samples from {total_frames:,} frames"
                    )

        # Shuffle samples within each priority
        for priority in all_samples:
            random.shuffle(all_samples[priority])

        # Create train/val/test splits
        splits = {"train": [], "val": [], "test": []}

        for priority in ["high", "medium", "low"]:
            samples = all_samples[priority]
            n_samples = len(samples)

            n_train = int(n_samples * self.split_ratios["train"])
            n_val = int(n_samples * self.split_ratios["val"])
            n_test = n_samples - n_train - n_val

            splits["train"].extend(samples[:n_train])
            splits["val"].extend(samples[n_train : n_train + n_val])
            splits["test"].extend(samples[n_train + n_val :])

            print(f"\\n{priority.upper()} Priority split:")
            print(f"  Train: {n_train:,}, Val: {n_val:,}, Test: {n_test:,}")

        # Final shuffle
        for split in splits:
            random.shuffle(splits[split])

        return splits

    def save_splits(
        self, splits: Dict[str, List[Tuple[str, int]]], sampling_info: Dict
    ) -> None:
        """Save train/val/test splits and metadata"""

        print("\\n💾 Saving dataset splits...")

        # Save sample indices
        for split_name, samples in splits.items():
            split_file = self.output_dir / f"{split_name}_samples.json"

            # Convert to serializable format
            samples_dict = {}
            for dataset, idx in samples:
                if dataset not in samples_dict:
                    samples_dict[dataset] = []
                samples_dict[dataset].append(idx)

            with open(split_file, "w") as f:
                json.dump(samples_dict, f, indent=2)

            print(f"  {split_name}: {len(samples):,} samples -> {split_file}")

        # Save metadata
        metadata = {
            "dataset_creation_info": {
                "total_samples": sum(len(samples) for samples in splits.values()),
                "splits": {name: len(samples) for name, samples in splits.items()},
                "priority_targets": self.target_samples,
                "split_ratios": self.split_ratios,
                "high_priority_datasets": self.high_priority,
                "medium_priority_datasets": self.medium_priority,
                "low_priority_datasets": self.low_priority,
            },
            "sampling_info": sampling_info,
        }

        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Metadata -> {metadata_file}")

    def create_dataset(self, seed: int = 42) -> None:
        """Main method to create the hand-focused dataset"""

        print("🤖 Creating Hand-Focused Dataset (1.5M samples)")
        print("=" * 60)

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Load dataset information
        dataset_info = self.load_dataset_info()

        # Calculate sampling rates
        sampling_rates = self.calculate_sampling_rates(dataset_info)

        # Print summary
        print("\\n📈 Sampling Summary:")
        total_available = sum(
            info["total_available"] for info in sampling_rates.values()
        )
        total_target = sum(info["target_samples"] for info in sampling_rates.values())

        for priority in ["high", "medium", "low"]:
            info = sampling_rates[priority]
            print(
                f"  {priority.upper()}: {info['target_samples']:,} samples "
                f"({info['target_samples'] / total_target * 100:.1f}%) from "
                f"{info['total_available']:,} available "
                f"(rate: {info['rate']:.4f})"
            )

        print(f"\\nTotal: {total_target:,} samples from {total_available:,} available")

        # Create sample indices and splits
        splits = self.create_sample_indices(dataset_info, sampling_rates)

        # Save results
        self.save_splits(splits, sampling_rates)

        print("\\n✅ Dataset creation complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print("\\n📊 Final split sizes:")
        for split_name, samples in splits.items():
            print(f"  {split_name}: {len(samples):,} samples")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "../data"  # Path to NPZ files
    OUTPUT_DIR = "dataset_splits"
    RANDOM_SEED = 42

    # Create dataset
    creator = HandPriorDatasetCreator(DATA_DIR, OUTPUT_DIR)
    creator.create_dataset(RANDOM_SEED)
