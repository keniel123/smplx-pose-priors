# Hand Prior Dataset Generator

This repository contains scripts to create a hand-focused motion dataset from multiple motion capture sources, optimized for training hand pose models.

## Overview

The dataset generator creates a balanced 1.5M sample dataset with emphasis on hand-relevant motions:

- **60% High Priority**: Sign language, dance, interaction, and synthetic data
- **25% Medium Priority**: Expressive activities (speech, interviews, etc.)
- **15% Low Priority**: Traditional motion capture data

**Train/Val/Test Split**: 1.2M/150K/150K samples

## Files

- `create_datamodule.py`: Main script to generate dataset splits
- `hand_prior_datamodule.py`: Multi-file PyTorch Lightning DataModule
- `hand_vae_datamodule.py`: VAE-optimized DataModule with 90D output
- `create_consolidated_npz.py`: Create single NPZ files for efficiency
- `test_consolidated_vae.py`: Consolidated DataModule implementation
- `README.md`: This file

## Quick Start

### 1. Generate Dataset Splits

```bash
cd code
python create_datamodule.py
```

This will:
- Scan all NPZ files in `../data/`
- Create balanced samples based on hand relevance
- Generate train/val/test splits
- Save results to `dataset_splits/` directory

### 2. Use with PyTorch Lightning

#### For VAE Training (Recommended):
```python
from hand_vae_datamodule import HandVAEDataModule

# VAE-optimized datamodule with 90D output
dm = HandVAEDataModule(
    data_dir="../data",
    splits_dir="dataset_splits",
    batch_size=8192,        # Large batches for VAE
    return_dict=False,      # Just (B, 90) tensors
    standardize=True,       # Auto-standardize with train stats
    mmap=True              # Memory-mapped NPZ loading
)

# Setup and use
dm.prepare_data()
dm.setup()

# Get standardized 90D batches directly
train_loader = dm.train_dataloader()
batch = next(iter(train_loader))  # Shape: (8192, 90)

trainer = pl.Trainer(max_epochs=100)
trainer.fit(vae_model, datamodule=dm)
```

#### For Research/Flexibility:
```python
from hand_prior_datamodule import HandPriorDataModule

# Multi-dataset datamodule with metadata
dm = HandPriorDataModule(
    data_dir="../data",
    splits_dir="dataset_splits",
    batch_size=32,
    num_workers=4,
)

dm.prepare_data()
dm.setup()

trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, datamodule=dm)
```

#### For Maximum Efficiency (Consolidated NPZ):
```python
# First, create consolidated NPZ files
python create_consolidated_npz.py

# Then use the optimized loader
from test_consolidated_vae import ConsolidatedHandDataModule

dm = ConsolidatedHandDataModule(
    consolidated_dir="consolidated_splits",
    batch_size=8192,
    return_dict=False  # Direct (B, 90) tensors
)
```

## Dataset Structure

## DataModule Options

### 1. Multi-File DataModule (`hand_prior_datamodule.py`)
For research and flexibility:
```python
{
    'lhand_pose': torch.Tensor,   # Shape: (45,) - Left hand parameters
    'rhand_pose': torch.Tensor,   # Shape: (45,) - Right hand parameters
    'dataset_name': str,          # Source dataset name
    'frame_idx': int,             # Frame index in original dataset
    'image_id': str               # Optional: Frame identifier
}
```

### 2. VAE-Optimized DataModule (`hand_vae_datamodule.py`)
For efficient VAE training with 90D concatenated poses:

**Tensor mode** (`return_dict=False`):
```python
torch.Tensor  # Shape: (90,) - Concatenated [lhand + rhand], standardized
```

**Dict mode** (`return_dict=True`):
```python
{
    'x90': torch.Tensor,          # Shape: (90,) - Standardized concat poses
    'lhand_pose': torch.Tensor,   # Shape: (45,) - Raw left hand
    'rhand_pose': torch.Tensor,   # Shape: (45,) - Raw right hand
    'mean': torch.Tensor,         # Dataset normalization mean
    'std': torch.Tensor,          # Dataset normalization std
    'dataset_name': str,          # Source dataset name
    'frame_idx': int,             # Frame index
}
```

## Dataset Categories

### High Priority (60% of samples)
**Sign Language:**
- SignAvatarsLang-001 (5.0M frames)
- SignAvatarsWord, SignAvatarsHam, signlanguage

**Dance/Performance:**
- DanceDB (1.5M frames)
- EyesJapan (2.9M frames)
- MotionXmusic, MotionXperform
- conductmusic, entertainment, singing

**Interaction:**
- GRAB (1.6M frames)
- MOYO, TCDHands, ham, arctic

**Synthetic:**
- Bedlam (3.2M frames)
- HUMAN4D, idea400, humman

### Medium Priority (25% of samples)
Expressive activities: magic, speech, talkshow, tvshow, interview, videoconference, online_class, livevlog, kungfu, fitness, movie, olympic, mscoco

### Low Priority (15% of samples)
Traditional motion capture: CMU, KIT, HDM05, ACCAD, SFU, TotalCapture, BMLrub, BMLmovi, WEIZMANN, EKUT, etc.

## Configuration

### Sampling Strategy
- **Target Total**: 1.5M samples
- **High Priority Sampling Rate**: ~4.5% of available frames
- **Medium Priority Sampling Rate**: ~24.7% of available frames
- **Low Priority Sampling Rate**: ~1.2% of available frames

### Split Configuration
```python
# In create_datamodule.py
target_samples = {
    'high': 900_000,    # 60%
    'medium': 375_000,  # 25%
    'low': 225_000      # 15%
}

split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
```

## Output Files

After running `create_datamodule.py`:

```
dataset_splits/
├── train_samples.json      # Training sample indices
├── val_samples.json        # Validation sample indices
├── test_samples.json       # Test sample indices
└── dataset_metadata.json   # Dataset creation metadata
```

## Advanced Usage

### Custom Transforms

```python
class CustomTransform:
    def __call__(self, sample):
        # Your custom preprocessing
        sample['lhand_pose'] = some_processing(sample['lhand_pose'])
        sample['rhand_pose'] = some_processing(sample['rhand_pose'])
        return sample

dm = HandPriorDataModule(transform=CustomTransform())
```

### Memory Management

For large datasets, consider:
- Reducing `num_workers` if running out of memory
- Using smaller `batch_size`
- The dataset caches loaded NPZ files - monitor memory usage

### Modifying Sampling Strategy

Edit the priority lists in `create_datamodule.py`:

```python
# Add/remove datasets from priority categories
self.high_priority = ['your_dataset', ...]
self.medium_priority = [...]
self.low_priority = [...]

# Adjust target sample distribution
self.target_samples = {
    'high': 1_050_000,  # 70%
    'medium': 300_000,  # 20%
    'low': 150_000      # 10%
}
```

## Requirements

```bash
pip install torch pytorch-lightning numpy tqdm
```

## Notes

- Set `RANDOM_SEED` for reproducible splits
- The script automatically handles datasets with different frame counts
- Missing datasets in priority lists are reported as warnings
- Hand pose parameters are in the original SMPL/SMPL-X format (45 parameters per hand)