# 🚀 SMPL-X Pose Priors with PyTorch Lightning

This repository implements conditional normalizing flows and probabilistic models for SMPL-X pose priors, designed for human pose estimation and generation tasks.

## 📋 Models Available

| Model | Purpose | Parameters | Training Time |
|-------|---------|------------|---------------|
| **Hand VAE** | Hand pose compression/generation | ~2M | ~1-2 hours |
| **Gaussian Prior** | Simple baseline pose prior | 330 | ~5 minutes |
| **Joint Flow** | Per-joint conditional flows | 517K | ~30-60 minutes |
| **Global Flow** | Full pose conditional flows | 343K | ~45-90 minutes |

## 🔧 Prerequisites

### Environment Setup
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individual packages:
pip install torch pytorch-lightning tensorboard numpy matplotlib seaborn tqdm wandb
```

### Data Setup
Your NPZ files should be in the project directory:
```
/Users/kenielpeart/Downloads/hand_prior/code/
├── train_poses.npz      # Training poses
├── val_poses.npz        # Validation poses
├── test_poses.npz       # Test poses (optional)
└── other_data.npz       # Any other pose NPZ files
```

## 🏃‍♂️ Running the Models

### 1. Hand VAE (Hand Pose Compression)
```bash
# Basic training (default settings)
python train_hand_vae_lightning.py

# With custom arguments
python train_hand_vae_lightning.py \
    --model_type standard \
    --z_dim 24 \
    --hidden 256 \
    --learning_rate 3e-4 \
    --batch_size 8192 \
    --max_epochs 100

# SO3 variant for rotation-aware training
python train_hand_vae_lightning.py \
    --model_type so3 \
    --z_dim 32 \
    --hidden 512
```

**Available Arguments:**
- `--model_type`: Choose 'standard' or 'so3' variant (default: standard)
- `--z_dim`: Latent dimension (default: 24)
- `--hidden`: Hidden layer size (default: 256)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--batch_size`: Batch size (default: 8192)
- `--max_epochs`: Training epochs (default: 100)
- `--data_dir`: Data directory (default: ../data)
- `--logger`: Use 'tensorboard' or 'wandb' (default: tensorboard)

**Features:**
- Compresses hand poses to latent space
- Trains on actual SMPL-X hand data
- Saves checkpoints to `checkpoints/hand_vae/`
- TensorBoard logs: `lightning_logs/hand_vae/`

### 2. Gaussian Prior (Baseline Model)
```bash
# Basic training (default settings)
python train_gaussian_real_data.py

# With custom arguments
python train_gaussian_real_data.py \
    --lr 1e-2 \
    --batch_size 32 \
    --max_epochs 20 \
    --scheduler_type step

# Different configuration
python train_gaussian_real_data.py \
    --lr 5e-3 \
    --batch_size 64 \
    --max_epochs 50 \
    --scheduler_type cosine \
    --patience 15
```

**Available Arguments:**
- `--data_dir`: Data directory (default: current directory)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-2)
- `--scheduler_type`: 'step' or 'cosine' (default: step)
- `--max_epochs`: Training epochs (default: 20)
- `--patience`: Early stopping patience (default: 10)
- `--num_workers`: Data loading workers (default: 0)
- `--accelerator`: Hardware accelerator (default: auto)

**Features:**
- Fastest training (~5-10 minutes)
- Fits Gaussian distributions per joint
- Good baseline for comparisons
- Handles 53→55 joint conversion automatically

### 3. Joint Flow (Per-Joint Conditional Flows)
```bash
# Basic training (default settings)
python train_joint_flow_real_data.py

# With custom arguments
python train_joint_flow_real_data.py \
    --hidden 128 \
    --K 4 \
    --lr 1e-3 \
    --batch_size 16 \
    --max_epochs 50

# Larger model configuration
python train_joint_flow_real_data.py \
    --hidden 256 \
    --K 6 \
    --lr 5e-4 \
    --batch_size 32 \
    --max_epochs 100 \
    --scheduler_type plateau
```

**Available Arguments:**
- `--data_dir`: Data directory (default: current directory)
- `--batch_size`: Batch size (default: 16)
- `--hidden`: Hidden layer size (default: 128)
- `--K`: Number of coupling layers (default: 4)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-5)
- `--grad_clip_val`: Gradient clipping (default: 1.0)
- `--scheduler_type`: 'cosine' or 'plateau' (default: cosine)
- `--max_epochs`: Training epochs (default: 50)
- `--patience`: Early stopping patience (default: 15)

**Features:**
- Each joint conditioned on parent joint
- Respects SMPL-X kinematic constraints
- More sophisticated than Gaussian
- ~30-60 minutes training time

### 4. Global Flow (Full Pose Flows)
```bash
# Basic training (default settings)
python train_global_flow_real_data.py

# With custom arguments
python train_global_flow_real_data.py \
    --hidden 512 \
    --K 6 \
    --lr 1e-3 \
    --batch_size 32 \
    --max_epochs 30 \
    --use_actnorm

# Larger model with conditioning
python train_global_flow_real_data.py \
    --hidden 768 \
    --K 8 \
    --lr 5e-4 \
    --batch_size 64 \
    --max_epochs 50 \
    --use_actnorm \
    --use_conditioning \
    --scheduler_type plateau
```

**Available Arguments:**
- `--data_dir`: Data directory (default: current directory)
- `--batch_size`: Batch size (default: 32)
- `--hidden`: Hidden layer size (default: 512)
- `--K`: Number of coupling layers (default: 6)
- `--lr`: Learning rate (default: 1e-3)
- `--use_actnorm`: Use ActNorm layers (default: True)
- `--scheduler_type`: 'cosine' or 'plateau' (default: cosine)
- `--use_conditioning`: Enable experimental conditioning (default: False)
- `--max_epochs`: Training epochs (default: 30)
- `--patience`: Early stopping patience (default: 10)
- `--gradient_clip_val`: Gradient clipping (default: 1.0)

**Features:**
- Models entire pose as high-dimensional distribution
- Uses ActNorm for training stability
- Best quality but longest training time
- Captures global pose correlations

## 📊 Monitoring Training

### TensorBoard
```bash
# In another terminal, run:
tensorboard --logdir lightning_logs/

# Then open: http://localhost:6006
```

### Key Metrics to Watch
- **NLL (Negative Log-Likelihood)**: Lower is better
- **BPD (Bits Per Dimension)**: Normalized likelihood
- **Round-trip MSE**: Reconstruction error
- **Sample statistics**: Generated sample quality

## 🔧 Joint Format Handling

The models automatically handle joint count conversion:
- **Input data**: 53 joints (your SMPL-X data format)
- **Model format**: 55 joints (with eye joints as zeros)

Conversion happens automatically in `comprehensive_pose_datamodule.py`:
```python
# Your 53 joints: [global_orient + body_pose + jaw + hands]
# Model 55 joints: [global_orient + body_pose + jaw + EYES + hands]
poses_55x3[:, :23, :] = poses_53x3[:, :23, :]  # Copy first 23
# Joints 23, 24 (eyes) stay as zeros
poses_55x3[:, 25:, :] = poses_53x3[:, 23:, :]  # Copy remaining
```

## 🐛 Troubleshooting

### "No .npz files found!"
```bash
# Make sure your data files are in the project directory:
ls *.npz
```

### "CUDA out of memory"
```bash
# Reduce batch size in training scripts:
# Edit the trainer file and change batch_size=16 (or smaller)
```

### "Training too slow"
```bash
# Increase num_workers in the training script:
# Set num_workers=4 (or more, up to your CPU cores)
```

### "NLL not decreasing"
```bash
# Try reducing learning rate or increasing gradient clipping
# Edit lr=1e-4 (instead of 1e-3) in the trainer
```

## 📈 Expected Results

### Training Times & Performance
- **Hand VAE**: ~1-2 hours, generates realistic hand poses
- **Gaussian**: ~5-10 minutes, NLL ~50-80 (baseline)
- **Joint Flow**: ~30-60 minutes, NLL ~30-50 (better than Gaussian)
- **Global Flow**: ~45-90 minutes, NLL ~25-45 (best results)

### Output Structure
```
checkpoints/
├── hand_vae/                 # Hand VAE checkpoints
├── gaussian-real-data/       # Gaussian model checkpoints
├── joint-flow-real-data/     # Joint flow checkpoints
└── global-flow-real-data/    # Global flow checkpoints

lightning_logs/
├── hand_vae/                 # Hand VAE training logs
├── gaussian_real_data/       # Gaussian training logs
├── joint_flow_real_data/     # Joint flow logs
└── global_flow_real_data/    # Global flow logs
```

## 🤔 Which Model Should I Use?

- **Hand VAE**: For hand pose compression and dimensionality reduction
- **Gaussian**: Fast baseline, simple applications, quick experiments
- **Joint Flow**: Best balance of quality vs speed, respects kinematics
- **Global Flow**: Highest quality, research applications, global correlations

## 🎯 Getting Started (Recommended Order)

1. **Start with Gaussian** - Quick baseline and data validation:
   ```bash
   python train_gaussian_real_data.py
   ```

2. **Try Joint Flow** - Better results with kinematic constraints:
   ```bash
   python train_joint_flow_real_data.py
   ```

3. **Use Global Flow** - Best quality for research:
   ```bash
   python train_global_flow_real_data.py
   ```

4. **Hand VAE** - For hand-specific compression:
   ```bash
   python train_hand_vae_lightning.py
   ```

## 📁 Model Files

### Core Models
- `joint_limit_gaussian.py` - Gaussian joint prior
- `joint_limit_flow.py` - Conditional normalizing flows per joint
- `global_conditional_flow.py` - Full pose conditional flows
- `hand_vae_prior.py` - Hand pose VAE
- `hand_vae_prior_so3.py` - SO3 hand VAE variant

### Data Modules
- `comprehensive_pose_datamodule.py` - Main SMPL-X pose data loader
- `hand_prior_datamodule.py` - Hand-specific data module
- `hand_vae_datamodule.py` - VAE-optimized data module

### Utilities
- `rotation_utils.py` - Rotation processing and 6D representation utilities

## 🎉 Next Steps

After training, you can:
1. **Load trained models** for inference
2. **Generate new poses** using the learned priors
3. **Use as regularizers** in other pose estimation tasks
4. **Compare different architectures** for your specific use case

## 📚 Additional Resources

- See `TRAINING_GUIDE.md` for detailed training instructions
- Check `lightning_logs/` for TensorBoard visualizations
- Explore `checkpoints/` for saved model weights