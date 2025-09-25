# 🚀 Complete Training Guide for SMPL-X Pose Priors

This guide shows you how to train all four models with real SMPL-X data (55-joint format with eye joints as zeros).

## 📋 Quick Summary

| Model | Command | Parameters | Training Time | Purpose |
|-------|---------|------------|---------------|---------|
| **Hand VAE** | `python train_hand_vae_lightning.py` | ~2M | ~1-2 hours | Hand pose VAE compression |
| **Gaussian Prior** | `python train_gaussian_real_data.py` | 330 | ~5 minutes | Simple baseline prior |
| **Joint Flow** | `python train_joint_flow_real_data.py` | 517K | ~30-60 minutes | Per-joint conditional flows |
| **Global Flow** | `python train_global_flow_real_data.py` | 343K | ~45-90 minutes | Full pose conditional flows |

## 🔧 Prerequisites

### Data Setup
Make sure your data files are in the correct location:
```bash
# Your NPZ files should be in the project directory:
/Users/kenielpeart/Downloads/hand_prior/code/
├── train_poses.npz      # Training poses
├── val_poses.npz        # Validation poses
├── test_poses.npz       # Test poses (optional)
└── other_data.npz       # Any other pose NPZ files
```

### Environment
```bash
pip install torch pytorch-lightning tensorboard numpy
```

## 🏃‍♂️ Running Each Model

### 1. Hand VAE (Already Connected to Real Data)
```bash
# This one is already fully set up with real data
python train_hand_vae_lightning.py
```

**Expected output:**
- Trains on your actual SMPL-X pose data
- Creates hand pose VAE with encoder/decoder
- Saves checkpoints to `checkpoints/hand_vae/`
- Logs to `lightning_logs/hand_vae/`

### 2. Gaussian Prior

**Test with dummy data:**
```bash
python gaussian_trainer.py
```

**Train with real data:**
```bash
python train_gaussian_real_data.py
```

**What it does:**
- Converts your 53-joint poses → 55-joint poses (adds eye joints as zeros)
- Fits simple Gaussian distributions per joint
- Very fast training (5-10 minutes)
- Perfect for baseline comparisons

### 3. Joint Limit Flow (Per-Joint Conditional Flows)

**Test with dummy data:**
```bash
python joint_flow_trainer.py
```

**Train with real data:**
```bash
python train_joint_flow_real_data.py
```

**What it does:**
- Uses conditional normalizing flows per joint
- Each joint conditioned on its parent joint's pose
- Respects SMPL-X kinematic constraints
- More sophisticated than Gaussian, models complex distributions

### 4. Global Flow (Full Pose Flow)

**Test with dummy data:**
```bash
python global_flow_trainer.py
```

**Train with real data:**
```bash
python train_global_flow_real_data.py
```

**What it does:**
- Models the entire pose as one high-dimensional distribution
- Uses ActNorm for training stability
- Can capture global pose correlations
- Most sophisticated model but requires more training time

## 📊 Monitoring Training

### TensorBoard
```bash
# In another terminal, run:
tensorboard --logdir lightning_logs/

# Then open: http://localhost:6006
```

### Key Metrics to Watch
- **NLL (Negative Log-Likelihood)**: Lower is better, primary training objective
- **BPD (Bits Per Dimension)**: Normalized likelihood, easier to compare across models
- **Round-trip MSE**: Reconstruction error (should approach 0)
- **Sample statistics**: Check if generated samples are realistic

## 🔧 Joint Count Handling

**Important:** Your data has 53 joints, but the pose prior models expect 55 joints (with eye joints). The real data trainers automatically handle this:

```python
def convert_53_to_55_joints(pose_53):
    # Your 53 joints: [global_orient + body_pose + jaw + hands]
    # Model 55 joints: [global_orient + body_pose + jaw + EYES + hands]

    pose_55 = zeros(B, 55, 3)
    pose_55[:, :23, :] = pose_53[:, :23, :]  # Copy first 23 joints
    # Skip joints 23, 24 (eyes) - stay as zeros
    pose_55[:, 25:, :] = pose_53[:, 23:, :]  # Copy remaining joints
```

## 🎯 Training Tips

### 1. Start with Gaussian
```bash
python train_gaussian_real_data.py
```
- Fastest to train and debug
- Good sanity check for your data
- Establishes baseline performance

### 2. Then Joint Flow
```bash
python train_joint_flow_real_data.py
```
- More sophisticated than Gaussian
- Takes longer but gives better results
- Watch for NLL decreasing steadily

### 3. Finally Global Flow
```bash
python train_global_flow_real_data.py
```
- Most complex model
- Needs the most tuning
- Best results but longest training time

## 🐛 Troubleshooting

### "No .npz files found!"
```bash
# Make sure your data files are in the right location:
ls *.npz

# If they're elsewhere, update the data_dir in the training script:
data_dir = "/path/to/your/data"
```

### "CUDA out of memory"
```bash
# Reduce batch size in the training script:
batch_size=16  # or even smaller: 8, 4
```

### "Training too slow"
```bash
# Increase num_workers in data loading:
num_workers=4  # or more, up to your CPU cores
```

### "NLL not decreasing"
```bash
# Try reducing learning rate:
lr=1e-4  # instead of 1e-3

# Or increase gradient clipping:
grad_clip_val=0.5  # instead of 1.0
```

## 📈 Expected Results

### Gaussian Prior
- **Training time**: 5-10 minutes
- **Final NLL**: ~50-80 (depends on data)
- **Use case**: Fast baseline, simple inference

### Joint Flow
- **Training time**: 30-60 minutes
- **Final NLL**: ~30-50 (better than Gaussian)
- **Use case**: Kinematic constraints, per-joint modeling

### Global Flow
- **Training time**: 45-90 minutes
- **Final NLL**: ~25-45 (best results)
- **Use case**: Full pose modeling, global correlations

## 🎉 Next Steps

After training, you can:

1. **Load trained models** for inference
2. **Generate new poses** using the learned priors
3. **Use as regularizers** in other pose estimation tasks
4. **Compare different architectures** for your specific use case

## 📁 Output Structure

After training, you'll have:
```
checkpoints/
├── gaussian-real-data/          # Gaussian model checkpoints
├── joint-flow-real-data/        # Joint flow checkpoints
├── global-flow-real-data/       # Global flow checkpoints
└── hand_vae/                    # Hand VAE checkpoints

lightning_logs/
├── gaussian_real_data/          # Gaussian training logs
├── joint_flow_real_data/        # Joint flow logs
├── global_flow_real_data/       # Global flow logs
└── hand_vae/                    # Hand VAE logs
```

## 🤔 Which Model Should I Use?

- **Gaussian**: Fast baseline, simple applications
- **Joint Flow**: Best balance of quality vs speed, respects kinematics
- **Global Flow**: Highest quality, research applications
- **Hand VAE**: Specialized for hand poses, dimensionality reduction

Happy training! 🚀