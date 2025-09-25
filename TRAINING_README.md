# Hand VAE Prior - PyTorch Lightning Training

Production-ready training system for Hand VAE Prior models with comprehensive logging, checkpointing, and visualization.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pytorch-lightning tensorboard matplotlib seaborn
```

### 2. Prepare Data
```bash
# Option A: Use consolidated NPZ files (recommended for large datasets)
python create_consolidated_npz.py

# Option B: Use existing dataset splits
python create_datamodule.py  # Creates dataset_splits/
```

### 3. Train Models

**Quick Test (5 minutes):**
```bash
python train_hand_vae_lightning.py \
  --model_type standard \
  --max_epochs 5 \
  --batch_size 64 \
  --consolidated_dir test_consolidated \
  --experiment_name quick_test
```

**Production Training (2-4 hours):**
```bash
python train_hand_vae_lightning.py \
  --model_type standard \
  --max_epochs 100 \
  --batch_size 8192 \
  --splits_dir dataset_splits \
  --experiment_name production_standard_vae
```

## 🎯 Model Types

### Standard VAE (`--model_type standard`)
- **Input**: (B, 90) flattened [lhand + rhand]
- **Loss**: Gaussian reconstruction + KL divergence
- **Best for**: General hand pose modeling, fast training
- **Batch size**: 8192+ (high throughput)

### SO(3) VAE (`--model_type so3`)
- **Input**: (B, 30, 3) axis-angle per joint
- **Loss**: Geodesic distance + KL divergence
- **Best for**: Rotation-aware modeling, geometric consistency
- **Batch size**: 4096 (more complex forward pass)

## 📊 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--z_dim` | 24 | Latent dimension |
| `--hidden` | 256 | Hidden layer size |
| `--batch_size` | 8192 | Batch size (4096 for SO3) |
| `--learning_rate` | 3e-4 | Learning rate |
| `--kl_warmup_epochs` | 10 | KL annealing epochs |
| `--max_epochs` | 100 | Training epochs |

## 📈 Monitoring

**Start TensorBoard:**
```bash
tensorboard --logdir logs/
```

**Key Metrics:**
- `train/loss`, `val/loss`: ELBO loss
- `train/recon_nll`: Reconstruction quality
- `train/kl`: Latent regularization
- `val/energy_mean`: Hypothesis selection metric

**Visualizations:**
- Energy distribution histograms
- Latent space scatter plots
- Loss component breakdowns
- Beta sensitivity analysis

## 💾 Output Files

**Checkpoints:** `./checkpoints/{experiment_name}/`
- `hand_vae_epoch_XX_val_loss_Y.YYY.ckpt` - Best models
- `last.ckpt` - Latest checkpoint

**Logs:** `./logs/hand-vae-prior/{experiment_name}/`
- TensorBoard events
- Validation plots (every 5 epochs)
- Test visualizations (end of training)

## 🔧 Advanced Usage

### Multi-GPU Training
```bash
python train_hand_vae_lightning.py \
  --accelerator gpu \
  --devices 2 \
  --strategy ddp \
  --batch_size 8192
```

### Custom Experiment
```bash
python train_hand_vae_lightning.py \
  --model_type so3 \
  --z_dim 32 \
  --hidden 512 \
  --kl_warmup_epochs 20 \
  --beta_max 1.5 \
  --free_bits 0.02 \
  --experiment_name large_so3_model
```

### Resume Training
```bash
python train_hand_vae_lightning.py \
  --resume_from_checkpoint checkpoints/my_experiment/last.ckpt
```

## 📊 Expected Results

### Standard VAE
- **Train Loss**: 50-200 (depends on data complexity)
- **Validation**: Should match training (good generalization)
- **Energy**: Lower values = better for hypothesis selection
- **Training Time**: ~2-4 hours for 100 epochs

### SO(3) VAE
- **Train Loss**: 800-2000 (geodesic distances)
- **Recon Sigma**: ~0.1-0.3 radians optimal
- **More stable latent space** due to geometric constraints
- **Training Time**: ~4-6 hours for 100 epochs

## 🛠️ Troubleshooting

**Out of Memory:**
- Reduce `--batch_size` (try 4096, 2048, 1024)
- Use `--precision 16-mixed` for memory efficiency

**Slow Training:**
- Use `--consolidated_dir` instead of `--splits_dir`
- Increase `--num_workers` (try 4-8)
- Use GPU: `--accelerator gpu --devices 1`

**Poor Convergence:**
- Increase `--kl_warmup_epochs` (try 15-20)
- Adjust `--beta_max` (try 0.5-2.0)
- For SO(3): Add `--free_bits 0.02`

## 🎯 Model Loading

```python
import torch
from train_hand_vae_lightning import HandVAELightningModule

# Load trained model
model = HandVAELightningModule.load_from_checkpoint(
    'checkpoints/my_experiment/best_model.ckpt'
)
model.eval()

# Use for inference
with torch.no_grad():
    energy = model.model.energy(hand_poses)  # Lower = better
```

## 📝 Examples

See `example_usage.py` for interactive examples and detailed usage patterns:

```bash
python example_usage.py        # Interactive examples
python example_usage.py --help # Detailed usage guide
```

---

🎉 **Ready to train world-class hand pose priors!**