#!/usr/bin/env python3
"""
Example usage of the Hand VAE Prior Lightning training script
"""

import subprocess
import sys
from pathlib import Path

def run_training_example():
    """Run a quick training example"""

    print("🚀 Hand VAE Prior Training Examples")
    print("=" * 50)

    # Example commands for different scenarios
    examples = [
        {
            "name": "Standard VAE - Quick Test",
            "description": "Train standard VAE with small settings for testing",
            "command": [
                "python", "train_hand_vae_lightning.py",
                "--model_type", "standard",
                "--z_dim", "16",
                "--hidden", "128",
                "--n_layers", "2",
                "--batch_size", "64",
                "--max_epochs", "5",
                "--consolidated_dir", "test_consolidated",
                "--num_workers", "0",
                "--experiment_name", "quick_test_standard"
            ]
        },
        {
            "name": "SO(3) VAE - Quick Test",
            "description": "Train SO(3) VAE with small settings for testing",
            "command": [
                "python", "train_hand_vae_lightning.py",
                "--model_type", "so3",
                "--z_dim", "16",
                "--hidden", "128",
                "--n_layers", "2",
                "--batch_size", "64",
                "--max_epochs", "5",
                "--consolidated_dir", "test_consolidated",
                "--num_workers", "0",
                "--experiment_name", "quick_test_so3"
            ]
        },
        {
            "name": "Production Standard VAE",
            "description": "Production training with full dataset and settings",
            "command": [
                "python", "train_hand_vae_lightning.py",
                "--model_type", "standard",
                "--z_dim", "24",
                "--hidden", "256",
                "--n_layers", "3",
                "--batch_size", "8192",
                "--max_epochs", "100",
                "--kl_warmup_epochs", "10",
                "--splits_dir", "dataset_splits",  # Use full dataset
                "--num_workers", "4",
                "--precision", "16-mixed",
                "--experiment_name", "production_standard_vae"
            ]
        },
        {
            "name": "Production SO(3) VAE",
            "description": "Production SO(3) training with geometric awareness",
            "command": [
                "python", "train_hand_vae_lightning.py",
                "--model_type", "so3",
                "--z_dim", "24",
                "--hidden", "256",
                "--n_layers", "3",
                "--batch_size", "4096",  # Smaller due to more complex forward pass
                "--max_epochs", "100",
                "--kl_warmup_epochs", "15",
                "--free_bits", "0.02",
                "--splits_dir", "dataset_splits",
                "--num_workers", "4",
                "--precision", "16-mixed",
                "--experiment_name", "production_so3_vae"
            ]
        }
    ]

    print("Available training examples:")
    for i, example in enumerate(examples):
        print(f"\n{i+1}. {example['name']}")
        print(f"   {example['description']}")
        print(f"   Command: {' '.join(example['command'])}")

    print(f"\n" + "="*50)

    # Interactive selection
    try:
        choice = input("Enter example number to run (1-4), or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            print("Goodbye!")
            return

        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(examples):
            example = examples[choice_idx]
            print(f"\n🔥 Running: {example['name']}")
            print(f"Command: {' '.join(example['command'])}")

            # Ask for confirmation
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                # Run the command
                result = subprocess.run(example['command'], cwd=Path(__file__).parent)
                if result.returncode == 0:
                    print("\n✅ Training completed successfully!")
                else:
                    print(f"\n❌ Training failed with exit code {result.returncode}")
            else:
                print("Cancelled.")
        else:
            print("Invalid choice.")

    except (ValueError, KeyboardInterrupt):
        print("\nCancelled.")


def show_usage_info():
    """Show detailed usage information"""

    print("📖 Hand VAE Prior Training - Usage Guide")
    print("=" * 50)

    print("""
🎯 TRAINING MODES:

1. Standard VAE (hand_vae_prior.py):
   • Input: (B, 90) flattened [lhand + rhand]
   • Loss: Gaussian reconstruction + KL divergence
   • Best for: General hand pose modeling
   • Typical settings: z_dim=24, hidden=256, batch_size=8192

2. SO(3) VAE (hand_vae_prior_so3.py):
   • Input: (B, 30, 3) axis-angle per joint
   • Loss: Geodesic distance + KL divergence
   • Best for: Rotation-aware modeling
   • Typical settings: z_dim=24, hidden=256, batch_size=4096

🔧 KEY PARAMETERS:

Model Architecture:
  --z_dim          Latent dimension (16-32, default: 24)
  --hidden         Hidden layer size (128-512, default: 256)
  --n_layers       Number of layers (2-4, default: 3)
  --dropout        Dropout rate (0.0-0.2, default: 0.1)

Training:
  --learning_rate  Learning rate (1e-4 to 1e-3, default: 3e-4)
  --batch_size     Batch size (Standard: 8192, SO(3): 4096)
  --max_epochs     Training epochs (50-200, default: 100)
  --kl_warmup_epochs  KL annealing (5-20, default: 10)
  --beta_max       Max KL weight (0.5-2.0, default: 1.0)

Data:
  --consolidated_dir  Use consolidated NPZ files (faster)
  --splits_dir       Use multi-file splits (more flexible)
  --num_workers      Data loading workers (0-8, default: 4)

📊 OUTPUTS:

Checkpoints: ./checkpoints/{experiment_name}/
  • hand_vae_epoch_XX_val_loss_Y.YYY.ckpt
  • last.ckpt (latest checkpoint)

Logs: ./logs/{project_name}/{experiment_name}/
  • TensorBoard logs with training curves
  • Validation plots every 5 epochs
  • Comprehensive test visualizations

Model Components:
  • model.state_dict(): Trained weights
  • model.x_mean, model.x_std: Data normalization
  • Hyperparameters saved in checkpoint

🎯 TYPICAL WORKFLOWS:

Quick Test (5 minutes):
  python train_hand_vae_lightning.py --model_type standard --max_epochs 5
  --batch_size 64 --consolidated_dir test_consolidated --experiment_name test

Full Training (2-4 hours):
  python train_hand_vae_lightning.py --model_type standard --max_epochs 100
  --batch_size 8192 --splits_dir dataset_splits --experiment_name production

GPU Training:
  python train_hand_vae_lightning.py --accelerator gpu --devices 1
  --precision 16-mixed --batch_size 16384

Multi-GPU:
  python train_hand_vae_lightning.py --accelerator gpu --devices 2
  --strategy ddp --batch_size 8192

🔍 MONITORING:

TensorBoard:
  tensorboard --logdir logs/

Key Metrics:
  • train/loss, val/loss: ELBO loss
  • train/recon_nll: Reconstruction quality
  • train/kl: Latent regularization
  • val/energy_mean: Hypothesis selection metric
  • train/beta: KL annealing schedule

Validation Plots (every 5 epochs):
  • Energy distribution histograms
  • Latent space visualizations
  • Loss component breakdowns
  • Reconstruction quality metrics

Test Plots (end of training):
  • Beta sensitivity analysis
  • Comprehensive latent analysis
  • Performance comparisons

📈 EXPECTED RESULTS:

Standard VAE:
  • Train loss: 50-200 (depends on data complexity)
  • Val loss: Similar to train (good generalization)
  • Energy: Lower = better for hypothesis selection
  • Latent dims: Should show diverse usage

SO(3) VAE:
  • Train loss: 800-2000 (geodesic distances)
  • Recon sigma: ~0.1-0.3 radians optimal
  • More stable latent space due to geometry

🚀 READY TO TRAIN!
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage_info()
    else:
        run_training_example()