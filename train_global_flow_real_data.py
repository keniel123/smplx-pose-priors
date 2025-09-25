#!/usr/bin/env python3
"""
Global Flow Trainer with Real SMPL-X Data

Connects the global conditional flow to real SMPL-X pose data
using the comprehensive data module. Handles joint count conversion
from 53 joints (data) to 55 joints (model with eye joints).
"""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
import argparse
from pathlib import Path

from global_conditional_flow import CondFlowNet
from comprehensive_pose_datamodule import ComprehensivePoseDataModule


# Joint conversion now handled in comprehensive_pose_datamodule.py
# Data module returns poses directly as (B, 55, 3) with eye joints as zeros


class GlobalFlowRealDataModule(pl.LightningModule):
    """
    Lightning module for Global Flow with real SMPL-X data.
    """

    def __init__(
        self,
        data_dir: str,
        hidden: int = 512,
        K: int = 6,
        use_actnorm: bool = True,
        lr: float = 1e-3,
        scheduler_type: str = "cosine",
        use_conditioning: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Dimensions
        self.pose_dim = 55 * 3  # 165D for 55 joints
        self.cond_dim = 512 if use_conditioning else 0  # Dummy conditioning

        # Create the flow model
        self.flow = CondFlowNet(
            dim=self.pose_dim,
            cond_dim=self.cond_dim,
            hidden=hidden,
            K=K,
            use_actnorm=use_actnorm
        )

        # Store hyperparameters
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.data_dir = data_dir
        self.use_conditioning = use_conditioning

    def _preprocess_batch(self, batch):
        """Preprocess batch data."""
        # Data module already returns (B, 55, 3) with eye joints as zeros
        pose_aa = batch  # Already in correct format [B, 55, 3]

        # Flatten to [B, 165]
        x = pose_aa.view(pose_aa.shape[0], -1)

        # Create dummy conditioning if needed
        if self.use_conditioning:
            c = torch.randn(x.shape[0], self.cond_dim, device=x.device)
        else:
            c = torch.zeros(x.shape[0], 1, device=x.device)  # Minimal conditioning

        return x, c

    def training_step(self, batch, batch_idx):
        """Training step - compute NLL loss."""
        x, c = self._preprocess_batch(batch)

        # Compute log probability
        log_prob = self.flow.log_prob(x, c)
        nll_loss = -log_prob.mean()

        # Compute BPD (bits per dimension)
        bpd = nll_loss / (math.log(2.0) * self.pose_dim)

        # Log metrics
        self.log('train/nll', nll_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/bpd', bpd, on_step=True, on_epoch=True, prog_bar=True)

        return nll_loss

    def validation_step(self, batch, batch_idx):
        """Validation step - compute comprehensive metrics."""
        x, c = self._preprocess_batch(batch)

        # Compute log probability and NLL
        log_prob = self.flow.log_prob(x, c)
        val_nll = -log_prob.mean()
        val_bpd = val_nll / (math.log(2.0) * self.pose_dim)

        # Log validation metrics
        self.log('val/nll', val_nll, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/bpd', val_bpd, on_step=False, on_epoch=True, prog_bar=True)

        # Additional validation metrics on first batch
        if batch_idx == 0:
            with torch.no_grad():
                # Sample from flow
                sample_size = min(4, x.shape[0])
                z_samples = torch.randn(sample_size, self.pose_dim, device=x.device)
                c_samples = c[:sample_size]

                try:
                    x_samples = self.flow.sample(z_samples, c_samples)

                    # Log sample statistics
                    self.log('val/sample_mean', x_samples.mean(), on_step=False, on_epoch=True)
                    self.log('val/sample_std', x_samples.std(), on_step=False, on_epoch=True)
                    self.log('val/sample_max', x_samples.abs().max(), on_step=False, on_epoch=True)

                    # Test round-trip reconstruction
                    x_test = x[:sample_size]
                    c_test = c[:sample_size]

                    z_fwd, _ = self.flow.forward(x_test, c_test)
                    x_recon = self.flow.sample(z_fwd, c_test)

                    round_trip_mse = F.mse_loss(x_recon, x_test)
                    self.log('val/round_trip_mse', round_trip_mse, on_step=False, on_epoch=True)

                    # Check base distribution properties
                    z_mean = z_fwd.mean()
                    z_std = z_fwd.std()
                    self.log('val/latent_mean', z_mean, on_step=False, on_epoch=True)
                    self.log('val/latent_std', z_std, on_step=False, on_epoch=True)

                except Exception as e:
                    # Log sampling errors but don't fail training
                    self.log('val/sampling_error', 1.0, on_step=False, on_epoch=True)

        return {'val_nll': val_nll, 'val_bpd': val_bpd}

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return [optimizer], [scheduler]
        elif self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/nll',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Global Flow with Real SMPL-X Data')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/Users/kenielpeart/Downloads/hand_prior/code',
                       help='Directory containing NPZ data files')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')

    # Model arguments
    parser.add_argument('--hidden', type=int, default=512,
                       help='Hidden layer size')
    parser.add_argument('--K', type=int, default=6,
                       help='Number of coupling layers')
    parser.add_argument('--use_actnorm', action='store_true', default=True,
                       help='Use ActNorm layers')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'plateau'],
                       help='Learning rate scheduler type')
    parser.add_argument('--use_conditioning', action='store_true', default=False,
                       help='Use conditioning (experimental)')

    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=30,
                       help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                       help='Minimum change for early stopping')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')

    # Logging arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/global-flow-real-data',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                       help='Log every N steps')

    # Hardware arguments
    parser.add_argument('--accelerator', type=str, default='auto',
                       help='Accelerator to use')
    parser.add_argument('--devices', type=str, default='auto',
                       help='Devices to use')

    return parser.parse_args()


def train_global_flow_with_real_data():
    """Main training function with real data."""
    args = parse_args()

    print("🔥 Training Global Flow with Real SMPL-X Data")
    print("=" * 60)
    print(f"Arguments: {vars(args)}")

    # Configuration
    data_dir = args.data_dir

    # Find NPZ files
    npz_files = list(Path(data_dir).glob("*.npz"))
    if not npz_files:
        print("❌ No .npz files found! Please ensure your data files are in the directory.")
        print(f"📁 Looking in: {data_dir}")
        print("💡 Expected files like: train_poses.npz, val_poses.npz, etc.")
        return False

    print(f"📊 Found {len(npz_files)} NPZ files:")
    for f in npz_files:
        print(f"  • {f.name}")

    # Create data module
    print("\n🔧 Setting up data module...")
    data_module = ComprehensivePoseDataModule(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_dict=False  # Return raw (53, 3) tensors
    )

    # Setup data
    data_module.setup()

    # Create model
    print("\n🔧 Creating Lightning module...")
    model = GlobalFlowRealDataModule(
        data_dir=str(data_dir),
        hidden=args.hidden,
        K=args.K,
        use_actnorm=args.use_actnorm,
        lr=args.lr,
        scheduler_type=args.scheduler_type,
        use_conditioning=args.use_conditioning
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="global-flow-real-{epoch:02d}-{val/nll:.4f}",
        monitor="val/nll",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True
    )

    early_stopping = EarlyStopping(
        monitor="val/nll",
        mode="min",
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="global_flow_real_data",
        version=None
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=1.0,
        enable_model_summary=True,
        enable_progress_bar=True,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=False
    )

    print(f"🚀 Starting training...")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Learning rate: {model.lr}")
    print(f"  Scheduler: {model.scheduler_type}")
    print(f"  Architecture: {model.flow.K} layers, {model.flow.hidden} hidden")
    print(f"  ActNorm: {model.flow.use_actnorm}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Conditioning: {args.use_conditioning}")
    print(f"  Checkpoints: {checkpoint_dir}")

    try:
        # Train the model
        trainer.fit(model, data_module)

        # Print results
        print(f"\n🎉 Training completed!")
        if checkpoint_callback.best_model_path:
            print(f"  Best model: {checkpoint_callback.best_model_path}")
            print(f"  Best val NLL: {checkpoint_callback.best_model_score:.4f}")
            best_bpd = checkpoint_callback.best_model_score / (math.log(2.0) * model.pose_dim)
            print(f"  Best val BPD: {best_bpd:.4f}")

        # Test sampling
        print(f"\n🔧 Testing model sampling...")
        model.eval()
        with torch.no_grad():
            z = torch.randn(4, model.pose_dim)
            c = torch.zeros(4, 1) if not model.use_conditioning else torch.randn(4, model.cond_dim)
            samples = model.flow.sample(z, c)
            print(f"✅ Generated samples: {samples.shape}")
            print(f"Sample statistics: mean={samples.mean():.3f}, std={samples.std():.3f}")

        print(f"\n💡 Training completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = train_global_flow_with_real_data()
    exit(0 if success else 1)