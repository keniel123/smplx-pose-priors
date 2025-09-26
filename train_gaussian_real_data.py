#!/usr/bin/env python3
"""
Gaussian Trainer with Real SMPL-X Data

Connects the Gaussian joint limit prior to real SMPL-X pose data
using the simple pose data module with consolidated splits. Handles joint
count conversion from 53 joints (data) to 55 joints (model with eye joints).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
import argparse
from pathlib import Path

from joint_limit_gaussian import JointLimitGaussian
from simple_pose_datamodule import SimplePoseDataModule
from config_utils import ConfigLoader, create_config_parser, override_config, print_config


# Joint conversion now handled in simple_pose_datamodule.py
# Data module returns poses directly as (B, 55, 3) with eye joints as zeros


class GaussianRealDataModule(pl.LightningModule):
    """
    Lightning module for Gaussian prior with real SMPL-X data.
    """

    def __init__(
        self,
        splits_dir: str,
        lr: float = 1e-2,
        scheduler_type: str = "step",
        step_size: int = 30,
        gamma: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create the Gaussian model (55 joints)
        self.gaussian = JointLimitGaussian(J=55, dim=3)

        # Store hyperparameters
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma
        self.splits_dir = splits_dir

    def training_step(self, batch, batch_idx):
        """Training step - compute NLL loss."""
        # Data module already returns (B, 55, 3) with eye joints as zeros
        pose_aa = batch  # Already in correct format

        # Compute negative log-likelihood
        nll = self.gaussian(pose_aa)
        loss = nll.mean()

        # Log metrics
        self.log('train/nll', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - compute metrics."""
        # Data module already returns (B, 55, 3) with eye joints as zeros
        pose_aa = batch  # Already in correct format

        # Compute NLL
        nll = self.gaussian.nll(pose_aa)
        val_nll = nll.mean()

        # Compute additional metrics
        log_prob = self.gaussian.log_prob_per_frame(pose_aa)
        val_log_prob = log_prob.mean()

        # Log validation metrics
        self.log('val/nll', val_nll, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/log_prob', val_log_prob, on_step=False, on_epoch=True)

        return {'val_nll': val_nll, 'val_log_prob': val_log_prob}

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.gamma
            )
            return [optimizer], [scheduler]
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return [optimizer], [scheduler]
        else:
            return optimizer


def parse_args():
    """Parse command line arguments."""
    # Create base config parser
    config_parser = create_config_parser()

    # Add additional arguments specific to this trainer
    parser = argparse.ArgumentParser(description='Train Gaussian Prior with Real SMPL-X Data',
                                   parents=[config_parser])

    return parser.parse_args()


def train_gaussian_with_real_data():
    """Main training function with real data."""
    args = parse_args()

    print("🔥 Training Gaussian Prior with Real SMPL-X Data")

    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)

    # Apply command line overrides
    if args.override:
        config = override_config(config, args.override)

    # Print final configuration
    print_config(config, "Gaussian Prior Configuration")

    # Extract configuration values
    splits_dir = config['data'].get('splits_dir', 'dataset_splits')

    # Check for consolidated NPZ files
    splits_path = Path(splits_dir)
    required_files = ['train.npz', 'val.npz', 'test.npz']
    missing_files = [f for f in required_files if not (splits_path / f).exists()]

    if missing_files:
        print("❌ Missing consolidated split files! Please run quick_consolidated_splits.py first.")
        print(f"📁 Looking in: {splits_dir}")
        print(f"💡 Missing files: {missing_files}")
        return False

    print(f"📊 Found consolidated split files in {splits_dir}:")
    for f in required_files:
        file_path = splits_path / f
        file_size = file_path.stat().st_size / (1024*1024)  # MB
        print(f"  • {f} ({file_size:.1f} MB)")

    # Create data module
    print("\n🔧 Setting up SimplePoseDataModule...")
    data_module = SimplePoseDataModule(
        splits_dir=str(splits_dir),
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        return_dict=config['data']['return_dict']
    )

    # Setup data
    data_module.setup()

    # Create model
    print("\n🔧 Creating Lightning module...")
    model = GaussianRealDataModule(
        splits_dir=str(splits_dir),
        lr=config['training']['learning_rate'],
        scheduler_type=config['training']['scheduler_type'],
        step_size=config['training']['step_size'],
        gamma=config['training']['gamma']
    )

    # Create checkpoint directory
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="gaussian-real-{epoch:02d}-{val/nll:.4f}",
        monitor="val/nll",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True
    )

    early_stopping = EarlyStopping(
        monitor="val/nll",
        mode="min",
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=config['logging']['logger_name'],
        version=None
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=config['logging']['log_every_n_steps'],
        val_check_interval=1.0,
        enable_model_summary=True,
        enable_progress_bar=True,
        deterministic=False
    )

    print(f"🚀 Starting training...")
    print(f"  Max epochs: {config['training']['max_epochs']}")
    print(f"  Learning rate: {model.lr}")
    print(f"  Scheduler: {model.scheduler_type}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Checkpoints: {checkpoint_dir}")

    try:
        # Train the model
        trainer.fit(model, data_module)

        # Print results
        print(f"\n🎉 Training completed!")
        if checkpoint_callback.best_model_path:
            print(f"  Best model: {checkpoint_callback.best_model_path}")
            print(f"  Best val NLL: {checkpoint_callback.best_model_score:.4f}")

        # Test sampling
        print(f"\n🔧 Testing model sampling...")
        model.eval()
        with torch.no_grad():
            samples = model.gaussian.sample(batch_size=4, num_frames=1)
            print(f"✅ Generated samples: {samples.shape}")
            print(f"Sample statistics: mean={samples.mean():.3f}, std={samples.std():.3f}")

        print(f"\n💡 Training completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False


if __name__ == "__main__":
    success = train_gaussian_with_real_data()
    exit(0 if success else 1)