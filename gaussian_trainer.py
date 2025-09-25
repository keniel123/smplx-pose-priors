#!/usr/bin/env python3
"""
PyTorch Lightning Trainer for Joint Limit Gaussian Prior

Lightning-based training pipeline for the factorized Gaussian prior model.
Provides professional training infrastructure with logging, checkpointing,
and validation monitoring.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os

from joint_limit_gaussian import JointLimitGaussian


class GaussianLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Joint Limit Gaussian training.

    Handles the training loop, validation, and metric logging for the
    factorized Gaussian prior model.
    """

    def __init__(
        self,
        J: int = 55,
        dim: int = 3,
        lr: float = 1e-2,
        scheduler_type: str = "step",  # "step", "cosine", "plateau"
        step_size: int = 30,
        gamma: float = 0.5,
        **kwargs
    ):
        """
        Initialize Gaussian Lightning module.

        Args:
            J: Number of joints (default: 55 for SMPL-X)
            dim: Dimension per joint (default: 3 for axis-angle)
            lr: Learning rate (default: 1e-2)
            scheduler_type: Learning rate scheduler type
            step_size: Step size for StepLR scheduler
            gamma: Gamma for StepLR scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        # Create the Gaussian model
        self.gaussian = JointLimitGaussian(J=J, dim=dim)

        # Store hyperparameters
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma

    def _preprocess_batch(self, batch):
        """
        Preprocess batch data - handle both raw tensors and dicts.

        Args:
            batch: Either raw tensor, dict with 'pose_aa' key, or list/tuple

        Returns:
            pose_aa: Preprocessed pose tensor [B, J, 3]
        """
        # Handle different batch formats
        if torch.is_tensor(batch):
            pose_aa = batch
        elif isinstance(batch, (list, tuple)):
            # DataLoader returns list for TensorDataset
            pose_aa = batch[0]
        elif isinstance(batch, dict):
            pose_aa = batch["pose_aa"]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        # Ensure correct shape [B, J, 3]
        if pose_aa.dim() == 2:
            # If flattened [B, J*3], reshape to [B, J, 3]
            B = pose_aa.shape[0]
            J = self.gaussian.J
            pose_aa = pose_aa.view(B, J, 3)

        return pose_aa

    def training_step(self, batch, batch_idx):
        """Training step - compute NLL loss."""
        pose_aa = self._preprocess_batch(batch)

        # Compute negative log-likelihood (loss)
        nll = self.gaussian(pose_aa)  # Returns NLL per sample
        loss = nll.mean()  # Average over batch

        # Log metrics
        self.log('train/nll', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/nll_step', loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - compute metrics."""
        pose_aa = self._preprocess_batch(batch)

        # Compute NLL
        nll = self.gaussian.nll(pose_aa)  # Use explicit nll method
        val_nll = nll.mean()

        # Compute additional metrics
        log_prob = self.gaussian.log_prob_per_frame(pose_aa)  # [B, F]
        val_log_prob = log_prob.mean()

        # Log validation metrics
        self.log('val/nll', val_nll, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/log_prob', val_log_prob, on_step=False, on_epoch=True)

        # Compute per-joint statistics
        nll_per_joint = nll.mean(0)  # [J]
        self.log('val/nll_mean_joint', nll_per_joint.mean(), on_step=False, on_epoch=True)
        self.log('val/nll_max_joint', nll_per_joint.max(), on_step=False, on_epoch=True)
        self.log('val/nll_min_joint', nll_per_joint.min(), on_step=False, on_epoch=True)

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


def create_dummy_data(num_samples: int = 1000, J: int = 55, dim: int = 3):
    """Create dummy pose data for testing."""
    # Generate realistic axis-angle poses (small angles)
    pose_data = torch.randn(num_samples, J, dim) * 0.3

    # Set eye joints to zero (indices 23, 24 for SMPL-X)
    if J == 55:
        pose_data[:, 23, :] = 0.0  # left eye
        pose_data[:, 24, :] = 0.0  # right eye

    return pose_data


def test_gaussian_lightning_trainer():
    """Test the Gaussian Lightning trainer."""
    print("🧪 Testing Gaussian Lightning Trainer")
    print("=" * 60)

    # Create dummy data
    print("🔧 Creating dummy data...")
    J, dim = 55, 3
    train_data = create_dummy_data(800, J, dim)
    val_data = create_dummy_data(200, J, dim)

    # Create data loaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"Train data: {train_data.shape}, Val data: {val_data.shape}")

    # Test data preprocessing
    print("\n🔧 Testing data preprocessing...")
    sample_batch = next(iter(train_loader))[0]
    print(f"Sample batch shape: {sample_batch.shape}")

    # Create Lightning module
    print("\n🔧 Creating Lightning module...")
    model = GaussianLightningModule(
        J=J, dim=dim, lr=1e-2, scheduler_type="step"
    )

    # Test preprocessing
    processed = model._preprocess_batch(sample_batch)
    print(f"Processed shape: {processed.shape}")

    # Test forward pass
    print("\n🔧 Testing forward pass...")
    with torch.no_grad():
        nll = model.gaussian(processed)
        log_prob = model.gaussian.log_prob_per_frame(processed)
        print(f"NLL shape: {nll.shape}, mean: {nll.mean():.3f}")
        print(f"Log prob shape: {log_prob.shape}, mean: {log_prob.mean():.3f}")

    # Setup Lightning trainer
    print("\n🚀 Setting up Lightning trainer...")

    # Create checkpoint directory
    checkpoint_dir = "test_checkpoints/gaussian/test-run"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="gaussian-{epoch:02d}-{val/nll:.4f}",
        monitor="val/nll",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True
    )

    early_stopping = EarlyStopping(
        monitor="val/nll",
        mode="min",
        patience=10,
        min_delta=1e-4,
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="gaussian_training",
        version="test"
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="cpu",  # Use CPU for testing
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_model_summary=True,
        enable_progress_bar=True,
        deterministic=False
    )

    print(f"🚀 Starting Gaussian Lightning Training")
    print(f"  Joints: {J}, Dimension: {dim}")
    print(f"  Architecture: Factorized Gaussian per joint")
    print(f"  Training: 5 epochs, LR: 0.01, Scheduler: step")
    print(f"  Logger: tensorboard, Checkpoints: {checkpoint_dir}")

    # Train the model
    print("\n🔥 Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Print results
    print(f"\n🎉 Training completed!")
    if checkpoint_callback.best_model_path:
        print(f"  Best model path: {checkpoint_callback.best_model_path}")
        print(f"  Best validation NLL: {checkpoint_callback.best_model_score:.4f}")

    # Test sampling
    print("\n🔧 Testing model sampling...")
    model.eval()
    with torch.no_grad():
        samples = model.gaussian.sample(4)  # Sample 4 poses
        print(f"✅ Generated samples: {samples.shape}")
        print(f"Sample statistics: mean={samples.mean():.3f}, std={samples.std():.3f}")

        # Check eye joints are handled correctly (samples have shape [B, F, J, 3])
        if J == 55 and samples.dim() == 4:
            eye_samples = samples[:, :, [23, 24], :].abs().max()
            print(f"Eye joint samples (should be small): max={eye_samples:.6f}")
        elif J == 55 and samples.dim() == 3:
            eye_samples = samples[:, [23, 24], :].abs().max()
            print(f"Eye joint samples (should be small): max={eye_samples:.6f}")

    # Test checkpoint loading
    print("\n🔧 Testing checkpoint loading...")
    if checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        loaded_model = GaussianLightningModule.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        print("✅ Loaded model from checkpoint")

        # Test loaded model
        with torch.no_grad():
            loaded_samples = loaded_model.gaussian.sample(2)
            print(f"✅ Loaded model sampling: {loaded_samples.shape}")

    print(f"\n🎉 Gaussian Lightning trainer test completed!")

    print(f"\n💡 Lightning Training Benefits:")
    print(f"  • Simple MLE fitting with automatic differentiation")
    print(f"  • Built-in logging and checkpointing")
    print(f"  • Per-joint statistics monitoring")
    print(f"  • Learning rate scheduling")
    print(f"  • Early stopping for optimal convergence")
    print(f"  • Easy scaling and distributed training")

    return True


def main():
    """Main training function."""
    return test_gaussian_lightning_trainer()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)