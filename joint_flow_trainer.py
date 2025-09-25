#!/usr/bin/env python3
"""
PyTorch Lightning Trainer for Joint Limit Normalizing Flow

Lightning-based training pipeline for the per-joint conditional normalizing flow model.
Provides professional training infrastructure with logging, checkpointing, gradient clipping,
and comprehensive validation monitoring.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os

from joint_limit_flow import JointLimitFlow


class JointFlowLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Joint Limit Flow training.

    Handles the training loop, validation, gradient clipping, and comprehensive
    metric logging for the per-joint conditional normalizing flow model.
    """

    def __init__(
        self,
        J: int = 55,
        cond_dim: int = 3,
        hidden: int = 128,
        K: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip_val: float = 1.0,
        scheduler_type: str = "cosine",  # "step", "cosine", "plateau"
        step_size: int = 30,
        gamma: float = 0.5,
        parents: list = None,
        **kwargs
    ):
        """
        Initialize Joint Flow Lightning module.

        Args:
            J: Number of joints (default: 55 for SMPL-X)
            cond_dim: Conditioning dimension per joint (default: 3)
            hidden: Hidden layer size for coupling networks (default: 128)
            K: Number of coupling layers per joint (default: 4)
            lr: Learning rate (default: 1e-3)
            weight_decay: Weight decay for regularization (default: 1e-5)
            grad_clip_val: Gradient clipping value (default: 1.0)
            scheduler_type: Learning rate scheduler type
            step_size: Step size for StepLR scheduler
            gamma: Gamma for StepLR scheduler
            parents: SMPL-X parent relationships list
        """
        super().__init__()
        self.save_hyperparameters(ignore=['parents'])

        # Create the flow model
        self.flow = JointLimitFlow(J=J, cond_dim=cond_dim, hidden=hidden, K=K)

        # Store hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip_val = grad_clip_val
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma

        # Store parents (not in hyperparameters since it's not serializable)
        if parents is None:
            # Default SMPL-X parents for 55 joints
            self.parents = self._get_default_smplx_parents()
        else:
            self.parents = parents

        # Register as buffer so it moves with model
        self.register_buffer('parents_tensor', torch.tensor(self.parents, dtype=torch.long))

    def _get_default_smplx_parents(self):
        """Get default SMPL-X parent relationships for 55 joints."""
        return [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
            16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 28, 29, 45,
            46, 47, 48, 49, 50, 51, 52, 53
        ]

    def _preprocess_batch(self, batch):
        """
        Preprocess batch data - handle both raw tensors and dicts.

        Args:
            batch: Either raw tensor, dict with 'pose_aa' key, or list/tuple

        Returns:
            pose_aa: Preprocessed pose tensor [N, J, 3]
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

        # Handle different input dimensions as per reference
        if pose_aa.dim() == 4:
            # [B, F, J, 3] -> [B*F, J, 3]
            B, F, J, _ = pose_aa.shape
            pose_aa = pose_aa.view(B * F, J, 3)
        elif pose_aa.dim() == 3:
            # Already [N, J, 3], keep as is
            pass
        elif pose_aa.dim() == 2:
            # If flattened [N, J*3], reshape to [N, J, 3]
            N = pose_aa.shape[0]
            J = self.flow.J
            pose_aa = pose_aa.view(N, J, 3)
        else:
            raise ValueError(f"Unsupported pose_aa shape: {pose_aa.shape}")

        return pose_aa

    def training_step(self, batch, batch_idx):
        """Training step - compute NLL loss with gradient clipping."""
        pose_aa = self._preprocess_batch(batch)

        # Compute negative log-likelihood
        nll = self.flow.nll(pose_aa, self.parents_tensor)  # Returns NLL per sample
        loss = nll.mean()  # Average over batch

        # Compute BPD (bits per dimension) - 3 dims per joint
        bpd = loss / (math.log(2.0) * 3.0)

        # Log metrics
        self.log('train/nll', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/nll_step', loss, on_step=True, on_epoch=False)
        self.log('train/bpd', bpd, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - compute comprehensive metrics."""
        pose_aa = self._preprocess_batch(batch)

        # Compute NLL
        nll = self.flow.nll(pose_aa, self.parents_tensor)
        val_nll = nll.mean()

        # Compute BPD
        val_bpd = val_nll / (math.log(2.0) * 3.0)

        # Log validation metrics
        self.log('val/nll', val_nll, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/bpd', val_bpd, on_step=False, on_epoch=True, prog_bar=True)

        # Compute per-joint statistics by evaluating each joint separately
        if batch_idx == 0:  # Only compute on first batch to save time
            with torch.no_grad():
                C = self.flow.build_conditions(pose_aa, self.parents)
                joint_nlls = []
                for j in range(min(5, self.flow.J)):  # Test first 5 joints only
                    x_j = pose_aa[:, j, :]  # [N, 3]
                    c_j = C[:, j, :]       # [N, cond_dim]
                    nll_j = -self.flow.flows[j].log_prob(x_j, c_j).mean()
                    joint_nlls.append(nll_j)

                joint_nlls = torch.stack(joint_nlls)
                self.log('val/nll_mean_joint', joint_nlls.mean(), on_step=False, on_epoch=True)
                self.log('val/nll_max_joint', joint_nlls.max(), on_step=False, on_epoch=True)
                self.log('val/nll_min_joint', joint_nlls.min(), on_step=False, on_epoch=True)

        # Test sampling on validation data
        if batch_idx == 0:  # Only on first validation batch
            try:
                with torch.no_grad():
                    # Sample from the flow
                    sample_size = min(4, pose_aa.shape[0])
                    samples = self.flow.sample_joints(self.parents, batch_size=sample_size)

                    # Log sample statistics
                    self.log('val/sample_mean', samples.mean(), on_step=False, on_epoch=True)
                    self.log('val/sample_std', samples.std(), on_step=False, on_epoch=True)
                    self.log('val/sample_max', samples.abs().max(), on_step=False, on_epoch=True)

                    # Test round-trip for individual joints
                    test_poses = pose_aa[:sample_size]
                    C = self.flow.build_conditions(test_poses, self.parents)

                    # Test one joint's round-trip
                    j = 0  # Test first joint
                    x_j = test_poses[:, j, :]  # [sample_size, 3]
                    c_j = C[:, j, :]          # [sample_size, cond_dim]

                    z_j, _ = self.flow.flows[j].fwd(x_j, c_j)
                    reconstructed_j, _ = self.flow.flows[j].inv(z_j, c_j)

                    round_trip_error = F.mse_loss(reconstructed_j, x_j)
                    self.log('val/round_trip_mse', round_trip_error, on_step=False, on_epoch=True)

            except Exception as e:
                # Log sampling errors but don't fail training
                self.log('val/sampling_error', 1.0, on_step=False, on_epoch=True)

        return {'val_nll': val_nll, 'val_bpd': val_bpd}

    def configure_optimizers(self):
        """Configure optimizer and scheduler with weight decay."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

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

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Configure gradient clipping as per reference implementation."""
        if self.grad_clip_val > 0:
            clip_grad_norm_(self.parameters(), self.grad_clip_val)


def create_dummy_data(num_samples: int = 1000, J: int = 55, dim: int = 3):
    """Create dummy pose data for testing."""
    # Generate realistic axis-angle poses (small angles)
    pose_data = torch.randn(num_samples, J, dim) * 0.3

    # Set eye joints to zero (indices 23, 24 for SMPL-X)
    if J == 55:
        pose_data[:, 23, :] = 0.0  # left eye
        pose_data[:, 24, :] = 0.0  # right eye

    return pose_data


def test_joint_flow_lightning_trainer():
    """Test the Joint Flow Lightning trainer."""
    print("🧪 Testing Joint Flow Lightning Trainer")
    print("=" * 60)

    # Create dummy data
    print("🔧 Creating dummy data...")
    J, dim = 55, 3
    train_data = create_dummy_data(800, J, dim)
    val_data = create_dummy_data(200, J, dim)

    # Create data loaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Train data: {train_data.shape}, Val data: {val_data.shape}")

    # Test data preprocessing
    print("\n🔧 Testing data preprocessing...")
    sample_batch = next(iter(train_loader))[0]
    print(f"Sample batch shape: {sample_batch.shape}")

    # Create Lightning module
    print("\n🔧 Creating Lightning module...")
    model = JointFlowLightningModule(
        J=J,
        cond_dim=dim,
        hidden=64,  # Smaller for testing
        K=2,        # Fewer layers for testing
        lr=1e-3,
        weight_decay=1e-5,
        grad_clip_val=1.0,
        scheduler_type="cosine"
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test preprocessing
    processed = model._preprocess_batch(sample_batch)
    print(f"Processed shape: {processed.shape}")

    # Test forward pass
    print("\n🔧 Testing forward pass...")
    with torch.no_grad():
        nll = model.flow.nll(processed, model.parents_tensor)
        print(f"NLL shape: {nll.shape}, mean: {nll.mean():.3f}")
        bpd = nll.mean() / (math.log(2.0) * 3.0)
        print(f"BPD: {bpd:.4f}")

    # Test sampling
    print("\n🔧 Testing sampling...")
    with torch.no_grad():
        samples = model.flow.sample_joints(model.parents, batch_size=4)
        print(f"Sample shape: {samples.shape}")

    # Setup Lightning trainer
    print("\n🚀 Setting up Lightning trainer...")

    # Create checkpoint directory
    checkpoint_dir = "test_checkpoints/joint-flow/test-run"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="joint-flow-{epoch:02d}-{val/nll:.4f}",
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
        name="joint_flow_training",
        version="test"
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="cpu",  # Use CPU for testing
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=5,
        val_check_interval=1.0,
        enable_model_summary=True,
        enable_progress_bar=True,
        gradient_clip_val=1.0,  # Enable gradient clipping
        deterministic=False
    )

    print(f"🚀 Starting Joint Flow Lightning Training")
    print(f"  Joints: {J}, Conditioning: {dim}D")
    print(f"  Architecture: {model.hparams.K} coupling layers, {model.hparams.hidden} hidden")
    print(f"  Training: 5 epochs, LR: {model.lr}, Weight decay: {model.weight_decay}")
    print(f"  Gradient clipping: {model.grad_clip_val}")
    print(f"  Logger: tensorboard, Checkpoints: {checkpoint_dir}")

    # Train the model
    print("\n🔥 Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Print results
    print(f"\n🎉 Training completed!")
    if checkpoint_callback.best_model_path:
        print(f"  Best model path: {checkpoint_callback.best_model_path}")
        print(f"  Best validation NLL: {checkpoint_callback.best_model_score:.4f}")
        best_bpd = checkpoint_callback.best_model_score / (math.log(2.0) * 3.0)
        print(f"  Best validation BPD: {best_bpd:.4f}")

    # Test sampling from trained model
    print("\n🔧 Testing trained model sampling...")
    model.eval()
    with torch.no_grad():
        samples = model.flow.sample_joints(model.parents, batch_size=4)
        print(f"✅ Generated samples: {samples.shape}")
        print(f"Sample statistics: mean={samples.mean():.3f}, std={samples.std():.3f}")

        # Test round-trip for individual joints
        test_poses = processed[:4]
        C = model.flow.build_conditions(test_poses, model.parents)

        # Test first joint's round-trip
        j = 0
        x_j = test_poses[:, j, :]  # [4, 3]
        c_j = C[:, j, :]          # [4, cond_dim]

        z_j, _ = model.flow.flows[j].fwd(x_j, c_j)
        reconstructed_j, _ = model.flow.flows[j].inv(z_j, c_j)
        round_trip_error = F.mse_loss(reconstructed_j, x_j)
        print(f"Round-trip MSE (joint {j}): {round_trip_error.item():.6f}")

    # Test checkpoint loading
    print("\n🔧 Testing checkpoint loading...")
    if checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        loaded_model = JointFlowLightningModule.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        print("✅ Loaded model from checkpoint")

        # Test loaded model
        with torch.no_grad():
            loaded_samples = loaded_model.flow.sample_joints(loaded_model.parents, batch_size=2)
            print(f"✅ Loaded model sampling: {loaded_samples.shape}")

    print(f"\n🎉 Joint Flow Lightning trainer test completed!")

    print(f"\n💡 Lightning Training Benefits:")
    print(f"  • Automatic gradient clipping for stable training")
    print(f"  • Per-joint statistics and BPD monitoring")
    print(f"  • Built-in sampling validation and round-trip testing")
    print(f"  • Professional logging and checkpointing")
    print(f"  • Weight decay regularization")
    print(f"  • Learning rate scheduling with multiple options")
    print(f"  • Easy scaling to GPU/multi-GPU setups")
    print(f"  • Comprehensive validation metrics")

    return True


def main():
    """Main training function."""
    return test_joint_flow_lightning_trainer()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)