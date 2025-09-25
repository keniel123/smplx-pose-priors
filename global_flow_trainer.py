#!/usr/bin/env python3
"""
PyTorch Lightning Trainer for Global Conditional Flow

Production-ready Lightning training pipeline for CondFlowNet with comprehensive monitoring,
validation metrics, and best model tracking. Designed for SMPL-X pose modeling
with conditional features (text, context, etc.).

Features:
- PyTorch Lightning integration for scalable training
- Automatic data reshaping for [B,F,J,3] -> [N,D] conversion
- Gradient clipping and optimization best practices
- Validation metrics including round-trip error and base distribution checks
- Lightning callbacks for checkpointing and monitoring
- Comprehensive logging with TensorBoard/W&B support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import Dict, Optional, Any, Tuple, Union
import time
import os
from pathlib import Path

# Import our modules
from global_conditional_flow import CondFlowNet
from rotation_utils import wrap_to_pi, soft_clamp_aa


@torch.no_grad()
def _flatten_inputs(pose_aa_bfJC: torch.Tensor, cond_bfC: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten pose and conditioning tensors for flow processing.

    Args:
        pose_aa_bfJC: Pose tensor [B, F, J, 3] (batch, frames, joints, axis-angle)
        cond_bfC: Conditioning tensor [B, F, C] (batch, frames, conditioning_dim)

    Returns:
        x: Flattened poses [N, D] where N=B*F, D=J*3
        c: Flattened conditioning [N, C] where N=B*F
    """
    B, F, J, _ = pose_aa_bfJC.shape
    x = pose_aa_bfJC.view(B * F, J * 3)    # [N, D]
    c = cond_bfC.view(B * F, -1)           # [N, C]
    return x, c


@torch.no_grad()
def preprocess_poses(poses: torch.Tensor,
                    apply_wrapping: bool = True,
                    apply_clamping: bool = True,
                    clamp_limit: float = math.pi) -> torch.Tensor:
    """
    Preprocess pose data for flow training.

    Args:
        poses: Raw pose tensor [..., J, 3]
        apply_wrapping: Whether to wrap angles to [-π, π]
        apply_clamping: Whether to apply soft clamping
        clamp_limit: Maximum magnitude for soft clamping

    Returns:
        processed_poses: Preprocessed pose tensor
    """
    if apply_wrapping:
        poses = wrap_to_pi(poses)

    if apply_clamping:
        # Apply soft clamping to each joint
        original_shape = poses.shape
        poses_flat = poses.view(-1, 3)
        poses_flat = soft_clamp_aa(poses_flat, limit=clamp_limit)
        poses = poses_flat.view(original_shape)

    return poses


class GlobalFlowLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training Global Conditional Flow.

    Handles all training/validation logic, metrics computation, and optimization.
    """

    def __init__(
        self,
        dim: int = 165,
        cond_dim: int = 512,
        hidden: int = 512,
        K: int = 6,
        use_actnorm: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        preprocess_poses: bool = True,
        grad_clip_val: Optional[float] = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create flow model
        self.flow = CondFlowNet(
            dim=dim,
            cond_dim=cond_dim,
            hidden=hidden,
            K=K,
            use_actnorm=use_actnorm
        )

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.preprocess_poses = preprocess_poses
        self.grad_clip_val = grad_clip_val

        # Metrics tracking
        self.train_step_outputs = []
        self.val_step_outputs = []

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute log probability."""
        return self.flow.log_prob(x, c)

    def _preprocess_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess batch data."""
        x_bfJC = batch["pose_aa"]
        c_bfC = batch["cond"]

        # Apply pose preprocessing if enabled
        if self.preprocess_poses:
            x_bfJC = preprocess_poses(x_bfJC, apply_wrapping=True, apply_clamping=True)

        # Flatten for flow processing
        x, c = _flatten_inputs(x_bfJC, c_bfC)
        return x, c

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, c = self._preprocess_batch(batch)

        # Compute NLL loss
        log_prob = self.flow.log_prob(x, c)
        nll_loss = -log_prob.mean()

        # Log metrics
        self.log('train/nll', nll_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/bpd', nll_loss / (math.log(2.0) * x.size(1)), on_step=False, on_epoch=True)

        # Store outputs for epoch-end processing
        self.train_step_outputs.append({
            'loss': nll_loss.detach(),
            'batch_size': x.size(0)
        })

        return nll_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step with comprehensive metrics."""
        x, c = self._preprocess_batch(batch)
        batch_size = x.size(0)

        # Compute NLL
        log_prob = self.flow.log_prob(x, c)
        nll_loss = -log_prob.mean()

        # Round-trip error
        z, logdet_fwd = self.flow.fwd(x, c)
        x_reconstructed, logdet_inv = self.flow.inv(z, c)
        rt_mse = F.mse_loss(x_reconstructed, x)

        # Base distribution checks (z should be ~ N(0,I))
        z_mean_mse = z.mean(0).pow(2).mean()
        z_var_error = (z.var(0).mean() - 1.0).abs()

        # Log determinant consistency
        logdet_error = (logdet_fwd + logdet_inv).abs().mean()

        # Compute bits per dimension
        bpd = nll_loss / (math.log(2.0) * x.size(1))

        # Log all metrics
        self.log('val/nll', nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/bpd', bpd, on_step=False, on_epoch=True)
        self.log('val/round_trip_mse', rt_mse, on_step=False, on_epoch=True)
        self.log('val/z_mean_mse', z_mean_mse, on_step=False, on_epoch=True)
        self.log('val/z_var_error', z_var_error, on_step=False, on_epoch=True)
        self.log('val/logdet_error', logdet_error, on_step=False, on_epoch=True)

        # Store outputs for epoch-end processing
        output = {
            'val_loss': nll_loss.detach(),
            'val_bpd': bpd.detach(),
            'val_rt_mse': rt_mse.detach(),
            'val_z_mean_mse': z_mean_mse.detach(),
            'val_z_var_error': z_var_error.detach(),
            'batch_size': batch_size
        }
        self.val_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.train_step_outputs:
            # Compute epoch averages
            total_loss = sum(out['loss'] * out['batch_size'] for out in self.train_step_outputs)
            total_samples = sum(out['batch_size'] for out in self.train_step_outputs)
            avg_loss = total_loss / max(total_samples, 1)

            self.log('train/epoch_nll', avg_loss, prog_bar=False)

            # Clear outputs
            self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_step_outputs:
            # Compute epoch averages
            metrics = {}
            total_samples = sum(out['batch_size'] for out in self.val_step_outputs)

            for key in ['val_loss', 'val_bpd', 'val_rt_mse', 'val_z_mean_mse', 'val_z_var_error']:
                total = sum(out[key] * out['batch_size'] for out in self.val_step_outputs if key in out)
                metrics[f'{key}_epoch'] = total / max(total_samples, 1)

            # Log epoch metrics
            for key, value in metrics.items():
                self.log(key, value, prog_bar=False)

            # Clear outputs
            self.val_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=self.learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.max_epochs // 3,
                gamma=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        else:
            return optimizer

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """Configure gradient clipping."""
        if self.grad_clip_val is not None:
            self.clip_gradients(optimizer, gradient_clip_val=self.grad_clip_val, gradient_clip_algorithm="norm")

    def sample_poses(self, n: int, conditioning: torch.Tensor) -> torch.Tensor:
        """Sample poses from the trained flow."""
        return self.flow.sample(n, conditioning)


def train_global_flow_lightning(
    dim: int = 165,
    cond_dim: int = 512,
    train_loader=None,
    val_loader=None,
    hidden: int = 512,
    K: int = 6,
    use_actnorm: bool = True,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    scheduler_type: str = "cosine",
    max_epochs: int = 100,
    grad_clip_val: Optional[float] = 1.0,
    accelerator: str = "auto",
    devices: Union[int, str] = "auto",
    precision: str = "32-true",
    log_dir: str = "./logs",
    checkpoint_dir: str = "./checkpoints",
    project_name: str = "global-flow",
    experiment_name: Optional[str] = None,
    logger_type: str = "tensorboard",
    early_stop_patience: int = 10,
    preprocess_poses: bool = True,
    **trainer_kwargs
) -> Tuple[GlobalFlowLightningModule, pl.Trainer]:
    """
    Train Global Conditional Flow using PyTorch Lightning.

    Args:
        dim: Flow input dimension (e.g., 165 for SMPL-X)
        cond_dim: Conditioning dimension
        train_loader: Training data loader
        val_loader: Validation data loader
        hidden: Hidden layer size
        K: Number of coupling layers
        use_actnorm: Whether to use ActNorm
        learning_rate: Learning rate
        weight_decay: Weight decay
        scheduler_type: LR scheduler type
        max_epochs: Maximum training epochs
        grad_clip_val: Gradient clipping value
        accelerator: Lightning accelerator
        devices: Number of devices
        precision: Training precision
        log_dir: Logging directory
        checkpoint_dir: Checkpoint directory
        project_name: Project name for logging
        experiment_name: Experiment name
        logger_type: Logger type (tensorboard, wandb)
        early_stop_patience: Early stopping patience
        preprocess_poses: Whether to preprocess poses
        **trainer_kwargs: Additional trainer arguments

    Returns:
        Tuple of (trained_model, trainer)
    """
    print(f"🚀 Starting Global Flow Lightning Training")
    print(f"  Dimensions: {dim}D input, {cond_dim}D conditioning")
    print(f"  Architecture: {K} layers, {hidden} hidden, ActNorm: {use_actnorm}")
    print(f"  Training: {max_epochs} epochs, LR: {learning_rate}, Scheduler: {scheduler_type}")

    # Create model
    model = GlobalFlowLightningModule(
        dim=dim,
        cond_dim=cond_dim,
        hidden=hidden,
        K=K,
        use_actnorm=use_actnorm,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_type=scheduler_type,
        max_epochs=max_epochs,
        preprocess_poses=preprocess_poses,
        grad_clip_val=grad_clip_val
    )

    # Setup logger
    if experiment_name is None:
        experiment_name = f"flow_d{dim}_c{cond_dim}_h{hidden}_k{K}"

    if logger_type == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=log_dir,
            name=project_name,
            version=experiment_name
        )
    elif logger_type == "wandb":
        logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            save_dir=log_dir
        )
    else:
        logger = None

    # Setup callbacks
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{checkpoint_dir}/{project_name}/{experiment_name}",
        filename="flow-{epoch:02d}-{val/nll:.4f}",
        monitor="val/nll",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if early_stop_patience > 0:
        early_stopping = EarlyStopping(
            monitor="val/nll",
            patience=early_stop_patience,
            mode="min",
            min_delta=1e-4,
            verbose=True
        )
        callbacks.append(early_stopping)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=1.0,
        gradient_clip_val=grad_clip_val if grad_clip_val else None,
        gradient_clip_algorithm="norm" if grad_clip_val else None,
        enable_model_summary=True,
        enable_progress_bar=True,
        **trainer_kwargs
    )

    print(f"\n📊 Trainer Configuration:")
    print(f"  Accelerator: {accelerator}, Devices: {devices}")
    print(f"  Precision: {precision}, Gradient Clip: {grad_clip_val}")
    print(f"  Logger: {logger_type}, Checkpoints: {checkpoint_callback.dirpath}")

    # Train model
    print(f"\n🔥 Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print(f"\n🎉 Training completed!")
    print(f"  Best model path: {checkpoint_callback.best_model_path}")
    print(f"  Final validation NLL: {checkpoint_callback.best_model_score:.4f}")

    return model, trainer


def create_dummy_data_loader(batch_size: int = 8,
                           num_batches: int = 10,
                           num_joints: int = 55,
                           cond_dim: int = 512,
                           num_frames: int = 1) -> torch.utils.data.DataLoader:
    """Create dummy data loader for testing."""

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples: int):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate realistic pose data
            poses = torch.randn(num_frames, num_joints, 3) * 0.3
            poses = wrap_to_pi(poses)  # Wrap to realistic range

            # Generate conditioning
            conditioning = torch.randn(num_frames, cond_dim)

            return {
                'pose_aa': poses,  # [F, J, 3]
                'cond': conditioning  # [F, C]
            }

    dataset = DummyDataset(batch_size * num_batches)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    """Test the Lightning-based global flow trainer"""
    print("🧪 Testing Global Flow Lightning Trainer")
    print("=" * 60)

    # Create dummy data loaders
    print("\n🔧 Creating dummy data loaders...")
    train_loader = create_dummy_data_loader(batch_size=4, num_batches=8, num_frames=2)
    val_loader = create_dummy_data_loader(batch_size=4, num_batches=3, num_frames=2)

    # Test data preprocessing
    print("\n🔧 Testing data preprocessing...")
    for batch in train_loader:
        poses = batch['pose_aa']
        print(f"Original poses shape: {poses.shape}")

        processed = preprocess_poses(poses, apply_wrapping=True, apply_clamping=True)
        print(f"Processed poses shape: {processed.shape}")

        # Test flattening
        cond = batch['cond']
        x, c = _flatten_inputs(poses, cond)
        print(f"Flattened: poses {poses.shape} -> {x.shape}, cond {cond.shape} -> {c.shape}")
        break

    # Test Lightning module directly
    print("\n🔧 Testing Lightning module...")
    model = GlobalFlowLightningModule(
        dim=165,
        cond_dim=512,
        hidden=128,
        K=3,
        use_actnorm=True,
        learning_rate=1e-3,
        preprocess_poses=True
    )

    # Test forward pass
    for batch in train_loader:
        x, c = model._preprocess_batch(batch)
        log_prob = model(x, c)
        print(f"✅ Forward pass: {x.shape} -> log_prob {log_prob.shape}")
        print(f"  Log prob stats: mean={log_prob.mean().item():.3f}, std={log_prob.std().item():.3f}")
        break

    # Test Lightning training
    print(f"\n🚀 Testing Lightning training...")
    trained_model, trainer = train_global_flow_lightning(
        dim=165,
        cond_dim=512,
        train_loader=train_loader,
        val_loader=val_loader,
        hidden=128,
        K=3,
        use_actnorm=True,
        learning_rate=1e-3,
        max_epochs=3,
        grad_clip_val=1.0,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        log_dir="./test_logs",
        checkpoint_dir="./test_checkpoints",
        project_name="test-flow",
        experiment_name="test-run",
        logger_type="tensorboard",
        early_stop_patience=5,
        enable_checkpointing=True
    )

    print(f"\n🎉 Lightning training test completed!")

    # Test sampling from trained model
    print(f"\n🔧 Testing trained model sampling...")
    dummy_cond = torch.randn(1, 512)
    samples = trained_model.sample_poses(n=4, conditioning=dummy_cond)
    print(f"✅ Generated samples: {samples.shape}")
    print(f"Sample statistics: mean={samples.mean().item():.3f}, std={samples.std().item():.3f}")

    # Test model loading from checkpoint
    print(f"\n🔧 Testing checkpoint loading...")
    if hasattr(trainer.checkpoint_callback, 'best_model_path') and trainer.checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")

        # Load model from checkpoint
        loaded_model = GlobalFlowLightningModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        print(f"✅ Loaded model from checkpoint")

        # Test loaded model
        test_samples = loaded_model.sample_poses(n=2, conditioning=dummy_cond)
        print(f"✅ Loaded model sampling: {test_samples.shape}")

    print(f"\n💡 Lightning Training Benefits:")
    print(f"  • Automatic GPU/multi-GPU handling")
    print(f"  • Built-in logging (TensorBoard/W&B)")
    print(f"  • Checkpointing and model restoration")
    print(f"  • Early stopping and LR scheduling")
    print(f"  • Progress bars and monitoring")
    print(f"  • Gradient clipping and mixed precision")
    print(f"  • Easy distributed training (DDP)")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)