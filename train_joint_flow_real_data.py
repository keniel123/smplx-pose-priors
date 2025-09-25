#!/usr/bin/env python3
"""
Joint Flow Trainer with Real SMPL-X Data

Connects the joint limit normalizing flow to real SMPL-X pose data
using the comprehensive data module. Handles joint count conversion
from 53 joints (data) to 55 joints (model with eye joints).
"""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
import argparse
from pathlib import Path

from joint_limit_flow import JointLimitFlow
from comprehensive_pose_datamodule import ComprehensivePoseDataModule
from config_utils import ConfigLoader, create_config_parser, override_config, print_config


# Joint conversion now handled in comprehensive_pose_datamodule.py
# Data module returns poses directly as (B, 55, 3) with eye joints as zeros


class JointFlowRealDataModule(pl.LightningModule):
    """
    Lightning module for Joint Flow with real SMPL-X data.
    """

    def __init__(
        self,
        data_dir: str,
        hidden: int = 128,
        K: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip_val: float = 1.0,
        scheduler_type: str = "cosine",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create the flow model (55 joints)
        self.flow = JointLimitFlow(J=55, cond_dim=3, hidden=hidden, K=K)

        # Store hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip_val = grad_clip_val
        self.scheduler_type = scheduler_type
        self.data_dir = data_dir

        # SMPL-X parents for 55 joints (with eye joints)
        self.parents = [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
            16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 28, 29, 45,
            46, 47, 48, 49, 50, 51, 52, 53
        ]

        # Register as buffer
        self.register_buffer('parents_tensor', torch.tensor(self.parents, dtype=torch.long))

    def training_step(self, batch, batch_idx):
        """Training step - compute NLL loss."""
        # Data module already returns (B, 55, 3) with eye joints as zeros
        pose_aa = batch  # Already in correct format

        # Compute negative log-likelihood
        nll = self.flow.nll(pose_aa, self.parents_tensor)
        loss = nll.mean()

        # Compute BPD
        bpd = loss / (math.log(2.0) * 3.0)

        # Log metrics
        self.log('train/nll', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/bpd', bpd, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - compute metrics."""
        # Data module already returns (B, 55, 3) with eye joints as zeros
        pose_aa = batch  # Already in correct format

        # Compute NLL
        nll = self.flow.nll(pose_aa, self.parents_tensor)
        val_nll = nll.mean()

        # Compute BPD
        val_bpd = val_nll / (math.log(2.0) * 3.0)

        # Log validation metrics
        self.log('val/nll', val_nll, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/bpd', val_bpd, on_step=False, on_epoch=True, prog_bar=True)

        # Test sampling on first validation batch
        if batch_idx == 0:
            try:
                with torch.no_grad():
                    # Sample from the flow
                    sample_size = min(4, pose_aa.shape[0])
                    samples = self.flow.sample_joints(self.parents, batch_size=sample_size)

                    # Log sample statistics
                    self.log('val/sample_mean', samples.mean(), on_step=False, on_epoch=True)
                    self.log('val/sample_std', samples.std(), on_step=False, on_epoch=True)
                    self.log('val/sample_max', samples.abs().max(), on_step=False, on_epoch=True)

                    # Test round-trip for first joint
                    C = self.flow.build_conditions(pose_aa[:sample_size], self.parents)
                    j = 0  # Test first joint
                    x_j = pose_aa[:sample_size, j, :]  # [sample_size, 3]
                    c_j = C[:sample_size, j, :]       # [sample_size, cond_dim]

                    z_j, _ = self.flow.flows[j].fwd(x_j, c_j)
                    reconstructed_j, _ = self.flow.flows[j].inv(z_j, c_j)

                    round_trip_error = F.mse_loss(reconstructed_j, x_j)
                    self.log('val/round_trip_mse', round_trip_error, on_step=False, on_epoch=True)

            except Exception as e:
                # Log sampling errors but don't fail training
                self.log('val/sampling_error', 1.0, on_step=False, on_epoch=True)

        return {'val_nll': val_nll, 'val_bpd': val_bpd}

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

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

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Configure gradient clipping."""
        if self.grad_clip_val > 0:
            clip_grad_norm_(self.parameters(), self.grad_clip_val)


def parse_args():
    """Parse command line arguments."""
    # Create base config parser
    config_parser = create_config_parser()

    # Add additional arguments specific to this trainer
    parser = argparse.ArgumentParser(description='Train Joint Flow with Real SMPL-X Data',
                                   parents=[config_parser])

    return parser.parse_args()


def train_joint_flow_with_real_data():
    """Main training function with real data."""
    args = parse_args()

    print("🔥 Training Joint Flow with Real SMPL-X Data")

    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)

    # Apply command line overrides
    if args.override:
        config = override_config(config, args.override)

    # Print final configuration
    print_config(config, "Joint Flow Configuration")

    # Extract configuration values
    data_dir = config['data']['data_dir']

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
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        return_dict=config['data']['return_dict']
    )

    # Setup data
    data_module.setup()

    # Create model
    print("\n🔧 Creating Lightning module...")
    model = JointFlowRealDataModule(
        data_dir=str(data_dir),
        hidden=config['model']['hidden'],
        K=config['model']['K'],
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        grad_clip_val=config['training']['grad_clip_val'],
        scheduler_type=config['training']['scheduler_type']
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create checkpoint directory
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="joint-flow-real-{epoch:02d}-{val/nll:.4f}",
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
        gradient_clip_val=config['training']['grad_clip_val'],
        deterministic=False
    )

    print(f"🚀 Starting training...")
    print(f"  Max epochs: {config['training']['max_epochs']}")
    print(f"  Learning rate: {model.lr}")
    print(f"  Weight decay: {model.weight_decay}")
    print(f"  Gradient clipping: {model.grad_clip_val}")
    print(f"  Scheduler: {model.scheduler_type}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Hidden units: {config['model']['hidden']}")
    print(f"  Flow layers: {config['model']['K']}")
    print(f"  Checkpoints: {checkpoint_dir}")

    try:
        # Train the model
        trainer.fit(model, data_module)

        # Print results
        print(f"\n🎉 Training completed!")
        if checkpoint_callback.best_model_path:
            print(f"  Best model: {checkpoint_callback.best_model_path}")
            print(f"  Best val NLL: {checkpoint_callback.best_model_score:.4f}")
            best_bpd = checkpoint_callback.best_model_score / (math.log(2.0) * 3.0)
            print(f"  Best val BPD: {best_bpd:.4f}")

        # Test sampling
        print(f"\n🔧 Testing model sampling...")
        model.eval()
        with torch.no_grad():
            samples = model.flow.sample_joints(model.parents, batch_size=4)
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
    success = train_joint_flow_with_real_data()
    exit(0 if success else 1)