#!/usr/bin/env python3
"""
PyTorch Lightning Training Script for Hand VAE Prior Models

Supports both Standard VAE and SO(3) VAE with comprehensive logging,
checkpointing, and performance visualization.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

# Import our models and datamodules
from hand_vae_prior import HandVAEPrior
from hand_vae_prior_so3 import HandVAEPriorSO3
from hand_vae_datamodule import HandVAEDataModule
from config_utils import (
    ConfigLoader,
    create_config_parser,
    override_config,
    print_config,
)


class HandVAELightningModule(pl.LightningModule):
    """PyTorch Lightning module for training Hand VAE models"""

    def __init__(
        self,
        model_type: str = "standard",  # "standard" or "so3"
        x_dim: int = 90,
        z_dim: int = 24,
        hidden: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        kl_warmup_epochs: int = 10,
        beta_max: float = 1.0,
        free_bits: float = 0.0,  # For SO(3) model
        grad_clip_val: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model based on type
        if model_type == "standard":
            self.model = HandVAEPrior(
                x_dim=x_dim,
                z_dim=z_dim,
                hidden=hidden,
                n_layers=n_layers,
                dropout=dropout,
            )
        elif model_type == "so3":
            self.model = HandVAEPriorSO3(
                z_dim=z_dim,
                hidden=hidden,
                n_layers=n_layers,
                dropout=dropout,
                free_bits=free_bits,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model_type = model_type

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kl_warmup_epochs = kl_warmup_epochs
        self.beta_max = beta_max
        self.grad_clip_val = grad_clip_val

        # Metrics storage for visualization
        self.train_metrics = []
        self.val_metrics = []

        # Best model tracking
        self.best_val_loss = float("inf")

    def setup(self, stage: Optional[str] = None):
        """Setup data statistics from datamodule"""
        if stage == "fit" and hasattr(self.trainer.datamodule, "stats"):
            stats = self.trainer.datamodule.stats
            if stats is not None:
                mean = torch.from_numpy(stats["mean"])
                std = torch.from_numpy(stats["std"])
                self.model.set_data_stats(mean, std)
                self.log("setup/data_stats_set", 1.0)
                print(
                    f"✅ Set data stats: mean range [{mean.min():.3f}, {mean.max():.3f}], std range [{std.min():.3f}, {std.max():.3f}]"
                )

    def _get_beta(self) -> float:
        """Compute KL annealing schedule"""
        if self.kl_warmup_epochs <= 0:
            return self.beta_max

        progress = min(1.0, self.current_epoch / self.kl_warmup_epochs)
        return self.beta_max * progress

    def _prepare_input(self, batch):
        """Prepare input based on model type"""
        if self.model_type == "standard":
            # Standard VAE expects (B, 90)
            return batch
        elif self.model_type == "so3":
            # SO(3) VAE expects (B, 30, 3)
            return batch.view(batch.size(0), 30, 3)

    def training_step(self, batch, batch_idx):
        """Training step"""
        x = self._prepare_input(batch)
        beta = self._get_beta()

        # Forward pass
        loss, output = self.model.elbo_loss(x, beta=beta)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/recon_nll", output["recon_nll"].mean(), on_step=False, on_epoch=True
        )
        self.log("train/kl", output["kl"].mean(), on_step=False, on_epoch=True)
        self.log("train/beta", beta, on_step=False, on_epoch=True, prog_bar=True)

        # Additional metrics for SO(3) model
        if self.model_type == "so3":
            # Log geodesic distances if available
            with torch.no_grad():
                if hasattr(self.model, "log_sigma"):
                    sigma = torch.exp(self.model.log_sigma)
                    self.log("train/recon_sigma", sigma, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x = self._prepare_input(batch)

        # Full beta for validation
        loss, output = self.model.elbo_loss(x, beta=self.beta_max)

        # Compute energy for hypothesis selection
        with torch.no_grad():
            energy = self.model.energy(x, beta=self.beta_max)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/recon_nll", output["recon_nll"].mean(), on_step=False, on_epoch=True
        )
        self.log("val/kl", output["kl"].mean(), on_step=False, on_epoch=True)
        self.log("val/energy_mean", energy.mean(), on_step=False, on_epoch=True)
        self.log("val/energy_std", energy.std(), on_step=False, on_epoch=True)

        return {
            "val_loss": loss,
            "recon_nll": output["recon_nll"],
            "kl": output["kl"],
            "energy": energy,
            "z_mu": output["z_mu"],
            "z_logvar": output.get("z_logvar", None),
        }

    def on_validation_epoch_end(self):
        """Create visualizations at end of validation epoch"""
        if self.current_epoch % 5 == 0:  # Every 5 epochs
            try:
                self._create_validation_plots()
            except Exception as e:
                print(f"Warning: Could not create validation plots: {e}")

    def test_step(self, batch, batch_idx):
        """Test step with comprehensive evaluation"""
        x = self._prepare_input(batch)

        # Test with multiple beta values
        results = {}
        for beta in [0.1, 0.5, 1.0, 2.0]:
            loss, output = self.model.elbo_loss(x, beta=beta)
            energy = self.model.energy(x, beta=beta)

            results[f"test/loss_beta_{beta}"] = loss
            results[f"test/energy_mean_beta_{beta}"] = energy.mean()
            results[f"test/recon_nll_beta_{beta}"] = output["recon_nll"].mean()
            results[f"test/kl_beta_{beta}"] = output["kl"].mean()

        # Log all metrics
        for key, value in results.items():
            self.log(key, value, on_step=False, on_epoch=True)

        return results

    def on_test_epoch_end(self):
        """Create comprehensive test visualizations"""
        self._create_test_plots()

    def _create_validation_plots(self):
        """Create validation plots for monitoring training"""
        try:
            # Get a batch from validation set
            val_loader = self.trainer.datamodule.val_dataloader()
            batch = next(iter(val_loader))
            if batch.device != self.device:
                batch = batch.to(self.device)

            x = self._prepare_input(batch[:100])  # Use first 100 samples

            with torch.no_grad():
                # Get model outputs
                output = self.model(x)
                energy = self.model.energy(x)

                # Create figure with subplots
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f"Validation Metrics - Epoch {self.current_epoch}")

                # 1. Energy distribution
                axes[0, 0].hist(energy.cpu().numpy(), bins=50, alpha=0.7)
                axes[0, 0].set_title("Energy Distribution")
                axes[0, 0].set_xlabel("Energy")
                axes[0, 0].set_ylabel("Count")

                # 2. Reconstruction NLL
                recon_nll = output["recon_nll"].cpu().numpy()
                axes[0, 1].hist(recon_nll, bins=50, alpha=0.7, color="orange")
                axes[0, 1].set_title("Reconstruction NLL")
                axes[0, 1].set_xlabel("NLL")
                axes[0, 1].set_ylabel("Count")

                # 3. KL Divergence
                kl = output["kl"].cpu().numpy()
                axes[0, 2].hist(kl, bins=50, alpha=0.7, color="green")
                axes[0, 2].set_title("KL Divergence")
                axes[0, 2].set_xlabel("KL")
                axes[0, 2].set_ylabel("Count")

                # 4. Latent space (first 2 dimensions)
                z_mu = output["z_mu"].cpu().numpy()
                axes[1, 0].scatter(z_mu[:, 0], z_mu[:, 1], alpha=0.6, s=1)
                axes[1, 0].set_title("Latent Space (dims 0,1)")
                axes[1, 0].set_xlabel("z_0")
                axes[1, 0].set_ylabel("z_1")

                # 5. Latent variance
                if "z_logvar" in output and output["z_logvar"] is not None:
                    z_var = torch.exp(output["z_logvar"]).cpu().numpy()
                    axes[1, 1].boxplot(
                        [z_var[:, i] for i in range(min(8, z_var.shape[1]))],
                        labels=[f"z_{i}" for i in range(min(8, z_var.shape[1]))],
                    )
                    axes[1, 1].set_title("Latent Variances (first 8 dims)")
                    axes[1, 1].set_ylabel("Variance")
                else:
                    axes[1, 1].text(
                        0.5, 0.5, "No variance info", ha="center", va="center"
                    )
                    axes[1, 1].set_title("Latent Variances")

                # 6. Loss components over batches
                axes[1, 2].bar(
                    ["Recon NLL", "KL", "Total"],
                    [recon_nll.mean(), kl.mean(), (recon_nll + kl).mean()],
                )
                axes[1, 2].set_title("Loss Components")
                axes[1, 2].set_ylabel("Value")

                plt.tight_layout()

                # Log to tensorboard if available
                if self.logger and hasattr(self.logger, "experiment"):
                    self.logger.experiment.add_figure(
                        "validation_plots", fig, self.current_epoch
                    )

                plt.close(fig)

        except Exception as e:
            print(f"Warning: Could not create validation plots: {e}")

    def _create_test_plots(self):
        """Create comprehensive test plots"""
        try:
            # Get test data
            test_loader = self.trainer.datamodule.test_dataloader()
            all_batches = []
            for batch in test_loader:
                if len(all_batches) < 5:  # Limit to first 5 batches
                    all_batches.append(batch.to(self.device))

            if not all_batches:
                return

            test_data = torch.cat(all_batches, dim=0)
            x = self._prepare_input(test_data)

            with torch.no_grad():
                # Test with different beta values
                betas = [0.1, 0.5, 1.0, 2.0]
                results = {}

                for beta in betas:
                    output = self.model(x)
                    loss, _ = self.model.elbo_loss(x, beta=beta)
                    energy = self.model.energy(x, beta=beta)

                    results[beta] = {
                        "loss": loss.item(),
                        "energy": energy.cpu().numpy(),
                        "recon_nll": output["recon_nll"].cpu().numpy(),
                        "kl": output["kl"].cpu().numpy(),
                        "z_mu": output["z_mu"].cpu().numpy(),
                    }

                # Create comprehensive test figure
                fig, axes = plt.subplots(3, 3, figsize=(18, 15))
                fig.suptitle(f"Test Results - {self.model_type.upper()} VAE")

                # Energy distributions for different betas
                for i, beta in enumerate(betas):
                    row, col = i // 2, (i % 2)
                    if row < 2 and col < 2:
                        axes[row, col].hist(results[beta]["energy"], bins=50, alpha=0.7)
                        axes[row, col].set_title(f"Energy Distribution (β={beta})")
                        axes[row, col].set_xlabel("Energy")

                # Beta comparison
                beta_energies = [results[beta]["energy"].mean() for beta in betas]
                axes[0, 2].plot(betas, beta_energies, "o-")
                axes[0, 2].set_title("Mean Energy vs Beta")
                axes[0, 2].set_xlabel("Beta")
                axes[0, 2].set_ylabel("Mean Energy")

                # Reconstruction quality
                axes[1, 2].boxplot(
                    [results[beta]["recon_nll"] for beta in betas], labels=betas
                )
                axes[1, 2].set_title("Reconstruction NLL vs Beta")
                axes[1, 2].set_xlabel("Beta")
                axes[1, 2].set_ylabel("Recon NLL")

                # Latent space analysis
                z_mu = results[1.0]["z_mu"]  # Use beta=1.0
                axes[2, 0].scatter(z_mu[:, 0], z_mu[:, 1], alpha=0.6, s=1)
                axes[2, 0].set_title("Latent Space (dims 0,1)")
                axes[2, 0].set_xlabel("z_0")
                axes[2, 0].set_ylabel("z_1")

                # Latent statistics
                z_means = z_mu.mean(axis=0)
                z_stds = z_mu.std(axis=0)
                dims = range(min(16, len(z_means)))

                axes[2, 1].bar(dims, z_means[: len(dims)])
                axes[2, 1].set_title("Latent Dimension Means")
                axes[2, 1].set_xlabel("Dimension")
                axes[2, 1].set_ylabel("Mean")

                axes[2, 2].bar(dims, z_stds[: len(dims)])
                axes[2, 2].set_title("Latent Dimension Stds")
                axes[2, 2].set_xlabel("Dimension")
                axes[2, 2].set_ylabel("Std")

                plt.tight_layout()

                # Save test plots
                save_path = (
                    Path(self.logger.log_dir) / "test_plots.png"
                    if self.logger
                    else Path("test_plots.png")
                )
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"📊 Test plots saved to {save_path}")

                # Log to tensorboard if available
                if self.logger and hasattr(self.logger, "experiment"):
                    self.logger.experiment.add_figure("test_plots", fig, 0)

                plt.close(fig)

        except Exception as e:
            print(f"Warning: Could not create test plots: {e}")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.learning_rate * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def main():
    """Main training function"""
    # Create base config parser
    config_parser = create_config_parser()

    # Add additional arguments specific to this trainer
    parser = argparse.ArgumentParser(
        description="Train Hand VAE Prior", parents=[config_parser]
    )

    args = parser.parse_args()

    print("🚀 Training Hand VAE Prior with YAML Configuration")

    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)

    # Apply command line overrides
    if args.override:
        config = override_config(config, args.override)

    # Print final configuration
    print_config(config, "Hand VAE Configuration")

    # Set experiment name
    experiment_name = config["logging"].get("experiment_name")
    if experiment_name is None:
        experiment_name = f"{config['model']['type']}_z{config['model']['z_dim']}_h{config['model']['hidden']}"

    print(f"📊 Experiment: {experiment_name}")

    print(f"📁 Using multi-file data from {config['data']['splits_dir']}")
    datamodule = HandVAEDataModule(
        data_dir=config["data"]["data_dir"],
        splits_dir=config["data"]["splits_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        return_dict=False,
        standardize=True,
    )

    # Create model
    model = HandVAELightningModule(
        model_type=config["model"]["type"],
        z_dim=config["model"]["z_dim"],
        hidden=config["model"]["hidden"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        kl_warmup_epochs=config["training"]["kl_warmup_epochs"],
        beta_max=config["training"]["beta_max"],
        free_bits=config["training"]["free_bits"],
    )

    # Setup logger
    if config["logging"]["logger"] == "tensorboard":
        logger = TensorBoardLogger(
            save_dir="logs",
            name=config["logging"]["project_name"],
            version=experiment_name,
        )
    elif config["logging"]["logger"] == "wandb":
        logger = WandbLogger(
            project=config["logging"]["project_name"], name=experiment_name
        )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config["logging"]["checkpoint_dir"]) / experiment_name,
        filename="hand_vae_{epoch:02d}_{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=config["logging"]["save_top_k"],
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss", patience=20, mode="min", min_delta=1e-4
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["hardware"]["accelerator"],
        devices=config["hardware"]["devices"],
        precision=config["hardware"]["precision"],
        strategy=config["hardware"]["strategy"],
        num_nodes=config["hardware"]["num_nodes"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    print("🔥 Starting training...")
    trainer.fit(model, datamodule)

    # Test
    print("🧪 Running test evaluation...")
    trainer.test(model, datamodule)

    print("✅ Training completed!")
    print(f"💾 Best model saved in: {checkpoint_callback.dirpath}")
    print(f"📊 Logs saved in: {logger.log_dir}")


if __name__ == "__main__":
    main()
