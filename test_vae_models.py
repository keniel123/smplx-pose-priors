#!/usr/bin/env python3
"""
Test script for both Hand VAE Prior models

Tests both standard VAE and SO(3) VAE models with the DataModule to ensure
compatibility and proper forward pass functionality.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from hand_vae_prior import HandVAEPrior
from hand_vae_prior_so3 import HandVAEPriorSO3
from hand_vae_datamodule import HandVAEDataModule


def test_standard_vae():
    """Test the standard HandVAEPrior with 90D flattened input"""

    print("🧪 Testing Standard Hand VAE Prior")
    print("=" * 40)

    # Create model
    model = HandVAEPrior(
        x_dim=90,
        z_dim=24,
        hidden=256,
        n_layers=3,
        dropout=0.1
    )

    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test with random data
    batch_size = 16
    x_90d = torch.randn(batch_size, 90)  # Random 90D hand poses

    print(f"📊 Input shape: {x_90d.shape}")

    # Set dummy standardization stats
    mean = torch.zeros(90)
    std = torch.ones(90)
    model.set_data_stats(mean, std)

    print("✅ Data stats set")

    # Test forward pass
    try:
        output = model(x_90d)
        print(f"✅ Forward pass successful")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  recon_nll shape: {output['recon_nll'].shape}")
        print(f"  kl shape: {output['kl'].shape}")
        print(f"  z_mu shape: {output['z_mu'].shape}")

        # Test loss computation
        loss, _ = model.elbo_loss(x_90d, beta=1.0)
        print(f"✅ Loss computation: {loss.item():.4f}")

        # Test energy function
        with torch.no_grad():
            energy = model.energy(x_90d, beta=1.0)
        print(f"✅ Energy computation: shape {energy.shape}, mean {energy.mean().item():.4f}")

        return True

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def test_so3_vae():
    """Test the SO(3) HandVAEPriorSO3 with 30x3 reshaped input"""

    print("\n🧪 Testing SO(3) Hand VAE Prior")
    print("=" * 40)

    # Create model
    model = HandVAEPriorSO3(
        z_dim=24,
        hidden=256,
        n_layers=3,
        dropout=0.1,
        free_bits=0.0
    )

    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test with random axis-angle data
    batch_size = 16
    aa_30x3 = torch.randn(batch_size, 30, 3) * 0.5  # Reasonable axis-angle magnitudes

    print(f"📊 Input shape: {aa_30x3.shape}")

    # Set dummy standardization stats (for flattened version)
    mean = torch.zeros(90)
    std = torch.ones(90)
    model.set_data_stats(mean, std)

    print("✅ Data stats set")

    # Test forward pass
    try:
        output = model(aa_30x3)
        print(f"✅ Forward pass successful")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  recon_nll shape: {output['recon_nll'].shape}")
        print(f"  kl shape: {output['kl'].shape}")
        print(f"  z_mu shape: {output['z_mu'].shape}")

        # Test loss computation
        loss, _ = model.elbo_loss(aa_30x3, beta=1.0)
        print(f"✅ Loss computation: {loss.item():.4f}")

        # Test energy function
        with torch.no_grad():
            energy = model.energy(aa_30x3, beta=1.0)
        print(f"✅ Energy computation: shape {energy.shape}, mean {energy.mean().item():.4f}")

        return True

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def test_datamodule_compatibility():
    """Test both models with actual DataModule data"""

    print("\n🧪 Testing DataModule Compatibility")
    print("=" * 40)

    # Create DataModule
    try:
        dm = HandVAEDataModule(
            data_dir="../data",
            splits_dir="test_dataset_splits",
            batch_size=8,
            return_dict=False,  # Get just 90D tensors
            standardize=True
        )

        dm.prepare_data()
        dm.setup()
        print("✅ DataModule setup successful")

    except Exception as e:
        print(f"❌ DataModule setup failed: {e}")
        return False

    # Get a real batch
    try:
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        print(f"✅ Real batch loaded: {batch.shape}")

    except Exception as e:
        print(f"❌ Batch loading failed: {e}")
        return False

    # Test Standard VAE with real data
    print("\n🔧 Testing Standard VAE with real data...")
    model1 = HandVAEPrior(x_dim=90, z_dim=24, hidden=128, n_layers=2)

    # Use real data stats if available
    if dm.stats is not None:
        model1.set_data_stats(
            torch.from_numpy(dm.stats['mean']),
            torch.from_numpy(dm.stats['std'])
        )
        print("✅ Real data stats applied to Standard VAE")

    try:
        output1 = model1(batch)
        loss1, _ = model1.elbo_loss(batch)
        print(f"✅ Standard VAE with real data: loss = {loss1.item():.4f}")

    except Exception as e:
        print(f"❌ Standard VAE with real data failed: {e}")
        return False

    # Test SO(3) VAE with reshaped real data
    print("\n🔧 Testing SO(3) VAE with real data...")
    model2 = HandVAEPriorSO3(z_dim=24, hidden=128, n_layers=2)

    # Use real data stats if available
    if dm.stats is not None:
        model2.set_data_stats(
            torch.from_numpy(dm.stats['mean']),
            torch.from_numpy(dm.stats['std'])
        )
        print("✅ Real data stats applied to SO(3) VAE")

    try:
        # Reshape from (B, 90) to (B, 30, 3) for SO(3) model
        batch_30x3 = batch.view(batch.size(0), 30, 3)
        print(f"  Reshaped batch: {batch.shape} -> {batch_30x3.shape}")

        output2 = model2(batch_30x3)
        loss2, _ = model2.elbo_loss(batch_30x3)
        print(f"✅ SO(3) VAE with real data: loss = {loss2.item():.4f}")

    except Exception as e:
        print(f"❌ SO(3) VAE with real data failed: {e}")
        return False

    return True


def test_model_comparison():
    """Compare both models on the same data"""

    print("\n🧪 Model Comparison")
    print("=" * 40)

    # Create identical data in both formats
    batch_size = 8
    x_90d = torch.randn(batch_size, 90) * 0.3  # Reasonable hand pose values
    x_30x3 = x_90d.view(batch_size, 30, 3)

    # Create models
    model1 = HandVAEPrior(x_dim=90, z_dim=16, hidden=128, n_layers=2)
    model2 = HandVAEPriorSO3(z_dim=16, hidden=128, n_layers=2)

    # Set same stats
    mean = torch.zeros(90)
    std = torch.ones(90)
    model1.set_data_stats(mean, std)
    model2.set_data_stats(mean, std)

    # Compare outputs
    print(f"📊 Test data shapes: {x_90d.shape} vs {x_30x3.shape}")

    try:
        # Standard VAE
        out1 = model1(x_90d)
        loss1, _ = model1.elbo_loss(x_90d)
        energy1 = model1.energy(x_90d)

        # SO(3) VAE
        out2 = model2(x_30x3)
        loss2, _ = model2.elbo_loss(x_30x3)
        energy2 = model2.energy(x_30x3)

        print(f"✅ Both models work on same data:")
        print(f"  Standard VAE: loss={loss1.item():.4f}, energy_mean={energy1.mean().item():.4f}")
        print(f"  SO(3) VAE:    loss={loss2.item():.4f}, energy_mean={energy2.mean().item():.4f}")
        print(f"  Latent dims:  {out1['z_mu'].shape[1]} vs {out2['z_mu'].shape[1]}")

        return True

    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        return False


def main():
    """Run all tests"""

    print("🚀 Testing Hand VAE Prior Models")
    print("=" * 50)

    results = []

    # Test each component
    results.append(("Standard VAE", test_standard_vae()))
    results.append(("SO(3) VAE", test_so3_vae()))
    results.append(("DataModule Compatibility", test_datamodule_compatibility()))
    results.append(("Model Comparison", test_model_comparison()))

    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! Both VAE models are ready for training.")
        print("\n💡 Usage notes:")
        print("  - Standard VAE: Use with 90D flattened DataModule output")
        print("  - SO(3) VAE: Reshape 90D to (B, 30, 3) before feeding to model")
        print("  - Both models expect standardized input data")
        print("  - Set data stats with model.set_data_stats(mean, std) before training")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())