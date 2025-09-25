#!/usr/bin/env python3
"""
Quick test script for both Hand VAE Prior models
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from hand_vae_prior import HandVAEPrior
from hand_vae_prior_so3 import HandVAEPriorSO3


def quick_test():
    """Quick test of both models"""

    print("🧪 Quick VAE Model Tests")
    print("=" * 30)

    # Test Standard VAE
    print("\n1️⃣ Standard VAE:")
    model1 = HandVAEPrior(x_dim=90, z_dim=24, hidden=128, n_layers=2)
    x_90d = torch.randn(4, 90) * 0.3

    # Set dummy stats
    model1.set_data_stats(torch.zeros(90), torch.ones(90))

    try:
        output1 = model1(x_90d)
        loss1, _ = model1.elbo_loss(x_90d)
        energy1 = model1.energy(x_90d)

        print(f"  ✅ Forward: recon_nll={output1['recon_nll'].mean():.2f}, kl={output1['kl'].mean():.2f}")
        print(f"  ✅ Loss: {loss1:.2f}")
        print(f"  ✅ Energy: {energy1.mean():.2f}")

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

    # Test SO(3) VAE
    print("\n2️⃣ SO(3) VAE:")
    model2 = HandVAEPriorSO3(z_dim=24, hidden=128, n_layers=2)
    x_30x3 = x_90d.view(4, 30, 3)

    # Set dummy stats
    model2.set_data_stats(torch.zeros(90), torch.ones(90))

    try:
        output2 = model2(x_30x3)
        loss2, _ = model2.elbo_loss(x_30x3)
        energy2 = model2.energy(x_30x3)

        print(f"  ✅ Forward: recon_nll={output2['recon_nll'].mean():.2f}, kl={output2['kl'].mean():.2f}")
        print(f"  ✅ Loss: {loss2:.2f}")
        print(f"  ✅ Energy: {energy2.mean():.2f}")

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

    # Test DataModule format compatibility
    print("\n3️⃣ DataModule Format Test:")
    try:
        # Simulate DataModule output (90D tensor)
        batch_90d = torch.randn(8, 90) * 0.2

        # Test Standard VAE (direct use)
        out1 = model1(batch_90d)

        # Test SO(3) VAE (reshape needed)
        batch_30x3 = batch_90d.view(8, 30, 3)
        out2 = model2(batch_30x3)

        print(f"  ✅ Standard VAE: input {batch_90d.shape} -> latent {out1['z_mu'].shape}")
        print(f"  ✅ SO(3) VAE: input {batch_90d.shape} -> {batch_30x3.shape} -> latent {out2['z_mu'].shape}")

    except Exception as e:
        print(f"  ❌ Compatibility test failed: {e}")
        return False

    print("\n🎉 All quick tests passed!")
    print("\n💡 Usage Summary:")
    print("  Standard VAE: model(batch_90d)  # Direct from DataModule")
    print("  SO(3) VAE:    model(batch_90d.view(-1, 30, 3))  # Reshape first")

    return True


if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)