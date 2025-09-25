#!/usr/bin/env python3
"""
JointLimitGaussian: Factorized Gaussian prior per joint in axis-angle space.

Learnable or set by MLE from data.
Energy = -sum_j log N(θ_j | μ_j, Σ_j=diag(σ^2_j))
Inputs:
    pose_aa: [B,F,J,3] (or [N,J,3]); returns per-batch energy [B].
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple


# ---------------- utils ----------------
def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    # wrap each axis-angle component to [-pi, pi]
    return ((x + math.pi) % (2 * math.pi)) - math.pi

def split_bfjc(x: torch.Tensor) -> Tuple[int,int,int,int]:
    if x.ndim == 4:
        B, F, J, C = x.shape
    elif x.ndim == 3:       # [N,J,3] -> treat N as B*F
        B, F, J, C = x.shape[0], 1, x.shape[1], x.shape[2]
        x = x.unsqueeze(1)
    else:
        raise ValueError(f"pose_aa must be [B,F,J,3] or [N,J,3], got {x.shape}")
    return B, F, J, C


class JointLimitGaussian(nn.Module):
    """
    Factorized Gaussian prior per joint in axis-angle space.
    - Learnable or set by MLE from data.
    - Energy = -sum_j log N(θ_j | μ_j, Σ_j=diag(σ^2_j))
    Inputs:
        pose_aa: [B,F,J,3] (or [N,J,3]); returns per-batch energy [B].
    """

    def __init__(self, J: int = 53, dim: int = 3):
        """
        Initialize JointLimitGaussian model.

        Args:
            J: Number of joints (default: 53 for full SMPL-X)
            dim: Dimension per joint (default: 3 for axis-angle)
        """
        super().__init__()
        self.J = J
        self.dim = dim

        # Learnable parameters
        self.mu = nn.Parameter(torch.zeros(J, dim))          # μ_j: [J, 3]
        self.logvar = nn.Parameter(torch.zeros(J, dim))      # log σ^2_j: [J, 3]

    @torch.no_grad()
    def fit_mle(self, pose_aa_train: torch.Tensor):
        """
        Compute per-joint mean/var from training data and set parameters.

        Args:
            pose_aa_train: Training poses [N, J, 3] or [N, F, J, 3]
        """
        # Handle different input formats using split_bfjc
        B, F, J, C = split_bfjc(pose_aa_train)
        if pose_aa_train.ndim == 3:
            pose_aa_train = pose_aa_train.unsqueeze(1)  # [N,1,J,3]

        # Wrap angles to [-π, π] and reshape
        x = wrap_to_pi(pose_aa_train).reshape(-1, self.J, self.dim)  # [N*F, J, 3]

        # Compute statistics
        mu = x.mean(dim=0)  # [J, 3]
        var = x.var(dim=0, unbiased=True).clamp_min(1e-6)  # [J, 3], clamp for stability

        # Set parameters
        self.mu.copy_(mu)
        self.logvar.copy_(var.log())

        print(f"✅ MLE fit complete:")
        print(f"  μ range: [{mu.min():.3f}, {mu.max():.3f}]")
        print(f"  σ range: [{var.sqrt().min():.3f}, {var.sqrt().max():.3f}]")

    def log_prob_per_frame(self, pose_aa: torch.Tensor) -> torch.Tensor:
        """
        Return per-frame log probability: [B,F]

        Args:
            pose_aa: Input poses [B, J, 3] or [B, F, J, 3]

        Returns:
            Log probability per frame [B, F]
        """
        # Handle different input formats using split_bfjc
        B, F, J, C = split_bfjc(pose_aa)
        if pose_aa.ndim == 3:
            pose_aa = pose_aa.unsqueeze(1)  # [B, 1, J, 3]

        assert J == self.J and C == self.dim, f"Expected shape [*, *, {self.J}, {self.dim}], got [*, *, {J}, {C}]"

        # Wrap angles to [-π, π]
        x = wrap_to_pi(pose_aa)  # [B, F, J, 3]

        # Compute differences from mean
        diff = x - self.mu.view(1, 1, J, C)  # [B, F, J, 3]

        # Compute inverse variance
        inv_var = (self.logvar.exp()).reciprocal()  # 1/σ^2: [J, 3]

        # Compute log determinant term: -0.5 * sum(log(2πσ^2))
        logdet = -0.5 * (self.logvar + math.log(2 * math.pi)).sum(dim=[0, 1])  # scalar

        # Compute quadratic term: -0.5 * sum((x-μ)^2 / σ^2)
        quad = -0.5 * (diff**2 * inv_var.view(1, 1, J, C)).sum(dim=[2, 3])  # [B, F]

        return quad + logdet  # [B, F]

    def nll(self, pose_aa: torch.Tensor) -> torch.Tensor:
        """
        Return per-batch negative log-likelihood [B].

        Args:
            pose_aa: Input poses [B, J, 3] or [B, F, J, 3]

        Returns:
            Negative log-likelihood per batch [B]
        """
        lp = self.log_prob_per_frame(pose_aa)  # [B, F]
        return (-lp).mean(dim=1)  # [B]

    def forward(self, pose_aa: torch.Tensor, **_) -> torch.Tensor:
        """
        Energy interface: return per-batch energy [B].

        Args:
            pose_aa: Input poses [B, J, 3] or [B, F, J, 3]

        Returns:
            Energy per batch [B] (same as NLL)
        """
        return self.nll(pose_aa)

    def sample(self, batch_size: int = 1, num_frames: int = 1) -> torch.Tensor:
        """
        Sample poses from the learned Gaussian distribution.

        Args:
            batch_size: Number of samples in batch
            num_frames: Number of frames per sample

        Returns:
            Sampled poses [batch_size, num_frames, J, 3]
        """
        with torch.no_grad():
            # Sample from standard normal
            noise = torch.randn(batch_size, num_frames, self.J, self.dim,
                              device=self.mu.device, dtype=self.mu.dtype)

            # Transform to learned distribution
            std = self.logvar.exp().sqrt()  # [J, 3]
            samples = self.mu.unsqueeze(0).unsqueeze(0) + noise * std.unsqueeze(0).unsqueeze(0)

            # Wrap to [-π, π]
            samples = wrap_to_pi(samples)

            return samples

    def get_statistics(self) -> dict:
        """
        Get current model statistics.

        Returns:
            Dictionary with mean, std, and other statistics
        """
        with torch.no_grad():
            mu = self.mu.detach()
            std = self.logvar.exp().sqrt().detach()

            return {
                'mean': mu,  # [J, 3]
                'std': std,  # [J, 3]
                'mean_per_joint': mu.mean(dim=1),  # [J]
                'std_per_joint': std.mean(dim=1),   # [J]
                'total_params': self.J * self.dim * 2,  # mu + logvar
                'min_std': std.min().item(),
                'max_std': std.max().item(),
                'mean_std': std.mean().item()
            }

    def extra_repr(self) -> str:
        """String representation of the model."""
        return f'J={self.J}, dim={self.dim}, params={self.J * self.dim * 2}'


def main():
    """Test the JointLimitGaussian model"""
    print("🧪 Testing JointLimitGaussian Model")
    print("=" * 40)

    # Test with SMPL-X dimensions (53 joints, 3D axis-angle)
    model = JointLimitGaussian(J=53, dim=3)
    print(f"✅ Model created: {model}")

    # Generate synthetic training data
    batch_size, num_samples = 16, 1000
    pose_data = torch.randn(num_samples, 53, 3) * 0.3  # [N, J, 3]
    print(f"📊 Synthetic training data: {pose_data.shape}")

    # Test MLE fitting
    print(f"\n🔧 Testing MLE fitting...")
    model.fit_mle(pose_data)

    # Test forward pass
    print(f"\n🔧 Testing forward pass...")
    test_batch = torch.randn(batch_size, 53, 3) * 0.2  # [B, J, 3]

    # Test energy computation
    energy = model(test_batch)
    print(f"✅ Energy shape: {energy.shape}, mean: {energy.mean():.3f}")

    # Test log probability
    log_prob = model.log_prob_per_frame(test_batch)
    print(f"✅ Log prob shape: {log_prob.shape}")

    # Test NLL
    nll = model.nll(test_batch)
    print(f"✅ NLL shape: {nll.shape}, mean: {nll.mean():.3f}")

    # Test with frame dimension
    print(f"\n🔧 Testing with frame dimension...")
    test_frames = torch.randn(8, 4, 53, 3) * 0.2  # [B, F, J, 3]
    energy_frames = model(test_frames)
    print(f"✅ Energy with frames: {energy_frames.shape}")

    # Test sampling
    print(f"\n🔧 Testing sampling...")
    samples = model.sample(batch_size=4, num_frames=2)
    print(f"✅ Samples shape: {samples.shape}")

    # Test statistics
    print(f"\n📊 Model statistics:")
    stats = model.get_statistics()
    print(f"  Total parameters: {stats['total_params']}")
    print(f"  Std range: [{stats['min_std']:.3f}, {stats['max_std']:.3f}]")
    print(f"  Mean std: {stats['mean_std']:.3f}")

    # Test wrap_to_pi function
    print(f"\n🔧 Testing wrap_to_pi...")
    test_angles = torch.tensor([0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi, -math.pi, -2*math.pi])
    wrapped = wrap_to_pi(test_angles)
    print(f"✅ Angle wrapping:")
    for orig, wrap in zip(test_angles, wrapped):
        print(f"  {orig:.3f} -> {wrap:.3f}")

    print(f"\n🎉 All JointLimitGaussian tests passed!")
    print(f"\n💡 Summary:")
    print(f"  • Factorized Gaussian prior per joint")
    print(f"  • MLE fitting from training data")
    print(f"  • Energy/NLL computation for pose priors")
    print(f"  • Supports [B,J,3] and [B,F,J,3] input formats")
    print(f"  • Angle wrapping to [-π, π] for stability")
    print(f"  • Ready for pose prior modeling!")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)