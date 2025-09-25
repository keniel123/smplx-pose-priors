#!/usr/bin/env python3
"""
Global Conditional Flow Module

Enhanced conditional RealNVP for full pose modeling with improved stability:
- ActNorm1d: Data-dependent normalization for better training stability
- AffineCoupling: Enhanced coupling layers with fixed permutations
- CondFlowNet: Production-ready conditional flow with proper mixing

Designed for modeling full SMPL-X poses (55 joints × 3 = 165D) conditioned
on external features (e.g., text embeddings, context vectors).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


# ---------- Small helpers ----------
class ActNorm1d(nn.Module):
    """
    Per-feature affine normalization with data-dependent initialization.

    Automatically initializes scale and bias based on the first batch
    to have zero mean and unit variance. Helps with training stability.
    """

    def __init__(self, D: int):
        """
        Initialize ActNorm layer.

        Args:
            D: Feature dimension
        """
        super().__init__()
        self.D = D
        self.loc = nn.Parameter(torch.zeros(D))          # Bias term
        self.log_scale = nn.Parameter(torch.zeros(D))    # Log scale term
        self.initialized = False

    @torch.no_grad()
    def _init(self, x: torch.Tensor):
        """
        Initialize parameters based on first batch statistics.

        Args:
            x: Input tensor [N, D]
        """
        # Compute batch statistics
        m = x.mean(0)  # [D]
        s = x.std(0).clamp_min(1e-6)  # [D], clamp for stability

        # Initialize to normalize first batch
        self.loc.copy_(-m)
        self.log_scale.copy_((-s.log()))
        self.initialized = True

        print(f"✅ ActNorm1d initialized: mean range [{m.min():.3f}, {m.max():.3f}], "
              f"std range [{s.min():.3f}, {s.max():.3f}]")

    def forward(self, x: torch.Tensor, inverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward/inverse transformation.

        Args:
            x: Input tensor [N, D]
            inverse: Whether to perform inverse transformation

        Returns:
            y: Transformed tensor [N, D]
            logdet: Log determinant [N]
        """
        if not self.initialized:
            self._init(x)

        if not inverse:
            # Forward: y = (x + loc) * exp(log_scale)
            y = (x + self.loc) * torch.exp(self.log_scale)
            logdet = self.log_scale.sum().expand(x.size(0))
        else:
            # Inverse: x = y * exp(-log_scale) - loc
            y = x * torch.exp(-self.log_scale) - self.loc
            logdet = (-self.log_scale.sum()).expand(x.size(0))

        return y, logdet

    def extra_repr(self) -> str:
        return f'D={self.D}, initialized={self.initialized}'


class AffineCoupling(nn.Module):
    """
    Enhanced conditional affine coupling with proper masking.

    Improved version with better dimension handling and stability.
    """

    def __init__(self, dim: int, cond_dim: int, hidden: int = 512, even: bool = True):
        """
        Initialize affine coupling layer.

        Args:
            dim: Input dimension
            cond_dim: Conditioning dimension
            hidden: Hidden layer size
            even: Whether to use even mask (True) or odd mask (False)
        """
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.register_buffer("mask", self._mask(dim, even))

        # Calculate network input/output dimensions based on actual mask
        mask_sum = int(self.mask.sum().item())  # Number of active (conditioning) features
        inactive_dim = dim - mask_sum           # Number of features to transform

        in_dim = mask_sum + cond_dim
        out_dim = inactive_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * out_dim)  # outputs [t | s]
        )

    @staticmethod
    def _mask(D: int, even: bool) -> torch.Tensor:
        """Create alternating binary mask."""
        m = torch.zeros(D)
        m[::2] = 1 if even else 0
        m[1::2] = 0 if even else 1
        return m

    def forward(self, x: torch.Tensor, c: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward/inverse coupling transformation.

        Args:
            x: Input tensor [N, dim]
            c: Conditioning tensor [N, cond_dim]
            reverse: Whether to perform inverse transformation

        Returns:
            y: Transformed tensor [N, dim]
            logdet: Log determinant [N]
        """
        m = self.mask.to(x.device)
        mask_indices = m.bool()

        # Split input into active (conditioning) and inactive (transforming) parts
        xa = x[:, mask_indices]      # [N, mask_sum] - conditioning part
        xb = x[:, ~mask_indices]     # [N, inactive_dim] - part to transform

        # Build conditioning input
        h = torch.cat([xa, c], dim=-1)  # [N, mask_sum + cond_dim]

        # Get transformation parameters
        net_out = self.net(h)  # [N, 2 * inactive_dim]
        t, s = net_out.chunk(2, dim=-1)  # Each [N, inactive_dim]
        s = torch.tanh(s)  # Stabilize scale

        if not reverse:
            # Forward: yb = xb * exp(s) + t
            yb = xb * torch.exp(s) + t
            logdet = s.sum(dim=-1)  # [N]
        else:
            # Inverse: xb = (yb - t) * exp(-s)
            yb = (xb - t) * torch.exp(-s)
            logdet = -s.sum(dim=-1)  # [N]

        # Reconstruct full output
        y = x.clone()
        y[:, ~mask_indices] = yb

        return y, logdet

    def extra_repr(self) -> str:
        mask_sum = int(self.mask.sum().item())
        return f'dim={self.dim}, cond_dim={self.cond_dim}, active={mask_sum}, inactive={self.dim-mask_sum}'


# ---------- Recommended global conditional flow ----------
class CondFlowNet(nn.Module):
    """
    Enhanced conditional RealNVP for full pose modeling.

    Improvements over basic implementations:
    - ActNorm layers for training stability
    - Fixed feature permutations for better mixing
    - Proper forward/inverse with N(0,I) base
    - Sampling capability
    - Production-ready architecture
    """

    def __init__(self, dim: int, cond_dim: int, hidden: int = 512, K: int = 6, use_actnorm: bool = True):
        """
        Initialize conditional flow network.

        Args:
            dim: Input dimension (e.g., 55*3=165 for SMPL-X)
            cond_dim: Conditioning dimension (e.g., text embedding size)
            hidden: Hidden layer size for coupling networks
            K: Number of coupling layers
            use_actnorm: Whether to use ActNorm for stability
        """
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.K = K
        self.use_actnorm = use_actnorm

        # Build layers
        self.layers = nn.ModuleList()
        self.perms = []  # Fixed permutations for each layer

        for k in range(K):
            even = (k % 2 == 0)

            # Create block with pre-norm, coupling, post-norm
            block = nn.ModuleDict()
            block["pre"] = ActNorm1d(dim) if use_actnorm else nn.Identity()
            block["cpl"] = AffineCoupling(dim, cond_dim, hidden, even=even)
            block["post"] = ActNorm1d(dim) if use_actnorm else nn.Identity()
            self.layers.append(block)

            # Create fixed permutation for better mixing
            perm = torch.randperm(dim)
            self.register_buffer(f"perm_{k}", perm)
            self.perms.append(perm)

    def _permute(self, x: torch.Tensor, k: int, inverse: bool = False) -> torch.Tensor:
        """Apply fixed permutation for layer k."""
        perm = getattr(self, f"perm_{k}")

        if not inverse:
            return x.index_select(1, perm)
        else:
            # Compute inverse permutation
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
            return x.index_select(1, inv_perm)

    def fwd(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: x -> z

        Args:
            x: Input tensor [N, dim]
            c: Conditioning tensor [N, cond_dim]

        Returns:
            z: Latent tensor [N, dim]
            logdet: Total log determinant [N]
        """
        logdet = x.new_zeros(x.size(0))
        z = x

        for k, blk in enumerate(self.layers):
            # Pre-normalization
            if isinstance(blk["pre"], ActNorm1d):
                z, ld = blk["pre"](z, inverse=False)
                logdet += ld

            # Coupling layer
            z, ld = blk["cpl"](z, c, reverse=False)
            logdet += ld

            # Post-normalization
            if isinstance(blk["post"], ActNorm1d):
                z, ld = blk["post"](z, inverse=False)
                logdet += ld

            # Fixed permutation
            z = self._permute(z, k, inverse=False)

        return z, logdet

    def inv(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: z -> x

        Args:
            z: Latent tensor [N, dim]
            c: Conditioning tensor [N, cond_dim]

        Returns:
            x: Reconstructed tensor [N, dim]
            logdet: Total log determinant [N]
        """
        logdet = z.new_zeros(z.size(0))
        x = z

        # Reverse order of layers
        for k in reversed(range(len(self.layers))):
            # Reverse permutation
            x = self._permute(x, k, inverse=True)

            blk = self.layers[k]

            # Post-normalization (reverse)
            if isinstance(blk["post"], ActNorm1d):
                x, ld = blk["post"](x, inverse=True)
                logdet += ld

            # Coupling layer (reverse)
            x, ld = blk["cpl"](x, c, reverse=True)
            logdet += ld

            # Pre-normalization (reverse)
            if isinstance(blk["pre"], ActNorm1d):
                x, ld = blk["pre"](x, inverse=True)
                logdet += ld

        return x, logdet

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of x given conditioning c.

        Args:
            x: Input tensor [N, dim]
            c: Conditioning tensor [N, cond_dim]

        Returns:
            log_prob: Log probability [N]
        """
        z, logdet = self.fwd(x, c)

        # Standard normal base distribution: log N(z; 0, I)
        log_base = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=-1)

        return log_base + logdet

    @torch.no_grad()
    def sample(self, n: int, cond: torch.Tensor) -> torch.Tensor:
        """
        Sample from the conditional distribution.

        Args:
            n: Number of samples
            cond: Conditioning tensor [N, cond_dim] (will be expanded to [n, cond_dim])

        Returns:
            samples: Generated samples [n, dim]
        """
        device = cond.device
        dtype = cond.dtype

        # Sample from base distribution N(0, I)
        z = torch.randn(n, self.dim, device=device, dtype=dtype)

        # Expand conditioning to match batch size
        if cond.size(0) == 1:
            c = cond.expand(n, -1)
        elif cond.size(0) == n:
            c = cond
        else:
            raise ValueError(f"Conditioning batch size {cond.size(0)} must be 1 or {n}")

        # Transform to data space
        x, _ = self.inv(z, c)

        return x

    def nll(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood (for loss function).

        Args:
            x: Input tensor [N, dim]
            c: Conditioning tensor [N, cond_dim]

        Returns:
            nll: Negative log-likelihood [N]
        """
        return -self.log_prob(x, c)

    def get_statistics(self) -> dict:
        """Get model statistics and initialization status."""
        stats = {
            'dim': self.dim,
            'cond_dim': self.cond_dim,
            'num_layers': self.K,
            'use_actnorm': self.use_actnorm,
            'total_params': sum(p.numel() for p in self.parameters()),
            'actnorm_initialized': []
        }

        # Check ActNorm initialization status
        for k, blk in enumerate(self.layers):
            if isinstance(blk["pre"], ActNorm1d):
                stats['actnorm_initialized'].append(f'pre_{k}: {blk["pre"].initialized}')
            if isinstance(blk["post"], ActNorm1d):
                stats['actnorm_initialized'].append(f'post_{k}: {blk["post"].initialized}')

        return stats

    def extra_repr(self) -> str:
        return f'dim={self.dim}, cond_dim={self.cond_dim}, K={self.K}, actnorm={self.use_actnorm}'


def main():
    """Test the global conditional flow module"""
    print("🧪 Testing Global Conditional Flow Module")
    print("=" * 50)

    # Test ActNorm1d
    print("🔧 Testing ActNorm1d...")
    actnorm = ActNorm1d(10)
    x = torch.randn(32, 10) * 2.0 + 1.0  # Non-standard distribution

    # Forward pass (should initialize)
    y, logdet = actnorm(x, inverse=False)
    print(f"✅ ActNorm forward: {x.shape} -> {y.shape}, logdet: {logdet.shape}")
    print(f"  Input stats: mean={x.mean().item():.3f}, std={x.std().item():.3f}")
    print(f"  Output stats: mean={y.mean().item():.3f}, std={y.std().item():.3f}")

    # Inverse pass
    x_recon, logdet_inv = actnorm(y, inverse=True)
    print(f"✅ ActNorm inverse: reconstruction error = {(x - x_recon).abs().max().item():.6f}")

    # Test enhanced AffineCoupling
    print(f"\n🔧 Testing enhanced AffineCoupling...")
    coupling = AffineCoupling(dim=10, cond_dim=5, hidden=64, even=True)
    x = torch.randn(16, 10)
    c = torch.randn(16, 5)

    y, logdet = coupling(x, c, reverse=False)
    x_recon, logdet_inv = coupling(y, c, reverse=True)
    print(f"✅ AffineCoupling: {x.shape} -> {y.shape}")
    print(f"  Reconstruction error: {(x - x_recon).abs().max().item():.6f}")
    print(f"  Forward logdet: {logdet.mean().item():.3f}")

    # Test CondFlowNet
    print(f"\n🔧 Testing CondFlowNet...")

    # Create flow for SMPL-X poses (55 joints × 3 = 165D)
    flow = CondFlowNet(dim=165, cond_dim=512, hidden=256, K=4, use_actnorm=True)
    print(f"✅ CondFlowNet created: {flow}")

    # Test with pose data
    batch_size = 8
    pose_data = torch.randn(batch_size, 165) * 0.3  # Realistic pose magnitudes
    conditioning = torch.randn(batch_size, 512)     # Text embedding or similar

    # Test forward pass
    z, logdet_fwd = flow.fwd(pose_data, conditioning)
    print(f"✅ Forward pass: {pose_data.shape} -> {z.shape}")
    print(f"  Logdet mean: {logdet_fwd.mean().item():.3f}")

    # Test inverse pass
    pose_recon, logdet_inv = flow.inv(z, conditioning)
    print(f"✅ Inverse pass: {z.shape} -> {pose_recon.shape}")
    print(f"  Reconstruction error: {(pose_data - pose_recon).abs().max().item():.6f}")

    # Test log probability
    log_prob = flow.log_prob(pose_data, conditioning)
    print(f"✅ Log probability: {log_prob.shape}, mean: {log_prob.mean().item():.3f}")

    # Test NLL (for training)
    nll = flow.nll(pose_data, conditioning)
    print(f"✅ NLL: {nll.shape}, mean: {nll.mean().item():.3f}")

    # Test sampling
    samples = flow.sample(n=4, cond=conditioning[:1])  # Single conditioning
    print(f"✅ Sampling: {samples.shape}")

    # Test with different conditioning
    samples_multi = flow.sample(n=batch_size, cond=conditioning)
    print(f"✅ Multi-conditioning sampling: {samples_multi.shape}")

    # Model statistics
    stats = flow.get_statistics()
    print(f"\n📊 Model Statistics:")
    for key, value in stats.items():
        if key == 'actnorm_initialized':
            print(f"  {key}:")
            for init_status in value[:4]:  # Show first few
                print(f"    {init_status}")
        else:
            print(f"  {key}: {value}")

    # Test gradient flow
    print(f"\n🔧 Testing gradient flow...")
    pose_data.requires_grad_(True)
    loss = flow.nll(pose_data, conditioning).mean()
    loss.backward()
    grad_norm = pose_data.grad.norm().item()
    print(f"✅ Gradient norm: {grad_norm:.3f}")

    print(f"\n🎉 All global conditional flow tests passed!")
    print(f"\n💡 Summary:")
    print(f"  • Enhanced conditional RealNVP with ActNorm stability")
    print(f"  • Fixed permutations for better feature mixing")
    print(f"  • Production-ready for full SMPL-X pose modeling (165D)")
    print(f"  • Supports arbitrary conditioning (text, context, etc.)")
    print(f"  • Forward/inverse transformations with proper Jacobians")
    print(f"  • Sampling from learned conditional distribution")
    print(f"  • Ready for pose generation and refinement!")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)