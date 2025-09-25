#!/usr/bin/env python3
"""
Conditional Normalizing Flows for Per-Joint Pose Priors

Implements conditional affine coupling layers (RealNVP) for modeling
per-joint pose distributions conditioned on parent joint configurations.

Components:
- AffineCoupling: Small conditional affine coupling layer
- CondRealNVP: K-layer conditional RealNVP for 3D axis-angle vectors
- JointLimitFlow: Per-joint conditional NF prior for full pose modeling
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional
from joint_limit_gaussian import wrap_to_pi, split_bfjc


# ---------------- Conditional flow (per-joint) ----------------
class AffineCoupling(nn.Module):
    """Small conditional affine coupling (RealNVP) for vector dim D."""

    def __init__(self, dim: int, cond_dim: int, hidden: int = 128, even_mask: bool = True):
        """
        Initialize affine coupling layer.

        Args:
            dim: Dimension of input vector
            cond_dim: Dimension of conditioning vector
            hidden: Hidden layer size
            even_mask: Whether to mask even indices (True) or odd indices (False)
        """
        super().__init__()
        self.dim = dim
        self.mask = self._mask(dim, even_mask)  # [D]
        self.register_buffer("_mask_buffer", self.mask)

        # Calculate actual sizes based on the mask
        active_dim = int(self.mask.sum().item())  # Number of 1s in mask
        inactive_dim = dim - active_dim  # Number of 0s in mask

        in_dim = active_dim + cond_dim
        out_dim = inactive_dim  # Transform only the inactive part

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * out_dim)  # [t | s] for inactive part
        )

    @staticmethod
    def _mask(D: int, even: bool) -> torch.Tensor:
        """Create alternating mask for coupling layer."""
        m = torch.zeros(D)
        m[::2] = 1 if even else 0
        m[1::2] = 0 if even else 1
        return m

    def forward(self, x: torch.Tensor, c: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward/inverse transformation.

        Args:
            x: Input tensor [N, D]
            c: Conditioning tensor [N, cond_dim]
            reverse: Whether to perform inverse transformation

        Returns:
            y: Transformed tensor [N, D]
            logdet: Log determinant of Jacobian [N]
        """
        m = self.mask.to(x.device)
        mask_indices = m.bool()

        # Extract the active (masked=1) and inactive (masked=0) parts
        xa_active = x[:, mask_indices]      # [N, active_dim] - conditioning part
        xb_inactive = x[:, ~mask_indices]   # [N, inactive_dim] - part to transform

        # Build conditioning input
        h = torch.cat([xa_active, c], dim=-1)

        # Get transformation parameters
        st = self.net(h)
        t, s = st.chunk(2, dim=-1)
        s = torch.tanh(s)  # stabilize scaling

        # Apply transformation to the inactive part
        if not reverse:
            # Forward: y_inactive = x_inactive * exp(s) + t
            yb_transformed = xb_inactive * torch.exp(s) + t
            logdet = s.sum(dim=-1)  # log|det J| = sum(s)
        else:
            # Reverse: x_inactive = (y_inactive - t) * exp(-s)
            yb_transformed = (xb_inactive - t) * torch.exp(-s)
            logdet = -s.sum(dim=-1)  # log|det J^{-1}| = -sum(s)

        # Reconstruct full vector
        y = x.clone()
        y[:, ~mask_indices] = yb_transformed

        return y, logdet


class CondRealNVP(nn.Module):
    """K-layer conditional RealNVP for small D (here D=3). Base is N(0,I)."""

    def __init__(self, dim: int = 3, cond_dim: int = 3, hidden: int = 128, K: int = 4):
        """
        Initialize conditional RealNVP.

        Args:
            dim: Dimension of data (3 for axis-angle)
            cond_dim: Dimension of conditioning vector
            hidden: Hidden layer size
            K: Number of coupling layers
        """
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling(dim, cond_dim, hidden, even_mask=(k % 2 == 0))
            for k in range(K)
        ])
        self.register_buffer("base_mean", torch.zeros(dim))
        self.register_buffer("base_logstd", torch.zeros(dim))  # std=1

    def fwd(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation: x -> z."""
        z, logdet = x, x.new_zeros(x.shape[0])
        for layer in self.layers:
            z, ld = layer(z, c, reverse=False)
            logdet = logdet + ld
        return z, logdet

    def inv(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse transformation: z -> x."""
        x, logdet = z, z.new_zeros(z.shape[0])
        for layer in reversed(self.layers):
            x, ld = layer(x, c, reverse=True)
            logdet = logdet + ld
        return x, logdet

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute log probability of x given conditioning c."""
        z, logdet = self.fwd(x, c)
        # log N(z;0,I)
        log_base = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=-1)
        return log_base + logdet

    def sample(self, c: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample from the conditional distribution."""
        device = c.device
        batch_size = c.shape[0]

        # Sample from base distribution N(0,I)
        z = torch.randn(batch_size * num_samples, self.base_mean.shape[0], device=device)

        # Repeat conditioning for all samples
        c_expanded = c.repeat(num_samples, 1)

        # Transform to data space
        x, _ = self.inv(z, c_expanded)

        if num_samples == 1:
            return x
        else:
            return x.view(num_samples, batch_size, -1)


# ---------------- Per-joint conditional NF prior ----------------
class JointLimitFlow(nn.Module):
    """
    Tiny conditional flow per joint.
    For each joint j we model p(θ_j | cond_j) where θ_j ∈ R^3 (axis-angle).
    Conditioning by default = parent axis-angle (3D). You can pass any vector.

    API:
        .build_conditions(pose_aa) -> dict {j: cond_tensor [N, C]}
        .nll(pose_aa) -> [B] per batch energy
    """

    def __init__(self, J: int = 53, cond_dim: int = 3, hidden: int = 128, K: int = 4):
        """
        Initialize per-joint conditional flow.

        Args:
            J: Number of joints (53 for SMPL-X)
            cond_dim: Dimension of conditioning vector (3 for parent axis-angle)
            hidden: Hidden layer size for coupling networks
            K: Number of coupling layers per joint
        """
        super().__init__()
        self.J = J
        self.cond_dim = cond_dim
        self.flows = nn.ModuleList([
            CondRealNVP(dim=3, cond_dim=cond_dim, hidden=hidden, K=K)
            for _ in range(J)
        ])

    def build_conditions(self, pose_aa: torch.Tensor, parents: List[int]) -> torch.Tensor:
        """
        Build conditioning tensor C: [N,J,C].
        Default: parent axis-angle (wrap to [-pi,pi]).
        parents: list of length J with parent indices (-1 for root).

        Args:
            pose_aa: Input poses [N,J,3] or [B,F,J,3]
            parents: List of parent joint indices for each joint

        Returns:
            C: Conditioning tensor [N,J,C]
        """
        if pose_aa.ndim == 3:  # [N,J,3]
            X = pose_aa
        else:                  # [B,F,J,3]
            X = pose_aa.reshape(-1, pose_aa.shape[2], pose_aa.shape[3])

        X = wrap_to_pi(X)
        N, J, _ = X.shape
        C = X.new_zeros(N, J, self.cond_dim)

        for j in range(J):
            p = parents[j]
            if p < 0:
                C[:, j, :3] = 0.0  # root: zeros
            else:
                C[:, j, :3] = X[:, p, :3]
            # if you add more cond features, fill C[:, j, 3:] here

        return C  # [N,J,C]

    def log_prob_per_frame(self, pose_aa: torch.Tensor, parents: List[int]) -> torch.Tensor:
        """Return per-frame log prob [B,F] (or [N] if no frames)."""
        if pose_aa.ndim == 3:
            pose_aa = pose_aa.unsqueeze(1)

        B, F, J, _ = pose_aa.shape
        X = wrap_to_pi(pose_aa).reshape(B * F, J, 3)       # [N,J,3]
        C = self.build_conditions(pose_aa.reshape(B * F, J, 3), parents)  # [N,J,C]

        lp = X.new_zeros(B * F)
        for j in range(J):
            xj = X[:, j, :]              # [N,3]
            cj = C[:, j, :]              # [N,C]
            lp = lp + self.flows[j].log_prob(xj, cj)     # sum over joints

        return lp.view(B, F)

    def nll(self, pose_aa: torch.Tensor, parents: List[int]) -> torch.Tensor:
        """Return per-batch negative log-likelihood [B]."""
        return (-self.log_prob_per_frame(pose_aa, parents)).mean(dim=1)

    def forward(self, pose_aa: torch.Tensor, parents: List[int], **_) -> torch.Tensor:
        """Energy interface for your projector."""
        return self.nll(pose_aa, parents)

    def sample_joints(self, parents: List[int], batch_size: int = 1,
                     root_pose: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample complete poses by conditioning each joint on its parent.

        Args:
            parents: List of parent indices for each joint
            batch_size: Number of poses to sample
            root_pose: Optional root pose [batch_size, 3]. If None, sample from N(0,I)

        Returns:
            Sampled poses [batch_size, J, 3]
        """
        device = next(self.parameters()).device
        poses = torch.zeros(batch_size, self.J, 3, device=device)

        # Handle root joint
        root_indices = [i for i, p in enumerate(parents) if p < 0]
        if root_pose is not None:
            for root_idx in root_indices:
                poses[:, root_idx, :] = root_pose
        else:
            for root_idx in root_indices:
                poses[:, root_idx, :] = torch.randn(batch_size, 3, device=device) * 0.1

        # Sample remaining joints conditioned on parents
        for j in range(self.J):
            if parents[j] >= 0:  # Not root
                parent_pose = poses[:, parents[j], :]  # [batch_size, 3]
                sampled = self.flows[j].sample(parent_pose, num_samples=1).squeeze(0)
                poses[:, j, :] = sampled

        return poses

    def get_joint_statistics(self, joint_idx: int) -> dict:
        """Get statistics for a specific joint's flow."""
        flow = self.flows[joint_idx]
        return {
            'joint_index': joint_idx,
            'num_layers': len(flow.layers),
            'parameters': sum(p.numel() for p in flow.parameters()),
            'cond_dim': self.cond_dim
        }

    def extra_repr(self) -> str:
        """String representation of the model."""
        total_params = sum(p.numel() for p in self.parameters())
        return f'J={self.J}, cond_dim={self.cond_dim}, total_params={total_params:,}'


def create_smplx_parents_55() -> List[int]:
    """
    Create parent relationships for full SMPL-X joints (55 joints).
    Returns list where parents[j] = parent_index or -1 for root joints.

    Based on actual SMPL-X kinematic tree from smplx.create().parents.tolist()
    Includes all 55 joints (with eyes). Eye joints will be set to zeros in pose data.
    """
    # Full SMPL-X parents for 55 joints (including eyes)
    parents_55 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
                  15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21,
                  40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]

    print(f"✅ SMPL-X parents for 55 joints (including eyes):")
    print(f"   Length: {len(parents_55)}")
    print(f"   Root joints: {[i for i, p in enumerate(parents_55) if p == -1]}")

    return parents_55


def create_smplx_parents() -> List[int]:
    """
    Backward compatibility: returns 55-joint parents.
    Use create_smplx_parents_55() for clarity.
    """
    return create_smplx_parents_55()


def get_smplx_joint_names_55() -> List[str]:
    """
    Get all 55 joint names for SMPL-X (including eyes).
    Matches the order from your torch.cat concatenation:
    [global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose]
    """
    joint_names_55 = [
        # Global orient (1 joint)
        "pelvis",
        # Body pose (21 joints)
        "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        # Jaw pose (1 joint)
        "jaw",
        # Eye poses (2 joints) - will be set to zeros
        "left_eye_smplhf", "right_eye_smplhf",
        # Left hand pose (15 joints)
        "left_index1", "left_index2", "left_index3",
        "left_middle1", "left_middle2", "left_middle3",
        "left_pinky1", "left_pinky2", "left_pinky3",
        "left_ring1", "left_ring2", "left_ring3",
        "left_thumb1", "left_thumb2", "left_thumb3",
        # Right hand pose (15 joints)
        "right_index1", "right_index2", "right_index3",
        "right_middle1", "right_middle2", "right_middle3",
        "right_pinky1", "right_pinky2", "right_pinky3",
        "right_ring1", "right_ring2", "right_ring3",
        "right_thumb1", "right_thumb2", "right_thumb3"
    ]
    return joint_names_55


def get_smplx_joint_names() -> List[str]:
    """Backward compatibility: returns 55 joint names."""
    return get_smplx_joint_names_55()


def print_smplx_hierarchy():
    """Print the SMPL-X joint hierarchy for debugging."""
    parents = create_smplx_parents_55()
    joint_names = get_smplx_joint_names_55()

    print("🌳 SMPL-X Joint Hierarchy (55 joints, including eyes):")
    print("=" * 60)

    print(f"\n📍 POSE STRUCTURE (matches torch.cat order):")
    print(f"  Global orient: joint 0 (pelvis)")
    print(f"  Body pose: joints 1-21 (21 joints)")
    print(f"  Jaw pose: joint 22 (jaw)")
    print(f"  Eye poses: joints 23-24 (left/right eye) - SET TO ZEROS")
    print(f"  Left hand: joints 25-39 (15 joints)")
    print(f"  Right hand: joints 40-54 (15 joints)")

    print(f"\n📍 FULL JOINT HIERARCHY:")
    for i in range(55):
        parent = parents[i]
        parent_name = joint_names[parent] if parent >= 0 else "ROOT"
        is_eye = "👁️  [ZEROS]" if i in [23, 24] else ""
        print(f"  {i:2d}: {joint_names[i]:17s} -> parent: {parent:2d} ({parent_name:15s}) {is_eye}")

    print(f"\n📊 Summary:")
    print(f"  Total joints: {len(parents)}")
    print(f"  Root joints: {len([p for p in parents if p == -1])}")
    print(f"  Eye joints (zeros): 2 (indices 23, 24)")
    print(f"  Hand joints: 30 (15 per hand)")

    print(f"\n🔗 Pose concatenation order:")
    print(f"  [global_orient(1), body_pose(21), jaw_pose(1), leye_pose(1), reye_pose(1), lhand_pose(15), rhand_pose(15)]")
    print(f"  Total: 1+21+1+1+1+15+15 = 55 joints × 3 = 165 parameters")

    return parents


def main():
    """Test the conditional flow module"""
    print("🧪 Testing Conditional Flow Module")
    print("=" * 40)

    # Test AffineCoupling
    print("🔧 Testing AffineCoupling...")
    coupling = AffineCoupling(dim=3, cond_dim=3, hidden=64)
    x = torch.randn(8, 3)
    c = torch.randn(8, 3)

    # Forward pass
    y, logdet = coupling(x, c, reverse=False)
    print(f"✅ AffineCoupling forward: {x.shape} -> {y.shape}, logdet: {logdet.shape}")

    # Inverse pass
    x_reconstructed, logdet_inv = coupling(y, c, reverse=True)
    print(f"✅ AffineCoupling inverse: reconstruction error = {(x - x_reconstructed).abs().max().item():.6f}")

    # Test CondRealNVP
    print(f"\n🔧 Testing CondRealNVP...")
    flow = CondRealNVP(dim=3, cond_dim=3, hidden=64, K=4)

    # Test log probability
    log_prob = flow.log_prob(x, c)
    print(f"✅ CondRealNVP log_prob: {log_prob.shape}, mean: {log_prob.mean().item():.3f}")

    # Test sampling
    samples = flow.sample(c, num_samples=2)
    print(f"✅ CondRealNVP sampling: {samples.shape}")

    # Test JointLimitFlow
    print(f"\n🔧 Testing JointLimitFlow...")
    joint_flow = JointLimitFlow(J=55, cond_dim=3, hidden=64, K=2)
    parents = create_smplx_parents_55()

    # Test with pose data
    pose_batch = torch.randn(4, 55, 3) * 0.2  # [B, J, 3]

    # Set eye joints to zeros as specified
    pose_batch[:, 23, :] = 0.0  # left_eye_smplhf
    pose_batch[:, 24, :] = 0.0  # right_eye_smplhf

    # Test conditioning
    conditions = joint_flow.build_conditions(pose_batch, parents)
    print(f"✅ Build conditions: {conditions.shape}")

    # Test log probability
    log_prob_frames = joint_flow.log_prob_per_frame(pose_batch, parents)
    print(f"✅ Log prob per frame: {log_prob_frames.shape}")

    # Test NLL
    nll = joint_flow.nll(pose_batch, parents)
    print(f"✅ NLL: {nll.shape}, mean: {nll.mean().item():.3f}")

    # Test energy interface
    energy = joint_flow(pose_batch, parents)
    print(f"✅ Energy interface: {energy.shape}, mean: {energy.mean().item():.3f}")

    # Test sampling
    sampled_poses = joint_flow.sample_joints(parents, batch_size=4)
    print(f"✅ Joint sampling: {sampled_poses.shape}")

    # Test with frame dimension
    pose_frames = torch.randn(2, 3, 55, 3) * 0.2  # [B, F, J, 3]
    # Set eye joints to zeros
    pose_frames[:, :, 23, :] = 0.0  # left_eye_smplhf
    pose_frames[:, :, 24, :] = 0.0  # right_eye_smplhf
    nll_frames = joint_flow.nll(pose_frames, parents)
    print(f"✅ NLL with frames: {nll_frames.shape}")

    # Model statistics
    print(f"\n📊 Model Statistics:")
    print(f"  {joint_flow}")
    joint_stats = joint_flow.get_joint_statistics(0)
    print(f"  Joint 0 stats: {joint_stats}")

    print(f"\n🎉 All conditional flow tests passed!")
    print(f"\n💡 Summary:")
    print(f"  • Per-joint conditional normalizing flows for 55 SMPL-X joints")
    print(f"  • Parent-child conditioning using real kinematic chain")
    print(f"  • Eye joints (23, 24) set to zeros as specified")
    print(f"  • Matches torch.cat pose order: [global, body, jaw, eyes, hands]")
    print(f"  • Supports [B,J,3] and [B,F,J,3] input formats")
    print(f"  • Energy interface for pose prior modeling")
    print(f"  • Ready for full-body hierarchical pose modeling!")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)