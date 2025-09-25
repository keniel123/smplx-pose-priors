#!/usr/bin/env python3
"""
Hand VAE Prior Model with SO(3) Representations

VAE architecture for hand pose modeling using SO(3) rotation representations
"""

# hand_vae_prior_so3.py
import math, torch, torch.nn as nn, torch.nn.functional as F

# ---------- SO(3) helpers ----------
def rot6d_to_mat(rot6d):  # [B,J,6] -> [B,J,3,3]
    a1 = F.normalize(rot6d[..., 0:3], dim=-1)
    a2 = rot6d[..., 3:6]
    b2 = F.normalize(a2 - (a1 * a2).sum(dim=-1, keepdim=True) * a1, dim=-1)
    b3 = torch.cross(a1, b2, dim=-1)
    return torch.stack([a1, b2, b3], dim=-2)  # (...,3,3)

def aa_to_mat(aa):       # [B,J,3] -> [B,J,3,3]
    # Rodrigues with safe small-angle handling
    theta = torch.clamp(aa.norm(dim=-1, keepdim=True), min=1e-8)
    k = aa / theta
    kx, ky, kz = k[...,0:1], k[...,1:2], k[...,2:3]
    K = torch.zeros(aa.shape[:-1] + (3,3), device=aa.device, dtype=aa.dtype)
    K[...,0,1], K[...,0,2] = -kz.squeeze(-1), ky.squeeze(-1)
    K[...,1,0], K[...,1,2] =  kz.squeeze(-1), -kx.squeeze(-1)
    K[...,2,0], K[...,2,1] = -ky.squeeze(-1), kx.squeeze(-1)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s, c = torch.sin(theta)[...,0], torch.cos(theta)[...,0]
    s = s.unsqueeze(-1).unsqueeze(-1); c = c.unsqueeze(-1).unsqueeze(-1)
    return I + s*K + (1-c) * (K @ K)

def geodesic_distance(R_pred, R_gt):  # [B,J,3,3] each -> [B,J]
    # d(R1,R2) = arccos((trace(R1^T R2)-1)/2)
    M = torch.matmul(R_pred.transpose(-1,-2), R_gt)
    trace = M[...,0,0] + M[...,1,1] + M[...,2,2]
    cos = (trace - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(cos)

# ---------- VAE bits ----------
def _kl_normal_std_normal(mu, logvar):  # [B,Z] -> [B]
    return 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).sum(dim=-1)

class HandVAEPriorSO3(nn.Module):
    """
    VPoser-style hand prior:
      - Input : axis-angle (LH+RH) [B, 30, 3]
      - Encode: flatten + MLP -> q(z|x)
      - Decode: z -> 6D reps -> rotation matrices
      - Recon: geodesic loss on SO(3) (optionally Gaussian in Lie algebra)
      - Energy(x): ELBO = recon_nll + beta * KL  (>= 0; lower is better)
    """
    def __init__(self, z_dim=24, hidden=256, n_layers=3, dropout=0.1,
                 learn_log_sigma=True, free_bits=0.0):
        super().__init__()
        self.J = 30
        self.x_dim = self.J * 3
        self.z_dim = z_dim
        self.free_bits = float(free_bits)  # per-latent nat floor (0.0 to ~0.05)

        # stats for encoder input only (do NOT standardize angles for recon)
        self.register_buffer("x_mean", torch.zeros(self.x_dim))
        self.register_buffer("x_std", torch.ones(self.x_dim))

        # encoder
        d, layers = self.x_dim, []
        layers += [nn.LayerNorm(d)]
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.encoder = nn.Sequential(*layers)
        self.enc_mu     = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        # decoder -> 6D per joint
        d, dlayers = z_dim, []
        for _ in range(n_layers):
            dlayers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.decoder = nn.Sequential(*dlayers)
        self.dec_6d   = nn.Linear(hidden, self.J * 6)

        # single global recon noise (in radians) for Gaussian on geodesic distance
        if learn_log_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(-1.5))  # ~0.22 rad
        else:
            self.register_buffer("log_sigma", torch.tensor(-1.5))

    # ---------- utils ----------
    @torch.no_grad()
    def set_data_stats(self, mean, std):
        self.x_mean.copy_(mean)
        self.x_std.copy_(std.clamp_min(1e-6))

    def _std(self, x):   # encoder-only standardization
        return (x - self.x_mean) / (self.x_std + 1e-6)

    # ---------- core ----------
    def encode(self, x_flat):
        h = self.encoder(x_flat)
        mu, logvar = self.enc_mu(h), self.enc_logvar(h).clamp(-8, 8)
        return mu, logvar

    def decode_to_R(self, z):  # -> rotation matrices per joint
        h = self.decoder(z)
        rot6d = self.dec_6d(h).view(-1, self.J, 6)
        R = rot6d_to_mat(rot6d)  # [B,J,3,3]
        return R

    def forward(self, aa):  # aa: [B,30,3] radians
        B = aa.size(0)
        x_flat = aa.view(B, -1)
        x_std = self._std(x_flat)
        mu_z, logvar_z = self.encode(x_std)
        z = mu_z + torch.exp(0.5*logvar_z) * torch.randn_like(mu_z)

        R_pred = self.decode_to_R(z)
        R_gt   = aa_to_mat(aa)

        # geodesic distances per joint (radians)
        d = geodesic_distance(R_pred, R_gt)  # [B,J]

        # Gaussian NLL on SO(3) distances (isotropic around gt), summed over joints
        sigma2 = torch.exp(2.0 * self.log_sigma)
        recon_nll = 0.5 * (d**2 / (sigma2 + 1e-8)).sum(dim=-1) \
                    + 0.5 * self.J * (2.0 * self.log_sigma + math.log(2*math.pi))  # [B]

        # Free-bits KL to avoid collapse
        kl_dim = 0.5*(torch.exp(logvar_z)+mu_z**2-1.0-logvar_z)
        if self.free_bits > 0:
            kl = torch.clamp(kl_dim, min=self.free_bits).sum(-1)
        else:
            kl = kl_dim.sum(-1)

        return {"recon_nll": recon_nll, "kl": kl, "z_mu": mu_z, "z_logvar": logvar_z}

    # public APIs
    def elbo_loss(self, aa, beta=1.0):
        out = self.forward(aa)
        elbo = out["recon_nll"] + beta * out["kl"]
        return elbo.mean(), out

    @torch.no_grad()
    def energy(self, aa, beta=1.0):
        out = self.forward(aa)
        return out["recon_nll"] + beta * out["kl"]