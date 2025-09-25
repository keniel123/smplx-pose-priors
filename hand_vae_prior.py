#!/usr/bin/env python3
"""
Hand VAE Prior Model

VAE architecture for hand pose modeling with SMPL-X hand parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ---------- utils ----------
def _kl_normal_std_normal(mu, logvar):
    # KL( N(mu, var) || N(0, I) ) = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    return 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).sum(dim=-1)

def _gauss_nll(x, mu, logvar):
    # per-sample NLL for diagonal Gaussian (sum over dims)
    return 0.5 * (math.log(2*math.pi) + logvar + (x - mu)**2 / torch.exp(logvar)).sum(dim=-1)

# ---------- model ----------
class HandVAEPrior(nn.Module):
    """
    Simple VAE for SMPL-X hands (axis-angle).
    Input x: [N, 90]  (LH 15x3 ++ RH 15x3) in radians.
    Stores standardization buffers; energy() returns ELBO-based energy per sample.
    """
    def __init__(
        self,
        x_dim: int = 90,
        z_dim: int = 24,
        hidden: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        init_logvar: float = -1.5,  # decoder variance init (~0.22 std)
        eps: float = 1e-6,
    ):
        super().__init__()
        self.x_dim, self.z_dim, self.eps = x_dim, z_dim, eps

        # standardization buffers (set via set_data_stats before training/inference)
        self.register_buffer("x_mean", torch.zeros(x_dim))
        self.register_buffer("x_std", torch.ones(x_dim))

        # encoder
        enc = []
        d = x_dim
        for _ in range(n_layers):
            enc += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.encoder = nn.Sequential(*enc)
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        # decoder
        dec = []
        d = z_dim
        for _ in range(n_layers):
            dec += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.decoder = nn.Sequential(*dec)
        self.dec_mu = nn.Linear(hidden, x_dim)
        self.dec_logvar = nn.Linear(hidden, x_dim)

        # init decoder log-variance to a sensible value
        nn.init.constant_(self.dec_logvar.weight, 0.0)
        nn.init.constant_(self.dec_logvar.bias, init_logvar)

        # lightweight layernorms help a bit
        self.ln_enc = nn.LayerNorm(hidden)
        self.ln_dec = nn.LayerNorm(hidden)

    # ---- data std ----
    @torch.no_grad()
    def set_data_stats(self, mean: torch.Tensor, std: torch.Tensor):
        assert mean.shape[-1] == self.x_dim and std.shape[-1] == self.x_dim
        self.x_mean.copy_(mean)
        self.x_std.copy_(std.clamp_min(1e-6))

    def _standardize(self, x):  # x: [..., D]
        return (x - self.x_mean) / (self.x_std + self.eps)

    def _destandardize(self, x_std):
        return x_std * (self.x_std + self.eps) + self.x_mean

    # ---- VAE core ----
    def encode(self, x_std):
        h = self.encoder(x_std)
        h = self.ln_enc(h)
        mu, logvar = self.enc_mu(h), self.enc_logvar(h).clamp(min=-10.0, max=10.0)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.decoder(z)
        h = self.ln_dec(h)
        x_mu = self.dec_mu(h)
        x_logvar = self.dec_logvar(h).clamp(min=-8.0, max=4.0)  # cap variance
        return x_mu, x_logvar

    def forward(self, x):
        """
        x: [N, 90] raw axis-angle (radians)
        returns dict with elbo pieces
        """
        x_std = self._standardize(x)
        mu_z, logvar_z = self.encode(x_std)
        z = self.reparam(mu_z, logvar_z)
        x_mu, x_logvar = self.decode(z)

        recon_nll = _gauss_nll(x_std, x_mu, x_logvar)   # [N]
        kl = _kl_normal_std_normal(mu_z, logvar_z)      # [N]
        return {
            "recon_nll": recon_nll,
            "kl": kl,
            "x_mu": x_mu, "x_logvar": x_logvar,
            "z_mu": mu_z, "z_logvar": logvar_z, "z": z
        }

    # ---- energies ----
    @torch.no_grad()
    def energy(self, x, beta: float = 1.0, iwae_K: int = 0):
        """
        Energy used in hypothesis selection. Lower is better.
        - ELBO (default): recon_nll + beta * KL  (single sample)
        - IWAE (optional): tighter bound (K importance samples)
        """
        x_std = self._standardize(x)

        if iwae_K and iwae_K > 1:
            # IWAE: log p(x) ≈ log(1/K * sum_k p(x|z_k)p(z_k)/q(z_k|x))
            mu_z, logvar_z = self.encode(x_std)
            std_z = torch.exp(0.5 * logvar_z)
            N, D = x_std.size(0), self.x_dim
            log_ws = []
            for _ in range(iwae_K):
                z = mu_z + std_z * torch.randn_like(std_z)
                x_mu, x_logvar = self.decode(z)
                log_px_z = -_gauss_nll(x_std, x_mu, x_logvar)          # [N]
                log_pz   = -0.5 * (z**2 + math.log(2*math.pi)).sum(-1) # [N]
                log_qz_x = -0.5 * (((z-mu_z)**2)/ (std_z**2 + 1e-8) + logvar_z + math.log(2*math.pi)).sum(-1)
                log_ws.append(log_px_z + log_pz - log_qz_x)
            log_w = torch.stack(log_ws, dim=0)                         # [K,N]
            log_mean_w = torch.logsumexp(log_w, dim=0) - math.log(iwae_K)  # [N]
            # Energy = -log p(x) (approx)
            return -log_mean_w
        else:
            out = self.forward(x)  # single-sample ELBO
            return out["recon_nll"] + beta * out["kl"]

    # ---- training loss ----
    def elbo_loss(self, x, beta: float = 1.0):
        out = self.forward(x)
        elbo = out["recon_nll"] + beta * out["kl"]   # [N]
        return elbo.mean(), out