#!/usr/bin/env python3
"""
Debug the coupling layer dimensions
"""
import torch
from joint_limit_flow import AffineCoupling

# Test dimensions
dim = 3
cond_dim = 3

print(f"Dim={dim}, cond_dim={cond_dim}")

# Test even mask
coupling_even = AffineCoupling(dim=dim, cond_dim=cond_dim, hidden=64, even_mask=True)
print(f"Even mask: {coupling_even.mask}")

# Test odd mask
coupling_odd = AffineCoupling(dim=dim, cond_dim=cond_dim, hidden=64, even_mask=False)
print(f"Odd mask: {coupling_odd.mask}")

# Test network input/output sizes
print(f"\nEven mask network:")
for i, layer in enumerate(coupling_even.net):
    if hasattr(layer, 'in_features'):
        print(f"  Layer {i}: {layer.in_features} -> {layer.out_features}")

print(f"\nOdd mask network:")
for i, layer in enumerate(coupling_odd.net):
    if hasattr(layer, 'in_features'):
        print(f"  Layer {i}: {layer.in_features} -> {layer.out_features}")

# Test actual forward pass
x = torch.randn(8, 3)
c = torch.randn(8, 3)

print(f"\nInput shapes: x={x.shape}, c={c.shape}")

# Test even coupling
print(f"\nEven coupling:")
mask = coupling_even.mask.bool()
xa_active = x[:, mask]
print(f"  Active part shape: {xa_active.shape}")
print(f"  Active part size: {xa_active.shape[1]}")
print(f"  Conditioning size: {c.shape[1]}")
print(f"  Network input size: {xa_active.shape[1] + c.shape[1]}")

# Test odd coupling
print(f"\nOdd coupling:")
mask = coupling_odd.mask.bool()
xa_active = x[:, mask]
print(f"  Active part shape: {xa_active.shape}")
print(f"  Active part size: {xa_active.shape[1]}")
print(f"  Conditioning size: {c.shape[1]}")
print(f"  Network input size: {xa_active.shape[1] + c.shape[1]}")