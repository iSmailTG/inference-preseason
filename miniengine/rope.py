
import torch
import torch.nn as nn


class MyRotaryEmbedding(nn.Module):
  def __init__(self, head_dim: int, theta: float):
    super().__init__()
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, torch.float32) / head_dim))
    self.register_buffer('inv_freq', inv_freq, persistent=False)

  @torch.no_grad()
  def forward(self, positions: torch.Tensor, dtype: torch.dtype):
    angles = positions.to(torch.float32)[..., None] * self.inv_freq
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def _rotate_half(x):
  half = x.shape[-1] // 2
  return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def apply_rope(q, k, cos, sin):
  cos = cos.unsqueeze(1)
  sin = sin.unsqueeze(1)
  return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
