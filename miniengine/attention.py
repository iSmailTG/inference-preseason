
import torch
import torch.nn as nn
import torch.nn.functional as F
from miniengine.layers import MyRMSNorm

class MyAttention(nn.Module):
  def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, eps, bias=False):
    super().__init__()

    assert num_heads % num_kv_heads == 0

    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim

    self.num_kv_groups = num_heads // num_kv_heads
    self.scale = head_dim ** -0.5

    self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
    self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
    self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)

    self.q_norm = MyRMSNorm(head_dim, eps)
    self.k_norm = MyRMSNorm(head_dim, eps)


  def forward(self, x, positions, cos, sin):

    B, S, _ = x.shape

    q = self.q_norm(self.q_proj(x).view(B, S, self.num_heads, self.head_dim)).transpose(1, 2)
    k = self.k_norm(self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)).transpose(1, 2)
    v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

    if cos is not None and sin is not None:
      q, k = apply_rope(q, k, cos, sin)

    k = k.repeat_interleave(self.num_kv_groups, dim=1)      
    v = v.repeat_interleave(self.num_kv_groups, dim=1)

    scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    invisible = positions[:, None, :] <= positions[:, :, None]
    scores = scores.masked_fill(~invisible[:, None], float('-inf'))

    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

    return self.o_proj(out)
