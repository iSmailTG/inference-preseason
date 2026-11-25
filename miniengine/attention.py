import torch
import torch.nn as nn
from miniengine.layers import MyRMSNorm
from miniengine.rope import apply_rope

class MyAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, eps, bias=False):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5                                      # head_dim, NOT hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        self.q_norm = MyRMSNorm(head_dim, eps)
        self.k_norm = MyRMSNorm(head_dim, eps)

    def forward(self, x, positions, cos=None, sin=None):
        B, S, _ = x.shape
        # project, split heads, QK-norm over head_dim (per head, before RoPE), then (B, heads, S, d)
        q = self.q_norm(self.q_proj(x).view(B, S, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)    # v: no norm, no RoPE
        if cos is not None and sin is not None:
            q, k = apply_rope(q, k, cos, sin)
        # GQA: query head h reads kv head h // group  ->  repeat_interleave (NOT repeat)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale                         # (B, H, S, S)
        visible = positions[:, None, :] <= positions[:, :, None]                           # (B, S, S): key j visible to query i
        scores = scores.masked_fill(~visible[:, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)           # softmax in fp32
        out = torch.matmul(weights, v)                                                     # (B, H, S, d)
        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)
