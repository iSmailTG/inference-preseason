
import torch
import torch.nn as nn
import torch.nn.functional as F
from miniengine.layers import MyRMSNorm


def apply_rope(q, k, cos, sin):
    # Day 2 stub: identity. On Day 3 you delete this function and import the real one from miniengine.rope.
    return q, k


class MyAttention(nn.Module):
    # Children (exact names): q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
    # forward(x, positions, cos=None, sin=None) -> (B, S, D)
    #   x: (B, S, D)      positions: (B, S) integer positions      return: (B, S, D)
    # Mask rule: key j is visible to query i  iff  positions[b, j] <= positions[b, i]
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, eps, bias=False):
        super().__init__()
        raise NotImplementedError("Day 2, Code I")

    def forward(self, x, positions, cos=None, sin=None):
        raise NotImplementedError("Day 2, Code II")
