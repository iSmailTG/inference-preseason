
import torch
import torch.nn as nn


class MyRotaryEmbedding(nn.Module):
    # Contract:
    #   __init__(head_dim, theta): store inv_freq as an fp32, NON-persistent buffer, shape (head_dim // 2,)
    #   forward(positions, dtype) -> (cos, sin), each (B, S, head_dim)
    #     positions: (B, S) integer positions (explicit, never assumed to be arange)
    #     compute the angles in float32, then cast cos and sin to `dtype`
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        raise NotImplementedError("Day 3, Code I")

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        raise NotImplementedError("Day 3, Code I")


def apply_rope(q, k, cos, sin):
    # q: (B, H, S, d)   k: (B, KV, S, d)   cos, sin: (B, S, d)   ->   (q_rotated, k_rotated)
    # Pairing convention: split halves, i.e. dimension i is paired with i + d/2.
    raise NotImplementedError("Day 3, Code II")
