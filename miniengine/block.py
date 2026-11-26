
import torch.nn as nn
from miniengine.layers import MyRMSNorm, MySwiGLU
from miniengine.attention import MyAttention


class MyDecoderLayer(nn.Module):
    # Contract: children named EXACTLY input_layernorm, self_attn, post_attention_layernorm, mlp.
    # forward(x, positions, cos, sin) -> (B, S, D)
    #   h = x + self_attn(input_layernorm(x));   y = h + mlp(post_attention_layernorm(h))
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps):
        super().__init__()
        raise NotImplementedError("Day 4, Block 3")

    def forward(self, x, positions, cos, sin):
        raise NotImplementedError("Day 4, Block 3")
