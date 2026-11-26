import torch.nn as nn
from miniengine.layers import MyRMSNorm, MySwiGLU
from miniengine.attention import MyAttention


class MyDecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps):
        super().__init__()
        # the attribute names ARE the checkpoint's key names: do not rename them
        self.input_layernorm = MyRMSNorm(hidden_size, eps)                                  # reads the stream before attention
        self.self_attn = MyAttention(hidden_size, num_heads, num_kv_heads, head_dim, eps)   # q_proj k_proj v_proj o_proj q_norm k_norm
        self.post_attention_layernorm = MyRMSNorm(hidden_size, eps)                         # a SECOND norm module, its own weight
        self.mlp = MySwiGLU(hidden_size, intermediate_size)                                 # gate_proj up_proj down_proj

    def forward(self, x, positions, cos, sin):                                              # x: (B,S,D)  positions: (B,S)  cos, sin: (B,S,d)
        h = x + self.self_attn(self.input_layernorm(x), positions, cos, sin)                # (B,S,D)  read a normalized COPY, write onto the RAW x
        return h + self.mlp(self.post_attention_layernorm(h))                               # (B,S,D)  same again; the base is the RAW h
