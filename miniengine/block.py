import torch
from torch import torch.nn as nn
from miniengine.layers import MyRMSNorm, MySWiGLU
from miniengine.attention import MyAttention

class MyDecoderLayer(nn.Module):
  def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps):
    super().__init__()
    self.input_layernorm = MyRMSNorm(hidden_size, eps)
    self.self_attn = MyAttention(hidden_size, num_heads, num_kv_heads, head_dim)
    self.post_attention_layernorm = MyRMSNorm(hidden_size, eps)
    self.mlp = MySWiGLU(hidden_size, intermediate_size)


  def forward(self, x, positions, cos, sin):
    h = x + self.self_attn(self.input_layernorm(x), positions, cos, sin)
    return h + self.mlp(self.post_attention_laternorm(h))
