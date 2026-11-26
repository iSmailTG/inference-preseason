import torch
import torch.nn as nn
from miniengine.layers import MyRMSNorm, MySwiGLU
from miniengine.attention import MyAttention

class MyDecoderLayer(nn.Module):
  def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps):
    super().__init__()
    self.input_layernorm = MyRMSNorm(hidden_size, eps)
    self.self_attn = MyAttention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
    self.post_attention_layernorm  = MyRMSNorm(hidden_size, eps)
    self.mlp = MySwiGLU(hidden_size, intermediate_size)


  def forward(self, x, positions, cos, sin):
    h = x + self.self_attn(self.input_layernorm(x), positions, cos, sin)
    return h + self.mlp(self.post_attention_layernorm(h))

  def forward_vllm(self, x, residual, positions, cos, sin):

    if residual is None:
      stream = x
    else:
      stream = x + residual

    attn_write = self.self_attn(self.input_layernorm(stream), positions, cos, sin)      
    h = stream + attn_write

    mlp_write = self.mlp(self.post_attention_layernorm(h))
    return mlp_write, h
