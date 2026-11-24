
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyRMSNorm(nn.Module):
  def __init__(self, hidden_size, eps: float= 1e-6) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states -> torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

    return self.weight * hidden_states.to(input_dtype)

class MySwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
      super().__init__()
      self.hidden_size = hidden_size
      self.intermediate_size = intermediate_size
      self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
      self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
      self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
      up = self.up_proj(hidden_states)
      gated = self.gate_proj(hidden_states)
      activations = F.silu(gated) * up
      outputs = self.down_proj(activations)
      return outputs
