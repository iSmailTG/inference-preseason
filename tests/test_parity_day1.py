
import pytest, torch
from miniengine.layers import MyRMSNorm, MySwiGLU
from tools.hf_ref import tiny_cfg, randomize_

pytestmark = pytest.mark.hf


def test_rmsnorm_vs_hf_tiny():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    cfg = tiny_cfg()
    hf = randomize_(Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps))
    mine = MyRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    mine.load_state_dict(hf.state_dict(), strict=True)
    x = torch.randn(2, 5, cfg.hidden_size)
    torch.testing.assert_close(mine(x), hf(x), rtol=1e-5, atol=1e-6)


def test_mlp_vs_hf_tiny():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    cfg = tiny_cfg()
    hf = randomize_(Qwen3MLP(cfg))
    mine = MySwiGLU(cfg.hidden_size, cfg.intermediate_size, bias=False)
    mine.load_state_dict(hf.state_dict(), strict=True)
    x = torch.randn(2, 5, cfg.hidden_size)
    torch.testing.assert_close(mine(x), hf(x), rtol=1e-4, atol=1e-5)
