
import pytest, torch
from miniengine.attention import MyAttention
from tools.hf_ref import tiny_cfg, hf_attention, causal_additive_mask, first

pytestmark = pytest.mark.hf


def build(cfg):
    hf = hf_attention(cfg)
    mine = MyAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                       cfg.head_dim, cfg.rms_norm_eps, bias=False)
    mine.load_state_dict(hf.state_dict(), strict=True)
    return hf, mine


def test_attention_vs_hf_identity_rope():
    cfg = tiny_cfg("eager")
    hf, mine = build(cfg)
    B, S = 2, 7
    x = torch.randn(B, S, cfg.hidden_size)
    positions = torch.arange(S)[None].expand(B, S).contiguous()
    cos = torch.ones(B, S, cfg.head_dim); sin = torch.zeros(B, S, cfg.head_dim)      # RoPE = identity
    ref = first(hf(hidden_states=x, position_embeddings=(cos, sin), attention_mask=causal_additive_mask(S, B)))
    torch.testing.assert_close(mine(x, positions, cos, sin), ref, rtol=1e-4, atol=1e-5)
