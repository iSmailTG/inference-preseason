
import pytest, torch
from miniengine.block import MyDecoderLayer
from tools.hf_ref import tiny_cfg, hf_layer, hf_cos_sin, causal_additive_mask, first

pytestmark = pytest.mark.hf


def build(cfg):
    hf = hf_layer(cfg)
    mine = MyDecoderLayer(cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads,
                          cfg.num_key_value_heads, cfg.head_dim, cfg.rms_norm_eps)
    mine.load_state_dict(hf.state_dict(), strict=True)
    return hf, mine


@pytest.mark.parametrize("positions", [torch.arange(9)[None].expand(2, 9).contiguous(),
                                       torch.tensor([[0, 3, 9, 27, 100, 513, 600]])])
def test_layer_vs_hf(positions):
    cfg = tiny_cfg("eager")
    hf, mine = build(cfg)
    B, S = positions.shape
    x = torch.randn(B, S, cfg.hidden_size)
    cos, sin = hf_cos_sin(cfg, positions)
    ref = first(hf(hidden_states=x, attention_mask=causal_additive_mask(S, B), position_ids=positions,
                   position_embeddings=(cos, sin)))
    torch.testing.assert_close(mine(x, positions, cos, sin), ref, rtol=1e-4, atol=1e-5)
