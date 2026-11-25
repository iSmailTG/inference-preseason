
import pytest, torch
from miniengine.rope import MyRotaryEmbedding, apply_rope
from miniengine.attention import MyAttention
from tools.hf_ref import tiny_cfg, rope_theta, hf_cos_sin, hf_attention, causal_additive_mask, first

pytestmark = pytest.mark.hf

SPARSE = torch.tensor([[0, 3, 9, 27, 100, 513]])


@pytest.mark.parametrize("positions", [torch.arange(16)[None], SPARSE])
def test_cos_sin_vs_hf(positions):
    cfg = tiny_cfg()
    rope = MyRotaryEmbedding(cfg.head_dim, rope_theta(cfg))
    cos, sin = rope(positions, torch.float32)
    hcos, hsin = hf_cos_sin(cfg, positions)
    torch.testing.assert_close(cos, hcos, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sin, hsin, rtol=1e-5, atol=1e-5)


def test_apply_rope_vs_hf():
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    cfg = tiny_cfg()
    hcos, hsin = hf_cos_sin(cfg, SPARSE)
    q = torch.randn(1, cfg.num_attention_heads, 6, cfg.head_dim)
    k = torch.randn(1, cfg.num_key_value_heads, 6, cfg.head_dim)
    hq, hk = apply_rotary_pos_emb(q, k, hcos, hsin)
    mq, mk = apply_rope(q, k, hcos, hsin)
    torch.testing.assert_close(mq, hq, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mk, hk, rtol=1e-5, atol=1e-5)


def test_full_attention_vs_hf_with_real_rope_and_sparse_positions():
    cfg = tiny_cfg("eager")
    hf = hf_attention(cfg)
    mine = MyAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                       cfg.head_dim, cfg.rms_norm_eps, bias=False)
    mine.load_state_dict(hf.state_dict(), strict=True)
    positions = torch.tensor([[0, 3, 9, 27, 100, 513, 600]])
    S = positions.shape[1]
    x = torch.randn(1, S, cfg.hidden_size)
    cos, sin = hf_cos_sin(cfg, positions)
    ref = first(hf(hidden_states=x, position_embeddings=(cos, sin), attention_mask=causal_additive_mask(S, 1)))
    out = mine(x, positions, cos, sin)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    mycos, mysin = MyRotaryEmbedding(cfg.head_dim, rope_theta(cfg))(positions, torch.float32)
    torch.testing.assert_close(mine(x, positions, mycos, mysin), ref, rtol=1e-4, atol=1e-5)
