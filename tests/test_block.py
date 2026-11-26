
import pytest, torch
from miniengine.block import MyDecoderLayer
from miniengine.rope import MyRotaryEmbedding

H, KV, d, D, I = 4, 2, 16, 32, 96


def make(seed=0):
    torch.manual_seed(seed)
    layer = MyDecoderLayer(D, I, H, KV, d, 1e-6)
    with torch.no_grad():
        for p in layer.parameters():
            if p.ndim == 1:
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0, 0.2)
    return layer


def rope_args(B, S, positions=None):
    if positions is None:
        positions = torch.arange(S)[None].expand(B, S).contiguous()
    cos, sin = MyRotaryEmbedding(d, 1e6)(positions, torch.float32)
    return positions, cos, sin


def test_children_and_state_dict_keys():
    expected = {"input_layernorm.weight", "post_attention_layernorm.weight",
                "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
    assert set(make().state_dict()) == expected


def test_zero_output_projections_make_the_layer_the_identity():
    # If both sublayers write zero into the residual stream, the layer must return x EXACTLY.
    # This catches adding the normalized x instead of the raw x as the residual.
    layer = make(); x = torch.randn(2, 5, D); p, c, s = rope_args(2, 5)
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()
    torch.testing.assert_close(layer(x, p, c, s), x, rtol=0, atol=0)


def test_only_attention_zeroed_leaves_the_mlp_residual():
    layer = make(1); x = torch.randn(2, 5, D); p, c, s = rope_args(2, 5)
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
    expected = x + layer.mlp(layer.post_attention_layernorm(x))
    torch.testing.assert_close(layer(x, p, c, s), expected, rtol=1e-5, atol=1e-6)


def test_only_mlp_zeroed_leaves_the_attention_residual():
    layer = make(2); x = torch.randn(2, 5, D); p, c, s = rope_args(2, 5)
    with torch.no_grad():
        layer.mlp.down_proj.weight.zero_()
    expected = x + layer.self_attn(layer.input_layernorm(x), p, c, s)
    torch.testing.assert_close(layer(x, p, c, s), expected, rtol=1e-5, atol=1e-6)


def test_causality_through_the_whole_layer():
    layer = make(); x = torch.randn(1, 6, D); p, c, s = rope_args(1, 6)
    y = layer(x, p, c, s)
    x2 = x.clone(); x2[:, 4:] = torch.randn(1, 2, D)
    y2 = layer(x2, p, c, s)
    torch.testing.assert_close(y[:, :4], y2[:, :4], rtol=1e-5, atol=1e-6)


def test_batch_items_do_not_interact():
    layer = make(); x = torch.randn(3, 5, D); p, c, s = rope_args(3, 5)
    full = layer(x, p, c, s)
    p1, c1, s1 = rope_args(1, 5)
    torch.testing.assert_close(layer(x[:1], p1, c1, s1)[0], full[0], rtol=1e-5, atol=1e-6)


def test_shape_dtype_and_gradients():
    layer = make(); x = torch.randn(2, 5, D, requires_grad=True); p, c, s = rope_args(2, 5)
    y = layer(x, p, c, s)
    assert y.shape == x.shape and y.dtype == x.dtype
    y.square().mean().backward()
    assert x.grad is not None and layer.mlp.gate_proj.weight.grad is not None
    assert layer.self_attn.q_norm.weight.grad is not None
