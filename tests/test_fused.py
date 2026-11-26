
import torch
from miniengine.block import MyDecoderLayer
from miniengine.layers import MyRMSNorm
from miniengine.rope import MyRotaryEmbedding
from miniengine.fused import add_norm, fused_layer, run_fused

H, KV, d, D, I = 4, 2, 16, 32, 96


def make_layer(seed):
    torch.manual_seed(seed)
    layer = MyDecoderLayer(D, I, H, KV, d, 1e-6)
    with torch.no_grad():
        for p in layer.parameters():
            if p.ndim == 1:
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0, 0.2)
    return layer


def rope_args(B, S):
    positions = torch.arange(S)[None].expand(B, S).contiguous()
    cos, sin = MyRotaryEmbedding(d, 1e6)(positions, torch.float32)
    return positions, cos, sin


def test_add_norm_returns_a_normalized_copy_and_the_raw_sum():
    torch.manual_seed(0)
    norm = MyRMSNorm(8, 1e-6); norm.weight.data.uniform_(0.5, 1.5)
    x, r = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    read, stream = add_norm(norm, x, r)
    torch.testing.assert_close(stream, x + r, rtol=0, atol=0)            # the stream is NOT normalized
    torch.testing.assert_close(read, norm(x + r))                        # the copy is
    read0, stream0 = add_norm(norm, x, None)                             # first layer: nothing to add
    torch.testing.assert_close(stream0, x, rtol=0, atol=0)
    torch.testing.assert_close(read0, norm(x))


def test_one_fused_layer_returns_two_tensors_that_sum_to_the_plain_layer():
    layer = make_layer(0); torch.manual_seed(1)
    x = torch.randn(2, 5, D); p, c, s = rope_args(2, 5)
    write, stream = fused_layer(layer, x, None, p, c, s)
    torch.testing.assert_close(write + stream, layer(x, p, c, s), rtol=1e-5, atol=1e-6)


def test_the_returned_stream_is_the_middle_of_the_layer():
    # After attention has written but before the MLP has: that is h = x + attn(norm(x)).
    layer = make_layer(3); torch.manual_seed(2)
    x = torch.randn(2, 5, D); p, c, s = rope_args(2, 5)
    _, stream = fused_layer(layer, x, None, p, c, s)
    h = x + layer.self_attn(layer.input_layernorm(x), p, c, s)
    torch.testing.assert_close(stream, h, rtol=1e-5, atol=1e-6)


def test_three_layers_plus_final_norm_match_the_plain_stack():
    layers = [make_layer(i) for i in range(3)]
    final = MyRMSNorm(D, 1e-6); final.weight.data.uniform_(0.5, 1.5)
    torch.manual_seed(4)
    x = torch.randn(2, 5, D); p, c, s = rope_args(2, 5)
    plain = x
    for layer in layers:
        plain = layer(plain, p, c, s)
    plain = final(plain)
    torch.testing.assert_close(run_fused(layers, final, x, p, c, s), plain, rtol=1e-5, atol=1e-5)
