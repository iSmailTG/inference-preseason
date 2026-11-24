
import pytest, torch
from miniengine.attention import MyAttention

H, KV, d, D = 4, 2, 16, 32          # head_dim (16) is deliberately NOT D/H (8), like Qwen3


def make(seed=0):
    torch.manual_seed(seed)
    a = MyAttention(D, H, KV, d, 1e-6, bias=False)
    with torch.no_grad():
        for p in a.parameters():
            if p.ndim == 1:
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0, 0.2)
    return a


def pos(B, S):
    return torch.arange(S)[None].expand(B, S).contiguous()


def naive_reference(a, x, positions):
    # Slow explicit loops: one batch item, one head, one query, one key at a time.
    B, S, _ = x.shape
    q = a.q_norm(a.q_proj(x).view(B, S, H, d))
    k = a.k_norm(a.k_proj(x).view(B, S, KV, d))
    v = a.v_proj(x).view(B, S, KV, d)
    out = torch.zeros(B, S, H, d)
    group = H // KV
    for b in range(B):
        for h in range(H):
            kv = h // group                       # query head h reads kv head h // group
            for i in range(S):
                allowed = [j for j in range(S) if positions[b, j] <= positions[b, i]]
                scores = torch.stack([(q[b, i, h] * k[b, j, kv]).sum() / d ** 0.5 for j in allowed])
                w = torch.softmax(scores, dim=0)
                out[b, i, h] = sum(w[t] * v[b, j, kv] for t, j in enumerate(allowed))
    return a.o_proj(out.reshape(B, S, H * d))


def test_state_dict_names_and_decoupled_head_dim():
    a = make()
    assert set(a.state_dict()) == {"q_proj.weight", "k_proj.weight", "v_proj.weight",
                                   "o_proj.weight", "q_norm.weight", "k_norm.weight"}
    assert tuple(a.q_proj.weight.shape) == (H * d, D)
    assert tuple(a.k_proj.weight.shape) == (KV * d, D)
    assert tuple(a.o_proj.weight.shape) == (D, H * d)


def test_output_shape_and_gradients():
    a = make()
    x = torch.randn(2, 5, D, requires_grad=True)
    y = a(x, pos(2, 5))
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None and a.q_proj.weight.grad is not None and a.q_norm.weight.grad is not None


def test_matches_naive_loop_sorted_positions():
    a = make(); x = torch.randn(2, 6, D); p = pos(2, 6)
    torch.testing.assert_close(a(x, p), naive_reference(a, x, p), rtol=1e-4, atol=1e-5)


def test_matches_naive_loop_unsorted_positions():
    # Position-based masking, not triu: here the token at index 3 has position 0 and must see only itself.
    a = make(1); x = torch.randn(1, 5, D); p = torch.tensor([[3, 1, 4, 0, 2]])
    torch.testing.assert_close(a(x, p), naive_reference(a, x, p), rtol=1e-4, atol=1e-5)


def test_causality_future_tokens_do_not_leak():
    a = make(); x = torch.randn(1, 6, D); p = pos(1, 6)
    y = a(x, p)
    x2 = x.clone(); x2[:, 4:] = torch.randn(1, 2, D)
    y2 = a(x2, p)
    torch.testing.assert_close(y[:, :4], y2[:, :4], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(y[:, 4:], y2[:, 4:])


def test_single_visible_key_gives_that_keys_value():
    # Token at index 1 has position 1; the keys at positions 3 and 2 are in its future, so it sees only itself.
    # Softmax over one key is 1, so its output is o_proj(v of itself, expanded to the query heads).
    a = make(); x = torch.randn(1, 3, D); p = torch.tensor([[3, 1, 2]])
    y = a(x, p)
    v = a.v_proj(x[:, 1]).view(1, KV, d).repeat_interleave(H // KV, dim=1)    # (1, H, d)
    expected = a.o_proj(v.reshape(1, H * d))
    torch.testing.assert_close(y[:, 1], expected, rtol=1e-4, atol=1e-5)


def test_qk_norm_makes_output_invariant_to_q_scale():
    a = make(); x = torch.randn(2, 5, D); p = pos(2, 5)
    y = a(x, p)
    with torch.no_grad():
        a.q_proj.weight.mul_(3.0)
    torch.testing.assert_close(a(x, p), y, rtol=1e-4, atol=1e-5)    # without QK-norm this would change
