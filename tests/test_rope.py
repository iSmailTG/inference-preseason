
import math, pytest, torch
from miniengine.rope import MyRotaryEmbedding, apply_rope


def test_hand_example_d4_first_pair():
    # theta=10000, d=4  ->  inv_freq = [1, 0.01]. At position 1 the pair (dim 0, dim 2) rotates by 1 radian.
    rope = MyRotaryEmbedding(4, 10000.0)
    cos, sin = rope(torch.tensor([[1]]), torch.float32)
    q = torch.tensor([1., 0., 0., 0.]).view(1, 1, 1, 4)
    out, _ = apply_rope(q, q.clone(), cos, sin)
    expected = torch.tensor([math.cos(1.0), 0.0, math.sin(1.0), 0.0])
    torch.testing.assert_close(out.flatten(), expected, atol=1e-5, rtol=0)


def test_hand_example_d4_second_pair():
    # The pair (dim 1, dim 3) rotates by 0.01 radian at position 1.
    rope = MyRotaryEmbedding(4, 10000.0)
    cos, sin = rope(torch.tensor([[1]]), torch.float32)
    q = torch.tensor([0., 1., 0., 0.]).view(1, 1, 1, 4)
    out, _ = apply_rope(q, q.clone(), cos, sin)
    expected = torch.tensor([0.0, math.cos(0.01), 0.0, math.sin(0.01)])
    torch.testing.assert_close(out.flatten(), expected, atol=1e-5, rtol=0)


def test_cos_sin_shape_halves_duplicated_and_dtype():
    rope = MyRotaryEmbedding(16, 1e6)
    cos, sin = rope(torch.arange(5)[None].expand(2, 5), torch.float32)
    assert cos.shape == (2, 5, 16) and sin.shape == (2, 5, 16)
    torch.testing.assert_close(cos[..., :8], cos[..., 8:])      # split-halves layout: both halves share angles
    torch.testing.assert_close(sin[..., :8], sin[..., 8:])
    c16, _ = rope(torch.arange(5)[None], torch.float16)
    assert c16.dtype == torch.float16


def test_inv_freq_is_fp32_and_not_in_state_dict():
    rope = MyRotaryEmbedding(16, 1e6)
    assert rope.inv_freq.dtype == torch.float32 and tuple(rope.inv_freq.shape) == (8,)
    assert "inv_freq" not in rope.state_dict()                  # computed, not learned: checkpoints do not contain it


def test_position_zero_is_identity():
    rope = MyRotaryEmbedding(16, 1e6)
    cos, sin = rope(torch.zeros(2, 3, dtype=torch.long), torch.float32)
    q = torch.randn(2, 4, 3, 16); k = torch.randn(2, 2, 3, 16)
    q2, k2 = apply_rope(q, k, cos, sin)
    torch.testing.assert_close(q2, q); torch.testing.assert_close(k2, k)


def test_rotation_preserves_vector_norm():
    rope = MyRotaryEmbedding(16, 1e6)
    positions = torch.tensor([[0, 3, 9, 27, 100, 513]])
    cos, sin = rope(positions, torch.float32)
    q = torch.randn(1, 4, 6, 16); k = torch.randn(1, 2, 6, 16)
    q2, k2 = apply_rope(q, k, cos, sin)
    torch.testing.assert_close(q2.norm(dim=-1), q.norm(dim=-1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k2.norm(dim=-1), k.norm(dim=-1), rtol=1e-4, atol=1e-4)


def test_matches_complex_number_reference():
    # Independent reference: treat (x[i], x[i + d/2]) as one complex number and multiply by e^{i * angle}.
    d, theta = 16, 1e6
    rope = MyRotaryEmbedding(d, theta)
    positions = torch.tensor([[0, 3, 9, 27, 100, 513]])
    cos, sin = rope(positions, torch.float32)
    x = torch.randn(1, 1, 6, d)
    out, _ = apply_rope(x, x.clone(), cos, sin)
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
    angles = positions[0].float()[:, None] * inv_freq[None, :]                       # (S, d/2)
    z = torch.complex(x[0, 0, :, : d // 2], x[0, 0, :, d // 2:]) * torch.polar(torch.ones_like(angles), angles)
    expected = torch.cat([z.real, z.imag], dim=-1)
    torch.testing.assert_close(out[0, 0], expected, rtol=1e-4, atol=1e-4)


def test_scores_depend_only_on_relative_offset():
    torch.manual_seed(0)
    d = 16
    rope = MyRotaryEmbedding(d, 10000.0)
    q = torch.randn(1, 1, 1, d); k = torch.randn(1, 1, 1, d)

    def score(m, n):
        cm, sm = rope(torch.tensor([[m]]), torch.float32)
        cn, sn = rope(torch.tensor([[n]]), torch.float32)
        qr, _ = apply_rope(q, q.clone(), cm, sm)
        _, kr = apply_rope(k.clone(), k, cn, sn)
        return (qr * kr).sum()

    torch.testing.assert_close(score(7, 3), score(107, 103), rtol=1e-3, atol=1e-3)   # same offset (4), different absolute
    spread = [score(20, 20 - off).item() for off in (0, 1, 2, 4, 8, 16)]            # different offsets give different scores
    assert max(spread) - min(spread) > 1e-2


def test_each_position_is_independent_of_its_neighbours():
    # Positions are explicit: a token's cos/sin depend on ITS position only, not on where it sits in the batch.
    rope = MyRotaryEmbedding(16, 1e6)
    p = torch.tensor([[0, 3, 9, 27, 100, 513]])
    cos, sin = rope(p, torch.float32)
    for i in range(p.shape[1]):
        c1, s1 = rope(p[:, i:i + 1], torch.float32)
        torch.testing.assert_close(cos[:, i:i + 1], c1)
        torch.testing.assert_close(sin[:, i:i + 1], s1)
