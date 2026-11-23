
import pytest, torch
import torch.nn.functional as F
from miniengine.layers import MyRMSNorm, MySwiGLU


def test_rmsnorm_worked_example():
    n = MyRMSNorm(3, 1e-12)
    y = n(torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(y, torch.tensor([0.4629, 0.9258, 1.3887]), atol=1e-4, rtol=0)
    assert y.mean().item() > 0.5            # output is NOT zero-mean: nothing was centered


def test_rmsnorm_feature_scale():
    n = MyRMSNorm(3, 1e-12)
    n.weight.data = torch.tensor([2., 0.5, 1.])
    y = n(torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(y, torch.tensor([0.9258, 0.4629, 1.3887]), atol=1e-4, rtol=0)


@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 3, 8)])
def test_rmsnorm_shapes_and_unit_rms(shape):
    n = MyRMSNorm(8, 1e-12)
    x = torch.randn(*shape) + 1.0
    y = n(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    rms = y.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=0)


def test_rmsnorm_zero_input_is_finite():
    assert torch.isfinite(MyRMSNorm(8, 1e-6)(torch.zeros(2, 3, 8))).all()


@pytest.mark.skipif(not hasattr(F, "rms_norm"), reason="needs torch >= 2.4")
def test_rmsnorm_matches_torch_builtin():
    x = torch.randn(2, 5, 64)
    n = MyRMSNorm(64, 1e-6)
    n.weight.data.uniform_(0.5, 2.0)
    torch.testing.assert_close(n(x), F.rms_norm(x, (64,), n.weight, 1e-6), rtol=1e-5, atol=1e-6)


def test_rmsnorm_gradients():
    n = MyRMSNorm(8, 1e-6)
    x = torch.randn(2, 3, 8, requires_grad=True)
    n(x).square().mean().backward()
    assert x.grad is not None and n.weight.grad is not None


def _fill(m, gate, up, down):
    for lin, v in [(m.gate_proj, gate), (m.up_proj, up), (m.down_proj, down)]:
        lin.weight.data.fill_(v)


def test_swiglu_scalar_example():
    m = MySwiGLU(1, 1)
    _fill(m, 2., 3., 1.)
    out = m(torch.ones(1, 1, 1))
    torch.testing.assert_close(out, torch.tensor([[[5.2848]]]), atol=1e-3, rtol=0)   # silu(2)*3
    assert abs(out.item() - 5.7154) > 0.1                                           # silu(up)*gate would give ~5.7154


def test_swiglu_negative_gate_distinguishes_silu_from_sigmoid():
    m = MySwiGLU(1, 1)
    _fill(m, -2., 3., 1.)
    out = m(torch.ones(1, 1, 1))
    torch.testing.assert_close(out, torch.tensor([[[-0.7152]]]), atol=1e-3, rtol=0)   # sigmoid would give +0.3576


def test_swiglu_names_and_shapes():
    m = MySwiGLU(16, 48)
    sd = m.state_dict()
    assert set(sd) == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    assert tuple(sd["gate_proj.weight"].shape) == (48, 16)
    assert tuple(sd["down_proj.weight"].shape) == (16, 48)


def test_swiglu_is_per_token():
    m = MySwiGLU(16, 48)
    x = torch.randn(2, 5, 16)
    perm = torch.tensor([3, 0, 4, 1, 2])
    torch.testing.assert_close(m(x[:, perm]), m(x)[:, perm], rtol=1e-5, atol=1e-6)    # the MLP never mixes positions


def test_swiglu_gradients():
    m = MySwiGLU(8, 24)
    x = torch.randn(2, 3, 8, requires_grad=True)
    m(x).square().mean().backward()
    assert x.grad is not None and m.gate_proj.weight.grad is not None
