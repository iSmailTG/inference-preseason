
import torch

def err(a, b):
    a = a.detach().float(); b = b.detach().float()
    d = (a - b).abs()
    max_abs = d.max().item()
    max_rel = (d / b.abs().clamp_min(1e-8)).max().item()   # relative error is misleading near zero
    return max_abs, max_rel

def compare(name, mine, ref, rtol=1e-4, atol=1e-5):
    if mine.shape != ref.shape:
        print(f"[FAIL] {name}: shape {tuple(mine.shape)} vs {tuple(ref.shape)}")
        return False
    ok = torch.allclose(mine.detach().float(), ref.detach().float(), rtol=rtol, atol=atol)
    max_abs, max_rel = err(mine, ref)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: shape={tuple(mine.shape)} dtype={mine.dtype} "
          f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} (rtol={rtol}, atol={atol})")
    return ok
