
import torch

def capture(module, names, **kwargs):
    store, hooks = {}, []
    mods = dict(module.named_modules())
    for n in names:
        def mk(n):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                store[n] = o.detach().clone()
            return hook
        hooks.append(mods[n].register_forward_hook(mk(n)))
    try:
        with torch.no_grad():
            out = module(**kwargs)
    finally:
        for h in hooks:
            h.remove()
    return out, store

def first_divergence(mine, ref, names, my_kwargs, ref_kwargs, rtol=1e-4, atol=1e-5):
    # Run both modules, compare the named sub-module outputs in execution order, report the first that differs.
    _, a = capture(mine, names, **my_kwargs)
    _, b = capture(ref, names, **ref_kwargs)
    for n in names:
        if a[n].shape != b[n].shape:
            print(f"SHAPE {n:28s} mine {tuple(a[n].shape)} vs ref {tuple(b[n].shape)}  (layout differs; compare by hand after transposing)")
            continue
        d = (a[n] - b[n]).abs().max().item()
        ok = torch.allclose(a[n], b[n], rtol=rtol, atol=atol)
        print(f"{'ok  ' if ok else 'DIFF'} {n:28s} max_abs={d:.3e}")
        if not ok:
            print("first divergence at:", n)
            return n
    print("no divergence at the checked points")
    return None
