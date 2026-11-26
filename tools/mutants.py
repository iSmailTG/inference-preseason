
"""Eight classic assembly bugs for MyDecoderLayer, applied in memory (no file edits, nothing to undo).

    with mutant(3):            # patches MyDecoderLayer inside the block, restores it afterwards
        ...
    report(3)                  # runs the layer tests in a fresh process with bug 3 and prints which tests notice
"""
import json, subprocess, sys
from contextlib import contextmanager

import torch.nn as nn
from miniengine.layers import MyRMSNorm, MySwiGLU
from miniengine.attention import MyAttention


def _m1(self, x, positions, cos, sin):          # the residual adds the NORMALIZED copy
    n = self.input_layernorm(x)
    h = n + self.self_attn(n, positions, cos, sin)
    return h + self.mlp(self.post_attention_layernorm(h))


def _m2(self, x, positions, cos, sin):          # post-norm: normalize AFTER each sublayer
    h = self.input_layernorm(x + self.self_attn(x, positions, cos, sin))
    return self.post_attention_layernorm(h + self.mlp(h))


def _m3(self, x, positions, cos, sin):          # the second residual is forgotten
    h = x + self.self_attn(self.input_layernorm(x), positions, cos, sin)
    return self.mlp(self.post_attention_layernorm(h))


def _m4(self, x, positions, cos, sin):          # the MLP reads the FIRST norm
    h = x + self.self_attn(self.input_layernorm(x), positions, cos, sin)
    return h + self.mlp(self.input_layernorm(h))


def _m5_init(self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps):
    nn.Module.__init__(self)                    # ONE norm module, used twice
    self.input_layernorm = MyRMSNorm(hidden_size, eps)
    self.self_attn = MyAttention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
    self.mlp = MySwiGLU(hidden_size, intermediate_size)


def _m5(self, x, positions, cos, sin):
    h = x + self.self_attn(self.input_layernorm(x), positions, cos, sin)
    return h + self.mlp(self.input_layernorm(h))


def _m6(self, x, positions, cos, sin):          # cos and sin are never passed down: RoPE is silently skipped
    h = x + self.self_attn(self.input_layernorm(x), positions)
    return h + self.mlp(self.post_attention_layernorm(h))


def _m7(self, x, positions, cos, sin):          # attention reads the RAW stream, not a normalized copy
    n = self.input_layernorm(x)                 # (computed, then ignored)
    h = x + self.self_attn(x, positions, cos, sin)
    return h + self.mlp(self.post_attention_layernorm(h))


def _m8(self, x, positions, cos, sin):          # cos and sin passed in the wrong order
    h = x + self.self_attn(self.input_layernorm(x), positions, sin, cos)
    return h + self.mlp(self.post_attention_layernorm(h))


MUTANTS = {
    1: ("residual adds the normalized copy", {"forward": _m1}),
    2: ("post-norm: normalize after each sublayer", {"forward": _m2}),
    3: ("second residual forgotten", {"forward": _m3}),
    4: ("MLP reads input_layernorm instead of post_attention_layernorm", {"forward": _m4}),
    5: ("one norm module for both sublayers", {"__init__": _m5_init, "forward": _m5}),
    6: ("cos and sin not passed to attention (RoPE skipped)", {"forward": _m6}),
    7: ("attention reads the raw stream", {"forward": _m7}),
    8: ("cos and sin passed in the wrong order", {"forward": _m8}),
}


@contextmanager
def mutant(n):
    import miniengine.block as B
    cls = B.MyDecoderLayer
    patch = MUTANTS[n][1]
    saved = {k: cls.__dict__[k] for k in patch}
    for k, f in patch.items():
        setattr(cls, k, f)
    try:
        yield
    finally:
        for k, f in saved.items():
            setattr(cls, k, f)


FILES = ["tests/test_block.py", "tests/test_parity_day4.py"]


def _child(n):                                   # runs inside the fresh process
    import pytest
    results = {}

    class Collect:
        def pytest_runtest_logreport(self, report):
            if report.when == "call" or (report.when == "setup" and report.failed):
                results[report.nodeid.split("::", 1)[1]] = report.outcome

    with mutant(n):
        pytest.main(["-q", "--tb=no", "-p", "no:cacheprovider", "-p", "no:randomly", *FILES], plugins=[Collect()])
    print("RESULT_JSON " + json.dumps(results))


def report(n):
    title = MUTANTS[n][0]
    r = subprocess.run([sys.executable, "-m", "tools.mutants", str(n)], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
    if line is None:
        print("the test run crashed:\n", (r.stdout + r.stderr)[-1500:]); return None
    res = json.loads(line[len("RESULT_JSON "):])
    failed = [k for k, v in res.items() if v != "passed"]
    oracle = [k for k in failed if not k.startswith("test_layer_vs_hf")]
    parity = [k for k in failed if k.startswith("test_layer_vs_hf")]
    identity = any(k.startswith("test_zero_output_projections") for k in oracle)
    print(f"BUG {n}: {title}")
    print(f"  zero-write identity test : {'RED' if identity else 'green (it did NOT notice)'}")
    print(f"  oracle tests that notice : {len(oracle)} of {sum(1 for k in res if not k.startswith('test_layer_vs_hf'))}")
    for k in oracle: print("      red:", k)
    print(f"  Hugging Face parity      : {'RED (' + str(len(parity)) + ' of ' + str(sum(1 for k in res if k.startswith('test_layer_vs_hf'))) + ')' if parity else 'green: it did NOT notice'}")
    return {"oracle": oracle, "parity": parity, "identity": identity}


if __name__ == "__main__":
    _child(int(sys.argv[1]))
