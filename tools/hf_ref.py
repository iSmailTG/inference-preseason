
import torch
from transformers import AutoConfig

MODEL = "Qwen/Qwen3-0.6B"

def real_cfg(attn="eager"):
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = attn          # pin it: eager and sdpa treat masks differently
    return cfg

def tiny_cfg(attn="eager"):
    # Small enough for CPU tests, and head_dim is deliberately NOT hidden_size/num_heads (16 != 32/4).
    from transformers import Qwen3Config
    kw = dict(vocab_size=128, hidden_size=32, intermediate_size=96, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
              max_position_embeddings=1024, attention_bias=False, tie_word_embeddings=True)
    try:
        cfg = Qwen3Config(rope_theta=1_000_000.0, **kw)
    except TypeError:
        cfg = Qwen3Config(**kw)
    cfg._attn_implementation = attn
    return cfg

def rope_theta(cfg):
    if getattr(cfg, "rope_theta", None) is not None:
        return float(cfg.rope_theta)
    rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    raise ValueError("cannot find rope_theta; print(cfg) and read the field yourself")

def randomize_(module, seed=0):
    # Random weights, so a missing gamma or a wrong projection cannot hide behind init values.
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in module.named_parameters():
            if p.ndim == 1:
                p.copy_(torch.empty(p.shape).uniform_(0.5, 1.5, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return module

def first(out):
    return out[0] if isinstance(out, (tuple, list)) else out

def causal_additive_mask(S, B=1, dtype=torch.float32):
    m = torch.full((S, S), float("-inf"), dtype=dtype).triu(1)
    return m[None, None].expand(B, 1, S, S).contiguous()

def hf_cos_sin(cfg, positions, dtype=torch.float32):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    rot = Qwen3RotaryEmbedding(config=cfg)
    return rot(torch.zeros(1, 1, dtype=dtype), positions)      # x only supplies dtype/device

def hf_attention(cfg, layer_idx=0, seed=0):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    return randomize_(Qwen3Attention(cfg, layer_idx), seed)

def hf_layer(cfg, layer_idx=0, seed=0):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    return randomize_(Qwen3DecoderLayer(cfg, layer_idx), seed)
