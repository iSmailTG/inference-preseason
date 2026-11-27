
import pytest, torch
from huggingface_hub import snapshot_download
from tools.hf_ref import real_cfg, rope_theta
from tools.real_check import capture_hf_layers, metrics
from miniengine.loader import load_layer
from miniengine.rope import MyRotaryEmbedding

pytestmark = [pytest.mark.hf, pytest.mark.slow]      # downloads the checkpoint: run on demand, not in CI

NAME = "Qwen/Qwen3-0.6B"


@pytest.mark.parametrize("layer_idx", [0, 13, 27])
def test_layer_matches_hf_on_real_prompt(layer_idx):
    model_dir = snapshot_download(NAME, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    cfg = real_cfg()
    _, inputs, store = capture_hf_layers(NAME, "The capital of France is", [layer_idx])
    x = store[layer_idx]["x"]
    S = x.shape[1]
    positions = torch.arange(S)[None]
    cos, sin = MyRotaryEmbedding(cfg.head_dim, rope_theta(cfg))(positions, torch.float32)
    mine = load_layer(model_dir, cfg, layer_idx)
    with torch.no_grad():
        out = mine(x, positions, cos, sin)
    m = metrics(out, store[layer_idx]["y"])
    assert m["rel_l2"] < 1e-5 and m["max_abs_over_scale"] < 1e-4, m
