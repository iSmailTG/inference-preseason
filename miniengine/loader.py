
import json, os
import torch
from safetensors import safe_open
from miniengine.block import MyDecoderLayer


def load_layer_state_dict(model_dir, layer_idx):
    # Return {key_without_prefix: float32 tensor} for model.layers.{layer_idx}.*  (e.g. 'self_attn.q_proj.weight').
    # Handle BOTH a single model.safetensors and a sharded checkpoint (model.safetensors.index.json -> weight_map).
    # Read only the tensors you need (safe_open + get_tensor); never load the whole file.
    raise NotImplementedError("Day 5, Code I")


def load_layer(model_dir, cfg, layer_idx):
    # Build MyDecoderLayer from cfg, load the state dict with strict=True, return it in eval mode.
    raise NotImplementedError("Day 5, Code II")
