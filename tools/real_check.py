
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf(model_name):
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()


def capture_hf_layers(model_name, prompt, layer_ids):
    # Run the REAL Hugging Face model once; record, for each requested layer, its actual input,
    # the (cos, sin) it was handed, and its actual output.
    tok = AutoTokenizer.from_pretrained(model_name)
    model = load_hf(model_name)
    inputs = tok(prompt, return_tensors="pt")
    store, hooks = {}, []
    for i in layer_ids:
        layer = model.model.layers[i]

        def pre(m, args, kwargs, i=i):
            hs = args[0] if len(args) > 0 else kwargs["hidden_states"]
            pe = kwargs.get("position_embeddings")
            store[i] = {"x": hs.detach().clone(),
                        "pe": tuple(t.detach().clone() for t in pe) if pe is not None else None}

        def post(m, args, output, i=i):
            out = output[0] if isinstance(output, tuple) else output
            store[i]["y"] = out.detach().clone()

        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(layer.register_forward_hook(post))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return model, inputs, store


def metrics(mine, ref):
    d = (mine - ref).abs()
    return {"max_abs": d.max().item(),
            "max_abs_over_scale": d.max().item() / ref.abs().max().item(),
            "rel_l2": ((mine - ref).norm() / ref.norm()).item()}
