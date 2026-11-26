
# Contract (the vLLM way): a layer takes and returns TWO tensors, (write, stream), and never does its own last add.
#   add_norm(norm, x, residual)                              -> (normalized_copy, stream)   residual may be None (first layer)
#   fused_layer(layer, x, residual, positions, cos, sin)     -> (write, stream)             uses the children of MyDecoderLayer
#   run_fused(layers, final_norm, x, positions, cos, sin)    -> (B, S, D)                   the whole stack plus the final norm

def add_norm(norm, x, residual):
    raise NotImplementedError("Day 4, Block 6")


def fused_layer(layer, x, residual, positions, cos, sin):
    raise NotImplementedError("Day 4, Block 6")


def run_fused(layers, final_norm, x, positions, cos, sin):
    raise NotImplementedError("Day 4, Block 6")
