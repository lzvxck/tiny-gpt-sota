import math
import torch.nn as nn


def init_weights(model: nn.Module, n_layers: int) -> None:
    """GPT-NeoX style weight initialisation.

    Linear layers: truncated normal σ=0.02.
    Output projections (attn o_proj + mlp down_proj): scaled by 1/sqrt(2*n_layers)
    so residual stream variance stays ~1 at initialisation.
    Embeddings: truncated normal σ=0.02.
    RMSNorm weights: ones (handled by default, but explicit here).
    """
    output_scale = 1.0 / math.sqrt(2 * n_layers)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if _is_output_proj(name):
                module.weight.data.mul_(output_scale)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)


def _is_output_proj(name: str) -> bool:
    return name.endswith("o_proj") or name.endswith("down_proj")
