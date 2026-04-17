"""Muon optimizer — orthogonalized Nesterov momentum for linear weight matrices.

Muon applies Newton-Schulz orthogonalization to gradients before the momentum
update. This keeps the effective learning update on the Stiefel manifold, which
empirically converges faster than AdamW for matrix-shaped parameters.

Usage in a hybrid setup:
    muon_params, adamw_params = split_params_for_hybrid(model)
    muon  = Muon(muon_params,  lr=0.02, momentum=0.95)
    adamw = torch.optim.AdamW(adamw_params, lr=4e-4, ...)

Muon is applied ONLY to 2D weight tensors of linear layers (not embeddings or norms).

References:
    - Shampoo: https://arxiv.org/abs/1802.09568
    - Muon implementation: https://github.com/KellerJordan/modded-nanogpt
    - "Muon is SOAP which is Shampoo which is ...": https://jeremybernste.in/writing/muon-is-soap
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Compute the orthogonal factor of G via Newton-Schulz iteration.

    Runs 5 iterations of the cubic NS polynomial:
        X <- a*X + b*A@X + c*A@A@X   where A = X@X.T
    Converges to the orthogonal factor (zeroth power) of G.
    All ops are BF16 matmuls — fast on any modern GPU.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16) / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz.

    Args:
        params:    2D weight tensors only (use split_params_for_hybrid to get these).
        lr:        Learning rate. Typical: 0.01 – 0.02. Higher than AdamW because
                   the orthogonalized update has unit spectral norm.
        momentum:  Nesterov momentum coefficient. Default 0.95.
        nesterov:  Use Nesterov momentum (recommended). Default True.
        ns_steps:  Newton-Schulz iterations. 5 is enough for float32/bf16.
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, (
                    f"Muon only supports 2D params, got shape {g.shape}. "
                    "Use split_params_for_hybrid() to separate params correctly."
                )

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)

                buf = state["buf"]
                buf.mul_(momentum).add_(g)

                # Nesterov lookahead
                g_update = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                # Orthogonalize
                g_orth = zeropower_via_newtonschulz5(g_update, steps=ns_steps)

                # Scale so the update has the same RMS as a unit-norm SGD step
                scale = max(g_orth.size(0), g_orth.size(1)) ** 0.5
                p.add_(g_orth, alpha=-lr * scale)

        return loss


def split_params_for_hybrid(model: torch.nn.Module) -> tuple[list[Tensor], list[Tensor]]:
    """Split model params into (muon_params, adamw_params).

    Muon params:  2D weight matrices of linear layers — NOT embeddings or lm_head.
    AdamW params: everything else (embeddings, norms, 1D params, lm_head/embed tie).

    The lm_head weight is tied to embed_tokens — we skip it so it's only counted
    once under AdamW via embed_tokens.
    """
    muon_params  = []
    adamw_params = []
    seen_ids: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue  # skip tied params (lm_head shares embed_tokens weight)
        seen_ids.add(id(param))

        is_matrix = param.ndim == 2
        is_embedding = "embed_tokens" in name
        is_lm_head = "lm_head" in name

        if is_matrix and not is_embedding and not is_lm_head:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params
