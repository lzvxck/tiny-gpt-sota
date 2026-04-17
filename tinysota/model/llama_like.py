"""Llama-style decoder-only transformer.

Architecture: RMSNorm (pre-norm) + RoPE + SwiGLU MLP + MQA attention.
All tensors are BF16 during forward; norms run in FP32 and cast back.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .rope import build_rope_freqs, build_rope_freqs_ntk, apply_rope
from .init_weights import init_weights


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    n_layers: int = 16
    d_model: int = 896
    n_heads: int = 7
    n_kv_heads: int = 1        # 1 = MQA, >1 = GQA
    head_dim: int = 128        # must equal d_model // n_heads
    ffn_dim: int = 2432
    norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    rope_scaling: Optional[dict] = None   # {"type": "ntk", "scale_factor": 2.0}
    tie_embeddings: bool = True
    max_seq_len: int = 4096
    dropout: float = 0.0

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads * head_dim "
            f"({self.n_heads} * {self.head_dim})"
        )

    @classmethod
    def from_dict(cls, d: dict) -> "LlamaConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Upcast to FP32 for numerical stability, cast back after
        x_f32 = x.float()
        norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU MLP: uses three linear projections (gate, up, down)."""
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """Multi-Query Attention (n_kv_heads=1) or GQA (n_kv_heads>1).

    Uses torch.nn.functional.scaled_dot_product_attention which auto-dispatches
    to FlashAttention-2 kernels when available and input is contiguous BF16.
    """
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.head_dim
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads  # repeat factor for K/V

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads    * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model,    bias=False)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Expand K/V to match Q heads (MQA/GQA broadcast)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # SDPA expects (B, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn      = Attention(cfg)
        self.mlp_norm  = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp       = SwiGLU(cfg.d_model, cfg.ffn_dim)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class LlamaModel(nn.Module):
    def __init__(self, cfg: LlamaConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = gradient_checkpointing

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._register_rope(cfg)
        init_weights(self, cfg.n_layers)

    def _register_rope(self, cfg: LlamaConfig) -> None:
        if cfg.rope_scaling is None:
            freqs_cis = build_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        else:
            assert cfg.rope_scaling["type"] == "ntk"
            freqs_cis = build_rope_freqs_ntk(
                cfg.head_dim,
                cfg.max_seq_len,
                cfg.rope_theta,
                cfg.rope_scaling["scale_factor"],
            )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Returns logits (B, T, vocab_size)."""
        x = self.embed_tokens(input_ids)
        freqs = self.freqs_cis

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Recompute layer activations during backward instead of storing them.
                # use_reentrant=False is required for compatibility with torch.compile.
                x = grad_checkpoint(layer, x, freqs, use_reentrant=False)
            else:
                x = layer(x, freqs)

        x = self.norm(x)
        return self.lm_head(x)

    def num_params(self, exclude_embeddings: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.embed_tokens.weight.numel()
        return total
