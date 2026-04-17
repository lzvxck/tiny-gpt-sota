import math
import torch
from torch import Tensor


def build_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 100000.0) -> Tensor:
    """Precompute inverse frequencies for RoPE."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)           # (seq, head_dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis                            # (seq, head_dim/2)


def build_rope_freqs_ntk(
    head_dim: int,
    max_seq_len: int,
    base_theta: float,
    scale_factor: float,
) -> Tensor:
    """NTK-aware RoPE: stretch theta to handle longer context without fine-tuning.

    Derived from the 'NTK-by-parts' observation by bloc97 (2023):
    scaling theta by scale_factor^(dim/(dim-2)) preserves high-frequency info.
    """
    new_theta = base_theta * (scale_factor ** (head_dim / (head_dim - 2)))
    return build_rope_freqs(head_dim, max_seq_len, theta=new_theta)


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply RoPE to query or key tensor.

    Args:
        x: (batch, seq, n_heads, head_dim)
        freqs_cis: (seq, head_dim/2) complex

    Returns:
        Tensor same shape as x.
    """
    seq = x.shape[1]
    xf = x.float().reshape(*x.shape[:-1], -1, 2)  # (..., head_dim/2, 2)
    xc = torch.view_as_complex(xf)                 # (..., head_dim/2) complex
    freqs = freqs_cis[:seq].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim/2)
    xc_rot = xc * freqs
    out = torch.view_as_real(xc_rot).flatten(-2)   # (..., head_dim)
    return out.to(x.dtype)
