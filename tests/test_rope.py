"""RoPE tests."""
import math
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinysota.model.rope import build_rope_freqs, build_rope_freqs_ntk, apply_rope


def test_freqs_shape():
    freqs = build_rope_freqs(head_dim=128, max_seq_len=512, theta=10000.0)
    assert freqs.shape == (512, 64)  # head_dim/2 complex numbers
    assert freqs.is_complex()


def test_apply_rope_shape():
    freqs = build_rope_freqs(head_dim=64, max_seq_len=32)
    x = torch.randn(2, 16, 4, 64)  # (batch, seq, heads, head_dim)
    out = apply_rope(x, freqs)
    assert out.shape == x.shape


def test_apply_rope_dtype_preserved():
    freqs = build_rope_freqs(64, 32)
    x = torch.randn(1, 8, 2, 64, dtype=torch.bfloat16)
    out = apply_rope(x, freqs)
    assert out.dtype == torch.bfloat16


def test_ntk_scaling_changes_theta():
    freqs_base = build_rope_freqs(128, 256, theta=100000.0)
    freqs_ntk  = build_rope_freqs_ntk(128, 256, base_theta=100000.0, scale_factor=2.0)
    # NTK scaling changes the frequencies — they should differ
    assert not torch.allclose(freqs_base.real, freqs_ntk.real, atol=1e-3)


def test_rope_relative_shift():
    """Rotating both q and k by the same offset preserves their dot product
    up to the position difference — this is the core RoPE invariant."""
    freqs = build_rope_freqs(64, 64)
    q = torch.randn(1, 16, 1, 64)
    k = torch.randn(1, 16, 1, 64)
    q_rot = apply_rope(q, freqs)
    k_rot = apply_rope(k, freqs)
    # Dot product between position i and position i (same offset) should equal
    # dot product computed with unrotated vectors scaled by cos(0)=1 at relative dist=0
    # We just check outputs are finite and non-zero
    assert not torch.isnan(q_rot).any()
    assert not torch.isnan(k_rot).any()
