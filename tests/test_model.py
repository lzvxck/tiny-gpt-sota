"""Model architecture tests — CPU-only, no GPU required."""
import math
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinysota.model.llama_like import LlamaModel, LlamaConfig, RMSNorm, SwiGLU, Attention


TINY = LlamaConfig(
    vocab_size=256, n_layers=2, d_model=128, n_heads=2,
    n_kv_heads=1, head_dim=64, ffn_dim=256, max_seq_len=64,
)


def test_param_count():
    m = LlamaModel(TINY)
    n = m.num_params()
    assert n > 0
    print(f"Tiny model: {n/1e3:.1f}K params")


def test_forward_shape():
    m = LlamaModel(TINY)
    ids = torch.randint(0, 256, (2, 32))
    logits = m(ids)
    assert logits.shape == (2, 32, 256)


def test_forward_no_nan():
    m = LlamaModel(TINY)
    ids = torch.randint(0, 256, (1, 16))
    logits = m(ids)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_causal_mask():
    """Changing tokens at position t should not affect logits at positions < t."""
    m = LlamaModel(TINY)
    m.eval()
    ids1 = torch.randint(0, 256, (1, 16))
    ids2 = ids1.clone()
    ids2[0, 8:] = torch.randint(0, 256, (8,))  # change suffix
    with torch.no_grad():
        l1 = m(ids1)
        l2 = m(ids2)
    assert torch.allclose(l1[0, :8], l2[0, :8], atol=1e-4), \
        "Causal mask violated: prefix logits changed when suffix was modified"


def test_tied_embeddings():
    m = LlamaModel(TINY)
    assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr(), \
        "Embedding weights are not tied"


def test_backward():
    m = LlamaModel(TINY)
    ids = torch.randint(0, 256, (2, 16))
    logits = m(ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), ids.view(-1))
    loss.backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


def test_rmsnorm():
    norm = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_swiglu():
    mlp = SwiGLU(128, 256)
    x = torch.randn(2, 8, 128)
    out = mlp(x)
    assert out.shape == x.shape


def test_gqa_config():
    """Test that GQA (n_kv_heads=2) also works, not just MQA."""
    cfg = LlamaConfig(
        vocab_size=256, n_layers=2, d_model=128, n_heads=4,
        n_kv_heads=2, head_dim=32, ffn_dim=256, max_seq_len=64,
    )
    m = LlamaModel(cfg)
    ids = torch.randint(0, 256, (1, 16))
    out = m(ids)
    assert out.shape == (1, 16, 256)


def test_model_full_config():
    """Smoke test the actual 145M config on CPU (slow but validates shapes)."""
    cfg = LlamaConfig(
        vocab_size=32000, n_layers=16, d_model=896, n_heads=7,
        n_kv_heads=1, head_dim=128, ffn_dim=2432, max_seq_len=128,
    )
    m = LlamaModel(cfg)
    n = m.num_params()
    print(f"145M config: {n/1e6:.1f}M params")
    assert 130e6 < n < 200e6, f"Unexpected param count: {n/1e6:.1f}M"
    # Forward with tiny seq to keep CPU test fast
    ids = torch.randint(0, 32000, (1, 16))
    out = m(ids)
    assert out.shape == (1, 16, 32000)
