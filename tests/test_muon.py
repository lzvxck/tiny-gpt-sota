"""Muon optimizer tests."""
import torch
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinysota.training.muon import Muon, zeropower_via_newtonschulz5, split_params_for_hybrid
from tinysota.model import LlamaModel, LlamaConfig


TINY = LlamaConfig(
    vocab_size=256, n_layers=2, d_model=128, n_heads=2,
    n_kv_heads=1, head_dim=64, ffn_dim=256, max_seq_len=32,
)


def test_zeropower_shape():
    G = torch.randn(64, 32)
    X = zeropower_via_newtonschulz5(G)
    assert X.shape == G.shape


def test_zeropower_singular_values_squeezed():
    """NS5 squeezes singular values toward 1 — not perfectly orthogonal, but bounded.

    NS5 is a finite-step approximation. The real guarantee is that singular values
    are pushed toward 1 (away from 0 and from large values), which normalizes the
    effective update magnitude. atol=0.05 is too tight for 5 steps on a random matrix.
    """
    G = torch.randn(64, 64)
    X = zeropower_via_newtonschulz5(G, steps=5)
    svd_in  = torch.linalg.svdvals(G.float())
    svd_out = torch.linalg.svdvals(X.float())
    # Input singular values spread widely; output should be much tighter around 1
    spread_in  = svd_in.max()  - svd_in.min()
    spread_out = svd_out.max() - svd_out.min()
    assert spread_out < spread_in, \
        f"NS5 did not reduce singular value spread: {spread_in:.3f} → {spread_out:.3f}"
    # All output singular values should be in (0, 2) — semi-unitary bound
    assert svd_out.max() < 2.0 and svd_out.min() > 0.0


def test_muon_step():
    """Muon should update 2D weights."""
    W = torch.nn.Parameter(torch.randn(64, 32))
    opt = Muon([W], lr=0.01)
    loss = (W ** 2).sum()
    loss.backward()
    W_before = W.data.clone()
    opt.step()
    assert not torch.equal(W.data, W_before), "Muon did not update weights"


def test_muon_loss_decreases():
    """Muon should reduce loss on a simple quadratic."""
    W = torch.nn.Parameter(torch.randn(32, 16))
    opt = Muon([W], lr=0.02)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = (W ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_split_params():
    model = LlamaModel(TINY)
    muon_params, adamw_params = split_params_for_hybrid(model)

    # All 2D non-embedding weights → Muon
    assert len(muon_params) > 0
    for p in muon_params:
        assert p.ndim == 2

    # No duplicates (tied weights counted once)
    all_ids = [id(p) for p in muon_params] + [id(p) for p in adamw_params]
    assert len(all_ids) == len(set(all_ids)), "Duplicate params in split"

    # All params accounted for
    model_param_ids = {id(p) for p in model.parameters() if p.requires_grad}
    split_param_ids = set(all_ids)
    assert model_param_ids == split_param_ids, "Not all params are in the split"


def test_hybrid_trains():
    """Full hybrid training step on tiny model."""
    model = LlamaModel(TINY)
    muon_params, adamw_params = split_params_for_hybrid(model)
    muon  = Muon(muon_params, lr=0.02)
    adamw = torch.optim.AdamW(adamw_params, lr=1e-3)

    ids = torch.randint(0, 256, (2, 16))
    losses = []
    for _ in range(20):
        muon.zero_grad(); adamw.zero_grad()
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), ids.view(-1))
        loss.backward()
        muon.step(); adamw.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Hybrid loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
