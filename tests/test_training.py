"""Training pipeline integration tests — CPU only."""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinysota.model import LlamaModel, LlamaConfig
from tinysota.data.shard_io import write_shard, gather_shards, shard_loader
from tinysota.training.schedule import cosine_with_warmup, linear_decay
from tinysota.training.checkpoint import save_checkpoint, load_checkpoint


TINY = LlamaConfig(
    vocab_size=256, n_layers=2, d_model=64, n_heads=2,
    n_kv_heads=1, head_dim=32, ffn_dim=128, max_seq_len=32,
)


def test_shard_write_read():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "shard_00000.bin"
        tokens = np.arange(1000, dtype=np.uint16)
        write_shard(path, tokens)
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        result = np.array(mm)   # copy into regular array so we can release the memmap
        del mm                  # release file lock before TemporaryDirectory cleanup (Windows)
        assert np.array_equal(result, tokens)


def test_shard_loader():
    with tempfile.TemporaryDirectory() as tmp:
        shards_dir = Path(tmp)
        # Write two small shards
        for i in range(2):
            tokens = np.array(list(range(256)) * 4, dtype=np.uint16)  # 1024 tokens
            write_shard(shards_dir / f"shard_{i:05d}.bin", tokens)
        paths = sorted(shards_dir.glob("*.bin"))
        loader = shard_loader(paths, seq_len=8, micro_batch=2)
        x, y = next(loader)
        assert x.shape == (2, 8)
        assert y.shape == (2, 8)


def test_loss_decreases():
    """Overfit a tiny model on a single batch — loss should fall."""
    model = LlamaModel(TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 256, (2, 16))

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), ids.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_cosine_schedule():
    model = LlamaModel(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = cosine_with_warmup(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    lrs = []
    for _ in range(100):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[9] > lrs[0]           # warmup increasing
    assert lrs[-1] < lrs[9]          # cosine decay
    assert lrs[-1] >= 1e-3 * 0.1 - 1e-6  # min_lr respected


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        model = LlamaModel(TINY)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = cosine_with_warmup(opt, 5, 50)

        save_checkpoint(tmp, step=10, tokens_seen=1000, model=model,
                        optimizer=opt, scheduler=sched)
        import time; time.sleep(0.2)  # let async thread finish

        # Load into fresh instances
        m2   = LlamaModel(TINY)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
        sch2 = cosine_with_warmup(opt2, 5, 50)

        ckpt = sorted(Path(tmp).glob("step_*.pt"))[0]
        step, tokens = load_checkpoint(ckpt, m2, opt2, sch2)
        assert step == 10
        assert tokens == 1000
