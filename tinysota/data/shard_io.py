"""Read tokenized shards stored as uint16 numpy memmaps (nanoGPT-style).

Each shard file is a flat array of token IDs written as numpy uint16.
We keep vocab <= 65535 (SentencePiece BPE 32K fits easily).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    """Write a token array to a shard file."""
    assert tokens.dtype == np.uint16
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(tokens),))
    arr[:] = tokens
    arr.flush()


def read_shard(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def shard_loader(
    shard_paths: list[Path],
    seq_len: int,
    micro_batch: int,
    dataset_name: str = "",
    shuffle_shards: bool = True,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Infinite iterator that yields (input_ids, labels) tensors.

    Sequences are packed end-to-end from shards.  Labels are input_ids
    shifted by one position (standard causal LM).

    Yields:
        input_ids: (micro_batch, seq_len) int64
        labels:    (micro_batch, seq_len) int64
    """
    chunk = seq_len * micro_batch + 1  # +1 so we can shift for labels
    buf = np.empty(0, dtype=np.uint16)

    while True:
        paths = list(shard_paths)
        if shuffle_shards:
            random.shuffle(paths)
        for p in paths:
            data = read_shard(p).astype(np.int64)
            buf = np.concatenate([buf.astype(np.int64), data])
            while len(buf) >= chunk:
                block = buf[:chunk]
                buf   = buf[chunk - 1:]  # overlap by 1 for label shift
                x = torch.from_numpy(block[:-1].reshape(micro_batch, seq_len))
                y = torch.from_numpy(block[1: ].reshape(micro_batch, seq_len))
                yield x, y


def gather_shards(data_dir: str | Path, split: str = "train") -> list[Path]:
    """Return sorted list of shard files matching data_dir/<split>/*.bin."""
    d = Path(data_dir)
    shards = sorted(d.glob(f"{split}/**/*.bin")) + sorted(d.glob(f"{split}/*.bin"))
    if not shards:
        raise FileNotFoundError(f"No .bin shards found under {d / split}")
    return shards
