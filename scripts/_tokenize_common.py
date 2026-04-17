"""Shared tokenization-to-shards logic used by all download scripts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

from tinysota.data.packing import pack_documents


def tokenize_dataset_to_shards(
    text_iter: Iterator[str],
    out_dir: str,
    tokenizer_path: str,
    max_tokens: int,
    shard_size: int,
    block_size: int,
    n_workers: int,
    dataset_name: str,
) -> None:
    """Tokenize a text stream and write packed uint16 shards to out_dir."""
    import sentencepiece as spm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    eos_id = sp.eos_id()

    def _encode(text_iter):
        for text in text_iter:
            ids = sp.EncodeAsIds(text)
            if ids:
                yield ids

    shard_idx = 0
    tokens_written = 0
    shard_buf = []

    with tqdm(total=max_tokens, unit="tok", desc=dataset_name) as pbar:
        for block in pack_documents(_encode(text_iter), block_size, eos_id=eos_id):
            shard_buf.append(block)
            tokens_written += block_size
            pbar.update(block_size)

            if tokens_written - shard_idx * shard_size >= shard_size:
                _flush_shard(out_dir, shard_idx, shard_buf, block_size)
                shard_idx += 1
                shard_buf = []

            if tokens_written >= max_tokens:
                break

    if shard_buf:
        _flush_shard(out_dir, shard_idx, shard_buf, block_size)

    print(f"[{dataset_name}] wrote {shard_idx+1} shards, {tokens_written/1e9:.2f}B tokens → {out_dir}")


def _flush_shard(out_dir: Path, idx: int, blocks: list[np.ndarray], block_size: int) -> None:
    arr = np.concatenate(blocks)
    path = out_dir / f"shard_{idx:05d}.bin"
    m = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(arr),))
    m[:] = arr
    m.flush()
