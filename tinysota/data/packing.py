"""Sequence packing utilities.

Packing concatenates tokenized documents separated by EOS tokens into
fixed-length blocks, eliminating all padding waste.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np


EOS_ID = 2  # SentencePiece default EOS; update if tokenizer differs


def pack_documents(
    token_iter: Iterable[list[int]],
    block_size: int,
    eos_id: int = EOS_ID,
) -> Iterator[np.ndarray]:
    """Yield fixed-size blocks packed from document token streams.

    Args:
        token_iter: iterable of token-id lists, one per document.
        block_size: output sequence length.
        eos_id: token appended after each document.

    Yields:
        np.ndarray of shape (block_size,) dtype uint16.
    """
    buf: list[int] = []
    for doc_tokens in token_iter:
        buf.extend(doc_tokens)
        buf.append(eos_id)
        while len(buf) >= block_size:
            block = np.array(buf[:block_size], dtype=np.uint16)
            buf = buf[block_size:]
            yield block
    # Discard the partial tail block — never pad.
