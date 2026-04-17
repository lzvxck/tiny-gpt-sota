"""Sequence packing tests."""
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinysota.data.packing import pack_documents


def test_output_shape():
    docs = [[1, 2, 3, 4, 5], [6, 7, 8]]
    blocks = list(pack_documents(docs, block_size=4))
    for b in blocks:
        assert b.shape == (4,)
        assert b.dtype == np.uint16


def test_no_partial_blocks():
    """pack_documents should never yield a block shorter than block_size."""
    docs = [[i for i in range(100)] for _ in range(10)]
    for block in pack_documents(docs, block_size=32):
        assert len(block) == 32


def test_eos_inserted():
    """EOS token (id=2) should appear between documents."""
    docs = [[1, 1, 1], [2, 2, 2]]  # use tokens != 2
    all_tokens = []
    for block in pack_documents(docs, block_size=4, eos_id=99):
        all_tokens.extend(block.tolist())
    # EOS id=99 should appear
    assert 99 in all_tokens


def test_large_doc_split():
    """Documents larger than block_size should be split across blocks."""
    big_doc = list(range(1000))
    blocks = list(pack_documents([big_doc], block_size=64))
    assert len(blocks) >= 15  # 1000/64 ≈ 15
