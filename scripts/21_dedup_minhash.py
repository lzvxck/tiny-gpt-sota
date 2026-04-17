"""Cross-dataset MinHash deduplication.

Reads all tokenized shards and removes near-duplicate sequences (Jaccard >= 0.8).
This is a post-tokenization dedup step to catch duplicates between datasets.

Note: FineWeb-Edu is already internally deduped; this catches cross-dataset overlaps.

Usage:
    python scripts/21_dedup_minhash.py --data_dir data/tokenized
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


NUM_PERM = 128
JACCARD_THRESHOLD = 0.8
BLOCK_SIZE = 2048


def shingle_tokens(tokens: np.ndarray, k: int = 5) -> set[bytes]:
    """k-gram shingles over token ids."""
    return {tokens[i:i+k].tobytes() for i in range(len(tokens) - k + 1)}


def build_minhash(shingles: set[bytes]) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles:
        m.update(s)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/tokenized")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_shards = sorted(data_dir.rglob("*.bin"))
    print(f"Found {len(all_shards)} shards")

    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
    duplicates: set[str] = set()

    for shard_path in tqdm(all_shards, desc="MinHash pass"):
        data = np.memmap(shard_path, dtype=np.uint16, mode="r")
        n_blocks = len(data) // BLOCK_SIZE
        for i in range(n_blocks):
            block = data[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            key = f"{shard_path}:{i}"
            shingles = shingle_tokens(block)
            if len(shingles) < 10:
                continue
            mh = build_minhash(shingles)
            result = lsh.query(mh)
            if result:
                duplicates.add(key)
            else:
                try:
                    lsh.insert(key, mh)
                except ValueError:
                    pass  # already inserted

    dup_rate = len(duplicates) / max(1, sum(len(np.memmap(p, dtype=np.uint16, mode="r")) // BLOCK_SIZE for p in all_shards))
    print(f"Duplicate blocks found: {len(duplicates)} ({dup_rate*100:.1f}%)")

    if duplicates:
        dup_file = data_dir / "duplicates.txt"
        dup_file.write_text("\n".join(sorted(duplicates)))
        print(f"Duplicate keys written to {dup_file}")
        print("Re-run 23_tokenize_pack.py with --skip_duplicates to filter them out.")
    else:
        print("No significant cross-dataset duplicates found.")


if __name__ == "__main__":
    main()
