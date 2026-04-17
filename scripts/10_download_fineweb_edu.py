"""Stream and tokenize FineWeb-Edu into .bin shards.

Uses HF datasets streaming — never downloads the full 1.3T dataset.
We pull the 10BT sample subset and take what we need.

Usage:
    python scripts/10_download_fineweb_edu.py --max_tokens 6_000_000_000 --workers 4
"""
import argparse
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts._tokenize_common import tokenize_dataset_to_shards
from tinysota.data.streaming import stream_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=6_000_000_000)
    parser.add_argument("--shard_size", type=int, default=100_000_000, help="tokens per shard")
    parser.add_argument("--out_dir", default="data/tokenized/fineweb_edu/train")
    parser.add_argument("--tokenizer", default="data/tokenizer/tinysota.model")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    tokenize_dataset_to_shards(
        text_iter=stream_dataset("fineweb_edu"),
        out_dir=args.out_dir,
        tokenizer_path=args.tokenizer,
        max_tokens=args.max_tokens,
        shard_size=args.shard_size,
        block_size=2048,
        n_workers=args.workers,
        dataset_name="fineweb_edu",
    )
