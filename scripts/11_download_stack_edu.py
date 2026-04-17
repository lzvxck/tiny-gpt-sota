"""Download and tokenize StackExchange (code/math Q&A from dolmino-mix-1124)."""
import argparse
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts._tokenize_common import tokenize_dataset_to_shards
from tinysota.data.streaming import stream_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=2_000_000_000)
    parser.add_argument("--shard_size", type=int, default=100_000_000)
    parser.add_argument("--out_dir", default="data/tokenized/python_edu/train")
    parser.add_argument("--tokenizer", default="data/tokenizer/tinysota.model")
    args = parser.parse_args()

    tokenize_dataset_to_shards(
        text_iter=stream_dataset("python_edu"),
        out_dir=args.out_dir,
        tokenizer_path=args.tokenizer,
        max_tokens=args.max_tokens,
        shard_size=args.shard_size,
        block_size=2048,
        n_workers=2,
        dataset_name="python_edu",
    )
