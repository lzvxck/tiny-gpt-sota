"""Evaluate perplexity on held-out FineWeb-Edu shards.

Usage:
    python scripts/40_eval_perplexity.py --checkpoint checkpoints/stage_a/step_00034332.pt
"""
import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinysota.model import LlamaModel, LlamaConfig
from tinysota.utils.config import load_yaml
from tinysota.data.shard_io import read_shard


def eval_perplexity(model, shard_paths, seq_len, device, dtype, n_batches=200):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_size = 4

    with torch.no_grad():
        for shard_path in shard_paths[:5]:  # use first 5 eval shards
            data = read_shard(shard_path).astype("int64")
            n = (len(data) // seq_len) * seq_len
            data = data[:n].reshape(-1, seq_len)
            for i in range(0, len(data) - batch_size, batch_size):
                x = torch.from_numpy(data[i:i+batch_size]).to(device)
                y = torch.from_numpy(data[i+1:i+batch_size+1]).to(device)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        y[:, 1:].reshape(-1),
                        reduction="sum",
                    )
                total_loss += loss.item()
                total_tokens += x.numel()
                if total_tokens > n_batches * batch_size * seq_len:
                    break

    ppl = math.exp(total_loss / max(1, total_tokens))
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_config", default="configs/model_150m.yaml")
    parser.add_argument("--eval_dir", default="data/tokenized/fineweb_edu/eval")
    parser.add_argument("--seq_len", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    cfg_dict = load_yaml(args.model_config)
    cfg = LlamaConfig.from_dict(cfg_dict)
    model = LlamaModel(cfg).to(device, dtype=dtype)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    eval_shards = sorted(Path(args.eval_dir).glob("*.bin"))
    if not eval_shards:
        print(f"No eval shards in {args.eval_dir}. Run download scripts with --split eval.")
        sys.exit(1)

    ppl = eval_perplexity(model, eval_shards, args.seq_len, device, dtype)
    print(f"Perplexity: {ppl:.2f}")
