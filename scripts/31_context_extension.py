"""Stage B: context extension to 4K + annealing.

Loads the final Stage A checkpoint and continues training with:
  - seq_len 2048 → 4096
  - NTK-aware RoPE scaling
  - Annealing mix (up-weighted quality sources)
  - Linear LR decay to near zero

Usage:
    python scripts/31_context_extension.py
    python scripts/31_context_extension.py --config configs/train_stage_b.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinysota.utils.config import TrainConfig
from tinysota.training.loop import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_stage_b.yaml")
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.no_compile:
        cfg.compile = False

    print(f"Stage B: context extension + annealing")
    print(f"  seq_len:     {cfg.seq_len}")
    print(f"  Total tokens:{cfg.total_tokens/1e9:.1f}B")
    print(f"  Resume from: {cfg.resume_from}")
    print(f"  RoPE scaling:{cfg.rope_scaling}")

    train(cfg)
