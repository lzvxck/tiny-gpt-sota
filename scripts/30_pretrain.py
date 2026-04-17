"""Stage A pretraining entry point.

Usage:
    python scripts/30_pretrain.py
    python scripts/30_pretrain.py --config configs/train_stage_a.yaml
    python scripts/30_pretrain.py --config configs/train_stage_a.yaml --no_compile
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinysota.utils.config import TrainConfig
from tinysota.training.loop import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_stage_a.yaml")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile (fallback for Triton issues)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.no_compile:
        cfg.compile = False
    if not args.resume:
        cfg.resume_from = None

    print(f"Stage A pretraining")
    print(f"  Config:      {args.config}")
    print(f"  Total tokens:{cfg.total_tokens/1e9:.1f}B")
    print(f"  Total steps: {cfg.total_steps:,}")
    print(f"  Eff batch:   {cfg.effective_batch_tokens/1e3:.0f}K tokens")
    print(f"  compile:     {cfg.compile}")

    train(cfg)
