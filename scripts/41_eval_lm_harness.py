"""Run lm-evaluation-harness benchmarks.

Tasks: hellaswag, arc_easy, arc_challenge, piqa, winogrande, mmlu

Usage:
    python scripts/41_eval_lm_harness.py --checkpoint checkpoints/stage_b/step_00007623.pt
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "mmlu"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_config", default="configs/model_150m.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/tinysota.model")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    try:
        import lm_eval
    except ImportError:
        print("lm-eval not installed. Run: pip install lm-eval>=0.4.3")
        sys.exit(1)

    from tinysota.model import LlamaModel, LlamaConfig
    from tinysota.utils.config import load_yaml

    device = "cuda"
    dtype = torch.bfloat16

    cfg_dict = load_yaml(args.model_config)
    cfg = LlamaConfig.from_dict(cfg_dict)
    model = LlamaModel(cfg).to(device, dtype=dtype)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    # Wrap for lm-eval
    from lm_eval.models.huggingface import HFLM
    import sentencepiece as spm
    from transformers import PreTrainedTokenizerFast
    from tokenizers import SentencePieceBPETokenizer

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    # lm-eval expects a HF-compatible interface; use HFLM wrapper with our model
    # For a custom SentencePiece model we need to wrap into HF tokenizer format
    print("Note: lm-eval integration requires HF-compatible tokenizer. "
          "Wrapping sentencepiece model...")

    # Simple wrapper — for full support consider exporting to HF format first
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.checkpoint}",  # placeholder — see note below
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n=== Eval Results ===")
    for task, metrics in results["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc_norm,none", "N/A"))
        print(f"  {task:<20} acc={acc}")
