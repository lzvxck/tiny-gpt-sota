"""Logging helpers — console (rich) + wandb."""
from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()
_wandb = None


def init_wandb(project: str, run_name: str, config: dict) -> None:
    global _wandb
    try:
        import wandb
        wandb.init(project=project, name=run_name, config=config)
        _wandb = wandb
    except Exception as e:
        console.print(f"[yellow]wandb init failed: {e}. Logging to console only.[/yellow]")


def log(step: int, metrics: dict[str, Any], tokens_seen: int, tok_per_sec: float) -> None:
    metrics["tokens_seen"] = tokens_seen
    metrics["tok_per_sec"] = tok_per_sec
    if _wandb is not None:
        _wandb.log(metrics, step=step)
    else:
        parts = " | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        console.print(f"[cyan]step {step:>8}[/cyan]  {parts}  [dim]{tok_per_sec:.0f} tok/s[/dim]")


class Throughput:
    def __init__(self, seq_len: int, micro_batch: int, grad_accum: int):
        self.tokens_per_step = seq_len * micro_batch * grad_accum
        self._t0 = time.perf_counter()
        self._step0 = 0

    def update(self, step: int) -> float:
        elapsed = time.perf_counter() - self._t0
        steps = step - self._step0
        if elapsed < 1e-6:
            return 0.0
        return (steps * self.tokens_per_step) / elapsed

    def reset(self, step: int) -> None:
        self._t0 = time.perf_counter()
        self._step0 = step
