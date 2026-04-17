"""Logging helpers — console (rich) + wandb."""
from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

console = Console()
_wandb = None
_progress: Progress | None = None
_task_id = None
_total_steps: int = 0


def init_wandb(project: str, run_name: str, config: dict) -> None:
    global _wandb
    try:
        import wandb
        wandb.init(project=project, name=run_name, config=config)
        _wandb = wandb
    except Exception as e:
        console.print(f"[yellow]wandb init failed: {e}. Console only.[/yellow]")


def init_progress(total_steps: int) -> None:
    global _progress, _task_id, _total_steps
    _total_steps = total_steps
    _progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("loss=[yellow]{task.fields[loss]:.4f}[/yellow]"),
        TextColumn("ppl=[magenta]{task.fields[ppl]:.1f}[/magenta]"),
        TextColumn("[dim]{task.fields[tok_s]} tok/s[/dim]"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    )
    _task_id = _progress.add_task(
        "Stage A", total=total_steps, loss=0.0, ppl=0.0, tok_s="—"
    )
    _progress.start()


def stop_progress() -> None:
    if _progress is not None:
        _progress.stop()


def log(step: int, metrics: dict[str, Any], tokens_seen: int, tok_per_sec: float) -> None:
    metrics["tokens_seen"] = tokens_seen
    metrics["tok_per_sec"] = tok_per_sec

    if _wandb is not None:
        _wandb.log(metrics, step=step)

    loss = metrics.get("loss", 0.0)
    ppl  = metrics.get("ppl",  0.0)
    lr   = metrics.get("lr",   0.0)
    tok_s = f"{tok_per_sec/1000:.1f}k" if tok_per_sec > 0 else "—"
    tokens_b = tokens_seen / 1e9

    if _progress is not None:
        _progress.update(_task_id, completed=step, loss=loss, ppl=ppl, tok_s=tok_s)
        # Print a summary line every log interval so it scrolls into the log
        _progress.console.print(
            f"[dim]step {step:>6}/{_total_steps}[/dim]  "
            f"loss=[yellow]{loss:.4f}[/yellow]  "
            f"ppl=[magenta]{ppl:.1f}[/magenta]  "
            f"lr={lr:.2e}  "
            f"tokens=[cyan]{tokens_b:.3f}B[/cyan]  "
            f"[dim]{tok_s} tok/s[/dim]"
        )
    else:
        parts = " | ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
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
