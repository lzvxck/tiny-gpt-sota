"""Checkpoint save / load utilities."""
from __future__ import annotations

import threading
from pathlib import Path

import torch


def save_checkpoint(
    ckpt_dir: str | Path,
    step: int,
    tokens_seen: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    keep_last: int = 3,
) -> None:
    """Save checkpoint asynchronously so GPU is not idle during file I/O."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    path = ckpt_dir / f"step_{step:08d}.pt"

    def _save():
        torch.save(state, path)
        _write_latest(ckpt_dir, path)
        _prune_old(ckpt_dir, keep_last)

    t = threading.Thread(target=_save, daemon=True)
    t.start()


def load_checkpoint(path: str | Path, model, optimizer, scheduler) -> tuple[int, int]:
    """Load checkpoint in-place. Returns (step, tokens_seen)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng"])
    return state["step"], state["tokens_seen"]


def resume_path(latest_txt: str | Path) -> Path | None:
    p = Path(latest_txt)
    if p.exists():
        return Path(p.read_text().strip())
    return None


def _write_latest(ckpt_dir: Path, path: Path) -> None:
    (ckpt_dir / "latest.txt").write_text(str(path))


def _prune_old(ckpt_dir: Path, keep_last: int) -> None:
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in ckpts[:-keep_last]:
        old.unlink(missing_ok=True)
