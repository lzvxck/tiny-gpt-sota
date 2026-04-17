"""Core training loop — shared by Stage A and Stage B."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

# Reduce CUDA allocator fragmentation on Windows (must be set before first CUDA alloc)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch import Tensor

from tinysota.model import LlamaModel, LlamaConfig
from tinysota.utils.config import TrainConfig, load_yaml
from tinysota.data.shard_io import gather_shards, shard_loader
from tinysota.training.schedule import cosine_with_warmup, linear_decay
from tinysota.training.checkpoint import save_checkpoint, load_checkpoint, resume_path
from tinysota.training.logging_utils import init_wandb, init_progress, stop_progress, log, Throughput
from tinysota.training.muon import Muon, split_params_for_hybrid


def build_model(cfg: TrainConfig) -> LlamaModel:
    model_dict = load_yaml(cfg.model_config)
    if cfg.rope_scaling is not None:
        model_dict["rope_scaling"] = cfg.rope_scaling
    lm_cfg = LlamaConfig.from_dict(model_dict)
    return LlamaModel(lm_cfg, gradient_checkpointing=cfg.gradient_checkpointing)


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if cfg.precision == "bf16" else torch.float16

    # Model
    model = build_model(cfg).to(device, dtype=dtype)
    n_params = model.num_params()
    print(f"Model: {n_params/1e6:.1f}M params")

    if cfg.compile:
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
            print(f"torch.compile: {cfg.compile_mode}")
        except Exception as e:
            print(f"torch.compile failed ({e}), running eager")

    # Optimizer — pure AdamW or hybrid AdamW+Muon
    warmup_steps = max(1, int(cfg.total_steps * cfg.warmup_pct))

    if cfg.optimizer == "hybrid":
        muon_params, adamw_params = split_params_for_hybrid(model)
        optimizer = _HybridOptimizer(
            muon_params, adamw_params,
            muon_lr=cfg.muon_lr,
            muon_momentum=cfg.muon_momentum,
            adamw_lr=cfg.lr,
            adamw_betas=tuple(cfg.betas),
            adamw_wd=cfg.weight_decay,
            adamw_eps=cfg.eps,
        )
        n_muon  = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        print(f"Hybrid optimizer: Muon {n_muon/1e6:.1f}M params | AdamW {n_adamw/1e6:.1f}M params")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=tuple(cfg.betas),
            weight_decay=cfg.weight_decay,
            eps=cfg.eps,
            fused=True,
        )

    # Scheduler — one LambdaLR wrapping the primary optimizer (AdamW)
    primary_opt = optimizer.adamw if cfg.optimizer == "hybrid" else optimizer
    if cfg.decay == "cosine":
        scheduler = cosine_with_warmup(primary_opt, warmup_steps, cfg.total_steps, cfg.min_lr_ratio)
    else:
        scheduler = linear_decay(primary_opt, cfg.total_steps, cfg.min_lr_ratio)

    # Muon LR scales with the same ratio as AdamW via a coupled step
    muon_scheduler = None
    if cfg.optimizer == "hybrid":
        if cfg.decay == "cosine":
            muon_scheduler = cosine_with_warmup(optimizer.muon, warmup_steps, cfg.total_steps, cfg.min_lr_ratio)
        else:
            muon_scheduler = linear_decay(optimizer.muon, cfg.total_steps, cfg.min_lr_ratio)

    # Resume
    start_step, tokens_seen = 0, 0
    if cfg.resume_from:
        rp = resume_path(cfg.resume_from)
        if rp and rp.exists():
            start_step, tokens_seen = load_checkpoint(rp, model, optimizer, scheduler)
            print(f"Resumed from {rp} at step {start_step}, {tokens_seen/1e9:.2f}B tokens")

    # Data
    shards = _gather_weighted_shards(cfg)
    loader = shard_loader(shards, cfg.seq_len, cfg.micro_batch)

    # Logging
    init_wandb(cfg.wandb_project, cfg.wandb_run_name, vars(cfg))
    init_progress(cfg.total_steps)
    throughput = Throughput(cfg.seq_len, cfg.micro_batch, cfg.grad_accum_steps)
    throughput.reset(start_step)

    # Training loop
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    step = start_step
    loss_acc = 0.0

    while step < cfg.total_steps:
        if cfg.optimizer == "hybrid":
            optimizer.muon.zero_grad(set_to_none=True)
            optimizer.adamw.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            x, y = next(loader)
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
                loss = loss / cfg.grad_accum_steps
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if muon_scheduler is not None:
            muon_scheduler.step()

        step += 1
        tokens_seen += cfg.effective_batch_tokens
        loss_acc += step_loss

        if step % cfg.log_every == 0:
            avg_loss = loss_acc / cfg.log_every
            loss_acc = 0.0
            tok_per_sec = throughput.update(step)
            throughput.reset(step)
            log(
                step,
                {
                    "loss": avg_loss,
                    "ppl": math.exp(min(avg_loss, 20)),
                    "lr": scheduler.get_last_lr()[0],
                },
                tokens_seen,
                tok_per_sec,
            )

        if step % cfg.ckpt_every == 0:
            save_checkpoint(cfg.ckpt_dir, step, tokens_seen, model, optimizer, scheduler)

    save_checkpoint(cfg.ckpt_dir, step, tokens_seen, model, optimizer, scheduler)
    stop_progress()
    print(f"Training complete. {tokens_seen/1e9:.2f}B tokens.")


class _HybridOptimizer:
    """Thin container holding both Muon and AdamW so the rest of the loop
    can call .step() / .state_dict() / .load_state_dict() uniformly."""

    def __init__(self, muon_params, adamw_params, muon_lr, muon_momentum,
                 adamw_lr, adamw_betas, adamw_wd, adamw_eps):
        self.muon  = Muon(muon_params, lr=muon_lr, momentum=muon_momentum)
        self.adamw = torch.optim.AdamW(
            adamw_params, lr=adamw_lr, betas=adamw_betas,
            weight_decay=adamw_wd, eps=adamw_eps, fused=True,
        )

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def _gather_weighted_shards(cfg: TrainConfig) -> list[Path]:
    """Collect shards from all datasets, repeat according to weights."""
    all_shards = []
    data_dir = Path(cfg.data_dir)
    for dataset_name, weight in cfg.dataset_weights.items():
        d = data_dir / dataset_name / "train"
        if not d.exists():
            print(f"[warn] shard dir not found: {d} — skipping")
            continue
        shards = sorted(d.glob("*.bin"))
        if not shards:
            print(f"[warn] no shards in {d} — skipping")
            continue
        # Repeat shards proportional to weight (crude but effective at this scale)
        repeat = max(1, round(weight * 10))
        all_shards.extend(shards * repeat)
    if not all_shards:
        raise RuntimeError(f"No shards found in {data_dir}. Run scripts/23_tokenize_pack.py first.")
    return all_shards
