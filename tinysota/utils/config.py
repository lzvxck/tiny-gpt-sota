import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class TrainConfig:
    model_config: str = "configs/model_150m.yaml"
    stage: str = "a"
    data_dir: str = "data/tokenized"
    seq_len: int = 2048
    dataset_weights: dict = field(default_factory=dict)
    total_tokens: int = 9_000_000_000
    micro_batch: int = 8
    grad_accum_steps: int = 16
    optimizer: str = "hybrid"      # "hybrid" = AdamW+Muon | "adamw_fused" = AdamW only
    lr: float = 4e-4               # AdamW learning rate
    betas: list = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    eps: float = 1e-8
    muon_lr: float = 0.02          # Muon LR — higher than AdamW; orthogonalization normalizes scale
    muon_momentum: float = 0.95
    warmup_pct: float = 0.02
    decay: str = "cosine"
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    precision: str = "bf16"
    gradient_checkpointing: bool = False
    compile: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"
    attn_impl: str = "sdpa"
    rope_scaling: Optional[dict] = None
    log_every: int = 10
    eval_every: int = 2000
    ckpt_every: int = 2000
    ckpt_dir: str = "checkpoints/stage_a"
    resume_from: Optional[str] = None
    wandb_project: str = "tiny-sota-llm"
    wandb_run_name: str = "run"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        d = load_yaml(path)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def effective_batch_tokens(self) -> int:
        return self.micro_batch * self.grad_accum_steps * self.seq_len

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.effective_batch_tokens
