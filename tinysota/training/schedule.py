"""Learning rate schedulers."""
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Linear warmup then cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def linear_decay(
    optimizer: Optimizer,
    total_steps: int,
    min_lr_ratio: float = 0.025,
) -> LambdaLR:
    """Linear decay from 1.0 to min_lr_ratio over total_steps."""
    def lr_lambda(step: int) -> float:
        progress = min(step / max(1, total_steps), 1.0)
        return 1.0 - (1.0 - min_lr_ratio) * progress

    return LambdaLR(optimizer, lr_lambda)
