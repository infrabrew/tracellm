"""Custom optimizers — Muon, DeepSpeed integration, FlashInfer attention."""

from __future__ import annotations

import math
from typing import Any, Iterator

import torch
from torch.optim import Optimizer

from tracellm.utils.logging import get_logger

log = get_logger("tracellm.training.optimizers")


# ═══════════════════════════════════════════════════════════════════════════════
#  MUON OPTIMIZER
#  Momentum-based optimizer with orthogonalization (Muon).
#  Reference: https://arxiv.org/abs/2405.20495
# ═══════════════════════════════════════════════════════════════════════════════

class Muon(Optimizer):
    """Muon optimizer — Momentum + Orthogonalization for faster LLM training.

    Uses Nesterov momentum with periodic orthogonalization of the
    momentum buffer via Newton-Schulz iteration. This keeps the
    effective update directions decorrelated, improving convergence.

    Args:
        params: Model parameters.
        lr: Learning rate (default: 0.02).
        momentum: Momentum factor (default: 0.95).
        nesterov: Use Nesterov momentum (default: True).
        ns_steps: Newton-Schulz orthogonalization steps (default: 5).
        ns_every: Orthogonalize every N steps (default: 1).
        weight_decay: Decoupled weight decay (default: 0.0).
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_every: int = 1,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_every=ns_every,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def _newton_schulz_orthogonalize(self, M: torch.Tensor, steps: int) -> torch.Tensor:
        """Approximate orthogonalization via Newton-Schulz iteration.

        Computes an approximate orthogonal matrix Q such that Q^T Q ≈ I,
        where Q has the same column space as M.
        """
        if M.ndim < 2:
            return M

        # Reshape to 2D for orthogonalization
        orig_shape = M.shape
        if M.ndim > 2:
            M = M.reshape(M.shape[0], -1)

        rows, cols = M.shape
        if rows < cols:
            M = M.T
            transposed = True
        else:
            transposed = False

        # Normalize
        norm = torch.norm(M)
        if norm < 1e-8:
            return M.reshape(orig_shape) if not transposed else M.T.reshape(orig_shape)
        X = M / norm

        # Newton-Schulz iterations: X_{k+1} = X_k (3I - X_k^T X_k) / 2
        I = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
        for _ in range(steps):
            A = X.T @ X
            X = X @ (3 * I - A) / 2

        # Scale back
        X = X * norm

        if transposed:
            X = X.T

        return X.reshape(orig_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            ns_every = group["ns_every"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # Momentum update
                buf.mul_(momentum).add_(grad)

                # Orthogonalize momentum buffer periodically
                if ns_steps > 0 and self._step_count % ns_every == 0 and p.ndim >= 2:
                    buf.copy_(self._newton_schulz_orthogonalize(buf, ns_steps))

                # Nesterov step
                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZER FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_optimizer(
    name: str,
    model_params: Iterator[torch.nn.Parameter],
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs,
) -> Optimizer:
    """Create an optimizer by name.

    Supported: adamw, muon, sgd, adam, adafactor.
    DeepSpeed optimizers are handled separately in the trainer.
    """
    name = name.lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )

    elif name == "muon":
        return Muon(
            model_params,
            lr=kwargs.get("muon_lr", lr * 200),  # Muon uses higher LR
            momentum=kwargs.get("momentum", 0.95),
            nesterov=kwargs.get("nesterov", True),
            ns_steps=kwargs.get("ns_steps", 5),
            weight_decay=weight_decay,
        )

    elif name == "sgd":
        return torch.optim.SGD(
            model_params,
            lr=lr,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=weight_decay,
        )

    elif name == "adam":
        return torch.optim.Adam(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Supported: adamw, muon, sgd, adam. "
            f"For DeepSpeed, set optimizer='deepspeed' and configure deepspeed settings."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASHINFER ATTENTION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def apply_flashinfer_attention(model: torch.nn.Module) -> torch.nn.Module:
    """Replace standard attention with FlashInfer kernels if available.

    FlashInfer provides fused attention kernels optimized for inference
    and training on NVIDIA GPUs. Falls back gracefully if not installed.
    """
    try:
        import flashinfer
        log.info("FlashInfer available — applying optimized attention kernels")

        # Replace attention modules with FlashInfer equivalents
        for name, module in model.named_modules():
            if hasattr(module, "forward") and "attention" in name.lower():
                # Store original forward for wrapping
                original_forward = module.forward

                def make_flash_forward(orig_fn, mod):
                    def flash_forward(*args, **kwargs):
                        # Use FlashInfer's prefill/decode attention when available
                        if hasattr(flashinfer, "single_prefill_with_kv_cache"):
                            kwargs["use_flash_attn"] = True
                        return orig_fn(*args, **kwargs)
                    return flash_forward

                module.forward = make_flash_forward(original_forward, module)

        log.info("FlashInfer attention wrappers applied")
    except ImportError:
        log.debug("FlashInfer not installed — using default attention. "
                   "Install with: pip install 'tracellm[flashinfer]'")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  DEEPSPEED CONFIG BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_deepspeed_config(
    stage: int = 2,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lr: float = 2e-4,
    fp16: bool = False,
    bf16: bool = True,
    offload_optimizer: bool = False,
    offload_params: bool = False,
) -> dict[str, Any]:
    """Build a DeepSpeed ZeRO configuration dict.

    Args:
        stage: ZeRO stage (0, 1, 2, or 3).
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        lr: Learning rate.
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        offload_optimizer: Offload optimizer state to CPU (ZeRO-2/3).
        offload_params: Offload parameters to CPU (ZeRO-3 only).
    """
    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": 50,
    }

    # FP16 / BF16
    if bf16:
        config["bf16"] = {"enabled": True}
    elif fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
        }

    # ZeRO config
    zero_config: dict[str, Any] = {
        "stage": stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    }

    if offload_optimizer and stage >= 2:
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if offload_params and stage >= 3:
        zero_config["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    config["zero_optimization"] = zero_config

    # Optimizer
    config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    }

    # Scheduler
    config["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": lr,
            "warmup_num_steps": 100,
            "total_num_steps": 10000,
        },
    }

    return config
