"""Hardware detection — GPUs, memory, compute capability."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Optional

import psutil
import torch


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    compute_capability: tuple[int, int]
    is_available: bool = True


@dataclass
class SystemInfo:
    platform: str
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    gpus: list[GPUInfo] = field(default_factory=list)
    mps_available: bool = False
    best_device: str = "cpu"
    best_dtype: str = "float32"

    @property
    def total_vram_gb(self) -> float:
        return sum(g.memory_total_gb for g in self.gpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)


def detect_hardware() -> SystemInfo:
    """Detect available compute hardware and return a SystemInfo summary."""
    ram_total = psutil.virtual_memory().total / (1024**3)
    ram_available = psutil.virtual_memory().available / (1024**3)

    info = SystemInfo(
        platform=platform.system(),
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=round(ram_total, 2),
        ram_available_gb=round(ram_available, 2),
    )

    # CUDA GPUs
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_mem / (1024**3)
            mem_free = mem_total  # approximate at init
            try:
                mem_free = torch.cuda.mem_get_info(i)[0] / (1024**3)
            except Exception:
                pass
            info.gpus.append(
                GPUInfo(
                    index=i,
                    name=props.name,
                    memory_total_gb=round(mem_total, 2),
                    memory_free_gb=round(mem_free, 2),
                    compute_capability=(props.major, props.minor),
                )
            )
        info.best_device = "cuda"
        # bf16 needs compute >= 8.0
        if info.gpus and info.gpus[0].compute_capability >= (8, 0):
            info.best_dtype = "bfloat16"
        else:
            info.best_dtype = "float16"

    # Apple MPS
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info.mps_available = True
        info.best_device = "mps"
        info.best_dtype = "float16"

    else:
        info.best_device = "cpu"
        info.best_dtype = "float32"

    return info


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to the best available device string."""
    if device != "auto":
        return device
    return detect_hardware().best_device


def resolve_dtype(dtype: str = "auto") -> torch.dtype:
    """Resolve 'auto' to the best dtype for the current hardware."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype != "auto":
        return mapping.get(dtype, torch.float32)
    hw = detect_hardware()
    return mapping.get(hw.best_dtype, torch.float32)


def print_hardware_summary() -> SystemInfo:
    """Detect hardware and print a summary. Returns the SystemInfo."""
    info = detect_hardware()
    print(f"Platform:  {info.platform}")
    print(f"CPUs:      {info.cpu_count}")
    print(f"RAM:       {info.ram_available_gb:.1f} / {info.ram_total_gb:.1f} GB")
    if info.gpus:
        for gpu in info.gpus:
            print(
                f"GPU {gpu.index}:    {gpu.name}  "
                f"{gpu.memory_free_gb:.1f} / {gpu.memory_total_gb:.1f} GB  "
                f"(cc {gpu.compute_capability[0]}.{gpu.compute_capability[1]})"
            )
    elif info.mps_available:
        print("GPU:       Apple MPS (unified memory)")
    else:
        print("GPU:       None — CPU-only mode")
    print(f"Device:    {info.best_device}")
    print(f"Dtype:     {info.best_dtype}")
    return info
