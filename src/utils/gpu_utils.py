"""GPU optimization utilities."""

import os
import torch
from loguru import logger


def setup_gpu_optimizations() -> None:
    """Configure PyTorch for optimal GPU performance."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info(f"GPU optimizations enabled. Device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, using CPU")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_optimal_workers() -> int:
    """Get optimal number of dataloader workers."""
    cpu_count = os.cpu_count() or 4
    return min(cpu_count, 8)


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Compile model for faster inference."""
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    return model
