"""Central model/training profiles.

`legacy` exactly describes the published 136M checkpoint. `mac` is the recommended
from-scratch profile for Apple Silicon. `kaggle` scales the same modern architecture
to a size that trains efficiently on one or two 16 GB Kaggle GPUs.
"""

import os

import torch


def get_device(preferred="auto"):
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


PROFILES = {
    "legacy": {
        "batch_size": 8,
        "block_size": 1024,
        "max_iters": 10000,
        "eval_interval": 100,
        "eval_iters": 20,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 500,
        "n_embd": 768,
        "n_head": 12,
        "n_kv_head": 12,
        "n_layer": 12,
        "dropout": 0.05,
        "gradient_accumulation_steps": 16,
        "norm_type": "layernorm",
        "mlp_type": "relu",
        "position_encoding": "learned",
        "tie_embeddings": False,
        "bias": True,
    },
    "mac": {
        "batch_size": 2,
        "block_size": 512,
        "max_iters": 10000,
        "eval_interval": 250,
        "eval_iters": 10,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 200,
        "n_embd": 512,
        "n_head": 8,
        "n_kv_head": 4,
        "n_layer": 8,
        "dropout": 0.05,
        "gradient_accumulation_steps": 16,
        "norm_type": "rmsnorm",
        "mlp_type": "swiglu",
        "position_encoding": "rope",
        "tie_embeddings": True,
        "bias": False,
    },
    "kaggle": {
        # Per-GPU batch. With two GPUs DDP doubles the global token batch while
        # gradient accumulation keeps memory use predictable on 16 GB T4s.
        "batch_size": 4,
        "block_size": 1024,
        "max_iters": 6000,
        "eval_interval": 250,
        "eval_iters": 10,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 300,
        "n_embd": 768,
        "n_head": 12,
        "n_kv_head": 4,
        "n_layer": 16,
        "dropout": 0.0,
        "gradient_accumulation_steps": 8,
        "norm_type": "rmsnorm",
        "mlp_type": "swiglu",
        "position_encoding": "rope",
        "tie_embeddings": True,
        "bias": False,
    },
    "tiny": {
        "batch_size": 2,
        "block_size": 128,
        "max_iters": 100,
        "eval_interval": 25,
        "eval_iters": 4,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "warmup_iters": 10,
        "n_embd": 128,
        "n_head": 4,
        "n_kv_head": 2,
        "n_layer": 4,
        "dropout": 0.0,
        "gradient_accumulation_steps": 2,
        "norm_type": "rmsnorm",
        "mlp_type": "swiglu",
        "position_encoding": "rope",
        "tie_embeddings": True,
        "bias": False,
    },
}


def get_hyperparams(profile=None, **overrides):
    profile = profile or os.environ.get("NANOMATH_PROFILE", "legacy")
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    config = dict(PROFILES[profile])
    config.update(overrides)
    config["device"] = get_device(config.get("device", "auto"))
    config["lr_decay_iters"] = config.get("lr_decay_iters", config["max_iters"])
    return config
