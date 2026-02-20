import torch


def get_hyperparams():
    # --- Settings you can change ---
    batch_size = 8  # Physical batch size (fits in 16GB VRAM with 1024 block size)
    max_iters = 10000  # We don't have enough compute time on Kaggle for this before you reach the 12 hour limit, ~5000 is achievable in one go
    learning_rate = 3e-4

    # Target effective batch size: 128 sequences
    target_batch_size = 128
    grad_accum = target_batch_size // batch_size

    CONFIG = {
        "batch_size": batch_size,
        "block_size": 1024,
        "max_iters": max_iters,
        "eval_interval": 20,
        "learning_rate": learning_rate,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu', # Train using GPU if available, else CPU (or you could change to 'mps' for Mac)
        "eval_iters": 20,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "dropout": 0.05,
        "gradient_accumulation_steps": grad_accum,
        "lr_decay_iters": max_iters,
        "min_lr": learning_rate / 10,
        "warmup_iters": 500
    }
    return CONFIG