"""Low-impact inference benchmark for NanoMath model profiles."""

import argparse
import time

import torch

from config import PROFILES, get_device, get_hyperparams
from model_architecture.gpt_language_model import GPTLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="legacy")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=32768)
    return parser.parse_args()


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    args = parse_args()
    config = get_hyperparams(args.profile)
    device = get_device(args.device)
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "float16" if device in {"mps", "cuda"} else "float32"
    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
    model_keys = (
        "n_embd",
        "block_size",
        "n_head",
        "n_layer",
        "dropout",
        "n_kv_head",
        "norm_type",
        "mlp_type",
        "position_encoding",
        "tie_embeddings",
        "bias",
    )
    model = GPTLanguageModel(
        vocab_size=args.vocab_size, **{key: config[key] for key in model_keys}
    ).to(device=device, dtype=dtype).eval()
    context = torch.randint(
        args.vocab_size, (1, min(args.prompt_tokens, model.block_size)), device=device
    )
    print(
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters "
        f"on {device} ({dtype_name})"
    )
    with torch.inference_mode():
        for use_cache in (False, True):
            model.generate(context, 2, use_cache=use_cache)
            synchronize(device)
            started = time.perf_counter()
            model.generate(context, args.new_tokens, use_cache=use_cache)
            synchronize(device)
            elapsed = time.perf_counter() - started
            label = "KV cache" if use_cache else "full recompute"
            print(f"{label:>14}: {args.new_tokens / elapsed:7.1f} tokens/s ({elapsed:.3f}s)")


if __name__ == "__main__":
    main()
