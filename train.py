"""Efficient, resumable NanoMath training for Apple Silicon and CUDA."""

import argparse
from contextlib import nullcontext
import inspect
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config import PROFILES, get_device, get_hyperparams
from model_architecture.gpt_language_model import GPTLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_KEYS = {
    "vocab_size",
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
}


class BinaryDataset:
    """Memory-mapped token stream with vectorized random batch extraction."""

    def __init__(self, path, block_size, weights_path=None):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run the data pipeline first")
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.offsets = np.arange(block_size + 1, dtype=np.int64)
        self.weights = None
        if weights_path and weights_path.exists():
            self.weights = np.memmap(weights_path, dtype=np.uint8, mode="r")
            if len(self.weights) != len(self.tokens):
                raise ValueError(f"{weights_path} is not aligned with {path}")
        if len(self.tokens) <= block_size:
            raise ValueError(f"{path} needs more than {block_size} tokens")

    def batch(self, batch_size, device, rng):
        starts = rng.integers(0, len(self.tokens) - self.block_size - 1, size=batch_size)
        indices = starts[:, None] + self.offsets[None, :]
        token_batch = np.asarray(self.tokens[indices], dtype=np.int64)
        x = torch.from_numpy(token_batch[:, :-1])
        y = torch.from_numpy(token_batch[:, 1:])

        loss_weights = None
        if self.weights is not None:
            weight_batch = np.asarray(self.weights[indices[:, 1:]], dtype=np.float32)
            loss_weights = torch.from_numpy(weight_batch)

        if torch.device(device).type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
            if loss_weights is not None:
                loss_weights = loss_weights.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            if loss_weights is not None:
                loss_weights = loss_weights.to(device)
        return x, y, loss_weights


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="mac")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "corpus")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "build/token.model")
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Skip loading SentencePiece (useful in an offline Kaggle runtime)",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "build/model_weights.pth")
    starting_point = parser.add_mutually_exclusive_group()
    starting_point.add_argument("--resume", type=Path, help="Resume model and optimizer state")
    starting_point.add_argument(
        "--init-from",
        type=Path,
        help="Fine-tune model weights from a raw or metadata checkpoint",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="Train on shorter sequences without changing checkpoint architecture",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--compile", action="store_true", help="Useful on CUDA; usually slower to start on MPS")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        help="Stop cleanly and save before a hosted notebook session expires",
    )
    return parser.parse_args()


def autocast_context(device, dtype_name):
    if dtype_name == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    return torch.autocast(device_type=device, dtype=dtype)


def resolve_dtype(device, requested):
    if requested != "auto":
        return requested
    if torch.device(device).type == "cuda":
        # Recent PyTorch can report BF16 as supported through emulation on older
        # GPUs. Native BF16 Tensor Core training starts with Ampere (SM 8.x);
        # T4/P100 should use their hardware-accelerated FP16 path instead.
        capability_major, _ = torch.cuda.get_device_capability(device)
        return "bfloat16" if capability_major >= 8 else "float16"
    # MPS float32 is the safest default for long training. Users can explicitly
    # select float16 when memory is the limiting factor.
    return "float32"


def learning_rate_at(step, config):
    if step < config["warmup_iters"]:
        return config["learning_rate"] * (step + 1) / max(1, config["warmup_iters"])
    if step >= config["lr_decay_iters"]:
        return config["min_lr"]
    ratio = (step - config["warmup_iters"]) / max(
        1, config["lr_decay_iters"] - config["warmup_iters"]
    )
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config["min_lr"] + coefficient * (
        config["learning_rate"] - config["min_lr"]
    )


def make_optimizer(model, learning_rate, device):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {"lr": learning_rate, "betas": (0.9, 0.95)}
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = torch.device(device).type == "cuda"
    return torch.optim.AdamW(groups, **kwargs)


def cpu_state_dict(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith(("_orig_mod.", "module.")):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def setup_distributed(requested_device):
    """Initialize one NCCL process per GPU when launched through torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return get_device(requested_device), 0, 1, 0
    if requested_device not in {"auto", "cuda"}:
        raise SystemExit("Distributed training requires --device auto or cuda")
    if not torch.cuda.is_available():
        raise SystemExit("torchrun requested distributed training but CUDA is unavailable")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return f"cuda:{local_rank}", dist.get_rank(), world_size, local_rank


def load_vocab_size(args):
    if args.vocab_size is not None:
        if not 4 <= args.vocab_size <= np.iinfo(np.uint16).max + 1:
            raise SystemExit("--vocab-size must be between 4 and 65,536")
        return args.vocab_size
    if not args.tokenizer.exists():
        raise SystemExit(f"Missing tokenizer: {args.tokenizer}")
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise SystemExit(
            "sentencepiece is required unless --vocab-size is supplied"
        ) from error
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    return tokenizer.get_piece_size()


def validate_resume_config(checkpoint, model):
    saved = checkpoint.get("model_config")
    if not saved:
        return
    current = model.get_model_config()
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in MODEL_KEYS
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}={old!r} (checkpoint) vs {new!r} (requested)"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise SystemExit(f"Resume architecture does not match: {details}")


def main():
    args = parse_args()
    if args.init_from and args.init_from.resolve() == args.output.resolve():
        raise SystemExit("--output must differ from --init-from to preserve the source weights")
    device, rank, world_size, local_rank = setup_distributed(args.device)
    is_master = rank == 0
    device_type = torch.device(device).type
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(process_seed)
    rng = np.random.default_rng(process_seed)

    overrides = {"device": device}
    for argument, key in (
        (args.max_iters, "max_iters"),
        (args.batch_size, "batch_size"),
        (args.gradient_accumulation_steps, "gradient_accumulation_steps"),
        (args.eval_interval, "eval_interval"),
        (args.eval_iters, "eval_iters"),
        (args.learning_rate, "learning_rate"),
    ):
        if argument is not None:
            overrides[key] = argument
    config = get_hyperparams(args.profile, **overrides)
    config["lr_decay_iters"] = config["max_iters"]
    dtype_name = resolve_dtype(device, args.dtype)

    model_config = {key: value for key, value in config.items() if key in MODEL_KEYS}
    model_config["vocab_size"] = load_vocab_size(args)
    model = GPTLanguageModel(**model_config).to(device)
    optimizer = make_optimizer(model, config["learning_rate"], device)
    start_step = 0

    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(clean_state_dict(state_dict), strict=True)
        if is_master:
            print(f"Initialized weights from {args.init_from}")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "optimizer" not in checkpoint:
            raise SystemExit("This is a weights-only checkpoint; use --init-from instead of --resume")
        validate_resume_config(checkpoint, model)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        # Do not replay the beginning of each rank's random batch stream after a
        # hosted-session restart. The offset remains deterministic for a given step.
        resume_seed = process_seed + start_step * 1_000_003
        random.seed(resume_seed)
        torch.manual_seed(resume_seed)
        if device_type == "cuda":
            torch.cuda.manual_seed(resume_seed)
        rng = np.random.default_rng(resume_seed)
        if is_master:
            print(f"Resumed from step {start_step:,}")

    sequence_length = args.sequence_length or config["block_size"]
    if not 1 <= sequence_length <= config["block_size"]:
        raise ValueError(
            f"--sequence-length must be between 1 and {config['block_size']}"
        )
    train_data = BinaryDataset(
        args.data_dir / "train.bin",
        sequence_length,
        args.data_dir / "train_weights.bin",
    )
    val_data = BinaryDataset(
        args.data_dir / "val.bin",
        sequence_length,
        args.data_dir / "val_weights.bin",
    )

    raw_model = model
    if args.compile:
        if device_type == "mps":
            print("Warning: torch.compile has a high startup cost on MPS; continuing as requested.")
        model = torch.compile(model, mode=args.compile_mode)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device_type == "cuda" and dtype_name == "float16")
    )
    if args.resume and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    optimizer.zero_grad(set_to_none=True)

    @torch.inference_mode()
    def evaluate():
        model.eval()
        result = {}
        for name, dataset in (("train", train_data), ("val", val_data)):
            losses = []
            for _ in range(config["eval_iters"]):
                x, y, weights = dataset.batch(config["batch_size"], device, rng)
                with autocast_context(device_type, dtype_name):
                    _, loss = model(x, y, loss_weights=weights)
                losses.append(loss.detach().float())
            mean_loss = torch.stack(losses).mean()
            if world_size > 1:
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss /= world_size
            result[name] = mean_loss.item()
        model.train()
        return result

    def save_checkpoint(path, step, include_optimizer=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": cpu_state_dict(raw_model),
            "model_config": raw_model.get_model_config(),
            "train_config": config,
            "step": step,
            "world_size": world_size,
        }
        if include_optimizer:
            checkpoint["optimizer"] = optimizer.state_dict()
            checkpoint["scaler"] = scaler.state_dict()
        torch.save(checkpoint, path)

    parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    weighted = train_data.weights is not None
    if is_master:
        global_batch = (
            config["batch_size"]
            * config["gradient_accumulation_steps"]
            * world_size
        )
        gpu = torch.cuda.get_device_name(local_rank) if device_type == "cuda" else device
        print(
            f"Training {parameters / 1e6:.2f}M parameters on {world_size} x {gpu} "
            f"({dtype_name}); sequence {sequence_length}, global batch {global_batch}, "
            f"assistant-weighted loss: {weighted}"
        )
    started = time.perf_counter()
    log_started = started
    log_tokens = 0
    last_step = start_step - 1
    for step in range(start_step, config["max_iters"]):
        lr = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        accumulated_loss = torch.zeros((), device=device)
        accumulation_steps = config["gradient_accumulation_steps"]
        for micro_step in range(accumulation_steps):
            x, y, weights = train_data.batch(config["batch_size"], device, rng)
            sync_context = (
                model.no_sync()
                if world_size > 1 and micro_step < accumulation_steps - 1
                else nullcontext()
            )
            with sync_context:
                with autocast_context(device_type, dtype_name):
                    _, loss = model(x, y, loss_weights=weights)
                    scaled_loss = loss / accumulation_steps
                scaler.scale(scaled_loss).backward()
            accumulated_loss += loss.detach()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        last_step = step

        if device_type == "mps":
            torch.mps.synchronize()
        tokens = (
            config["batch_size"]
            * sequence_length
            * accumulation_steps
            * world_size
        )
        log_tokens += tokens
        if step % 10 == 0:
            if device_type == "cuda":
                torch.cuda.synchronize()
            log_elapsed = time.perf_counter() - log_started
            if is_master:
                print(
                    f"step {step:>6}: loss {(accumulated_loss / accumulation_steps).item():.4f}, "
                    f"lr {lr:.2e}, {log_tokens / max(log_elapsed, 1e-9):,.0f} tok/s"
                )
            log_started = time.perf_counter()
            log_tokens = 0
        if step % config["eval_interval"] == 0:
            losses = evaluate()
            if is_master:
                print(f"           eval train {losses['train']:.4f}, val {losses['val']:.4f}")
        if args.checkpoint_interval and step > 0 and step % args.checkpoint_interval == 0:
            if is_master:
                save_checkpoint(
                    args.output.with_name("latest_checkpoint.pth"),
                    step,
                    include_optimizer=True,
                )
            if world_size > 1:
                dist.barrier()
        should_stop = bool(
            args.max_runtime_minutes
            and (time.perf_counter() - started) / 60 >= args.max_runtime_minutes
        )
        if world_size > 1 and args.max_runtime_minutes:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            if is_master:
                print("Runtime limit reached; saving a resumable checkpoint.")
                save_checkpoint(
                    args.output.with_name("latest_checkpoint.pth"),
                    step,
                    include_optimizer=True,
                )
            break

    if is_master and last_step >= 0:
        save_checkpoint(args.output, last_step)
        print(f"Finished in {(time.perf_counter() - started) / 60:.1f} minutes; saved {args.output}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
