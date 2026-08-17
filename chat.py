"""Local NanoMath CLI with Apple-Silicon acceleration and cached decoding."""

import argparse
from pathlib import Path
import time

import sentencepiece as spm
import torch

from config import get_device, get_hyperparams
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


def _strip_wrapper_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith(("_orig_mod.", "module.")):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def load_model(weights_path, tokenizer, device, dtype):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
        model_config = dict(checkpoint["model_config"])
    else:
        # Published NanoMath weights predate metadata checkpoints.
        state_dict = checkpoint
        model_config = get_hyperparams("legacy")

    state_dict = _strip_wrapper_prefixes(state_dict)
    token_weight = state_dict.get("token_embedding_table.weight")
    if token_weight is not None:
        model_config["vocab_size"] = token_weight.shape[0]
        model_config["n_embd"] = token_weight.shape[1]
    else:
        vocab_size = tokenizer.get_piece_size()
        model_config["vocab_size"] = (vocab_size + 127) // 128 * 128

    position_weight = state_dict.get("position_embedding_table.weight")
    if position_weight is not None:
        model_config["block_size"] = position_weight.shape[0]
    model_args = {key: value for key, value in model_config.items() if key in MODEL_KEYS}
    model_args["dropout"] = 0.0

    model = GPTLanguageModel(**model_args)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype).eval()
    return model, model_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "build/model_weights.pth")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "build/token.model")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 uses deterministic greedy decoding (recommended for maths)",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-kv-cache", action="store_true", help="Debug the slower legacy decoder")
    parser.add_argument("--show-speed", action="store_true")
    parser.add_argument("prompt", nargs="*", help="Run one prompt instead of opening interactive mode")
    return parser.parse_args()


def main():
    args = parse_args()
    one_shot_prompt = " ".join(args.prompt) if args.prompt else None
    device = get_device(args.device)
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "float16" if device in {"mps", "cuda"} else "float32"
    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
    if not args.tokenizer.exists() or not args.weights.exists():
        raise SystemExit(
            "Missing model files. Put token.model and model_weights.pth in build/, "
            "or pass --tokenizer and --weights. See README.md for the download command."
        )

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    print(f"Loading {args.weights.name} on {device}...")
    model, model_config = load_model(args.weights, tokenizer, device, dtype)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Ready: {parameters / 1e6:.2f}M parameters, {dtype_name}, "
        f"{model_config.get('position_encoding', 'learned')} positions, KV cache on"
    )

    end_id = tokenizer.piece_to_id("<|end|>")
    stop_ids = None if end_id == tokenizer.unk_id() else [end_id]

    def answer(user_input):
        prompt = f"<|user|> {user_input} <|end|>\n<|assistant|>"
        prompt_ids = tokenizer.encode_as_ids(prompt)
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        started = time.perf_counter()
        generated = model.generate(
            context,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_token_ids=stop_ids,
            use_cache=not args.no_kv_cache,
        )
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        new_tokens = generated[0, context.size(1) :].tolist()
        text = tokenizer.decode(new_tokens).split("<|end|>", 1)[0].strip()
        print(f"NanoMath: {text}")
        if args.show_speed:
            print(f"[{len(new_tokens) / max(elapsed, 1e-9):.1f} tokens/s]")

    if one_shot_prompt:
        answer(one_shot_prompt)
        return

    print("Type 'quit' to exit. Greedy decoding is enabled for arithmetic.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input:
            answer(user_input)
            print()


if __name__ == "__main__":
    main()
