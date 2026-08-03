"""Run the NanoMath data pipeline with full or laptop-friendly dataset sizes."""

import argparse
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(script, *arguments):
    command = [sys.executable, str(SCRIPT_DIR / script), *map(str, arguments)]
    print(f"\n========== RUNNING {script} ==========")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Build a useful smoke-test corpus (10k synthetic + GSM8K + 10k Numina)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    download_args = ["--max-numina", "10000"] if args.quick else []
    synthetic_count = 10_000 if args.quick else 1_000_000
    tokenizer_args = ["--max-documents", "30000"] if args.quick else []

    run_step("download_data.py", *download_args)
    run_step(
        "generate_math_problems.py",
        "--num-samples",
        synthetic_count,
        "--seed",
        args.seed,
    )
    run_step("tokenizer.py", *tokenizer_args, "--seed", args.seed)
    run_step("pre_tokenize.py")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
