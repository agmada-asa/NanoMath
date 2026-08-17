"""Download and format GSM8K and an optional streamed NuminaMath subset."""

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "corpus"
SEPARATOR = "<|FILE_SEP|>"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-numina",
        type=int,
        help="Limit NuminaMath examples for a smaller local run; default streams all",
    )
    parser.add_argument("--skip-numina", action="store_true")
    return parser.parse_args()


def write_examples(path, examples, question_key, answer_key, limit=None):
    count = 0
    with path.open("w", encoding="utf-8") as destination:
        for example in examples:
            text = (
                f"<|user|> {example[question_key]} <|end|>\n"
                f"<|assistant|> {example[answer_key]} <|end|>"
            )
            destination.write(text + SEPARATOR)
            count += 1
            if limit is not None and count >= limit:
                break
    print(f"Wrote {count:,} examples to {path}")


def main():
    args = parse_args()
    from datasets import load_dataset

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading GSM8K...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    write_examples(CORPUS_DIR / "gsm8k_data.txt", gsm8k, "question", "answer")

    if not args.skip_numina:
        print("Streaming NuminaMath-CoT...")
        numina = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
        write_examples(
            CORPUS_DIR / "numina_math.txt",
            numina,
            "problem",
            "solution",
            limit=args.max_numina,
        )


if __name__ == "__main__":
    main()
