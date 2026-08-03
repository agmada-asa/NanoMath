"""Compile a document-level split and train NanoMath's math-aware tokenizer."""

import argparse
import csv
import mmap
from pathlib import Path
import random

import sentencepiece as spm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "corpus"
BUILD_DIR = PROJECT_ROOT / "build"
SEPARATOR = "<|FILE_SEP|>"
SEP_BYTES = SEPARATOR.encode()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument(
        "--reuse-tokenizer",
        action="store_true",
        help="Only compile/split text; keep an existing build/token.model for fine-tuning",
    )
    return parser.parse_args()


def convert_csv_to_text(csv_path, output_path):
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        reader = csv.reader(source)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            entry = (
                f"<|user|> {row[0]} <|end|>\n<|assistant|>\n"
                f"<|thinking|>\nOperation: {row[2]}\n{row[3]}\n"
                f"<|answer|> {row[1]} <|end|>"
            )
            destination.write(entry + SEPARATOR)


def build_index(paths):
    cards = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        print(f"Indexing {path.name}...")
        with path.open("rb") as source, mmap.mmap(
            source.fileno(), 0, access=mmap.ACCESS_READ
        ) as memory:
            start = 0
            while True:
                end = memory.find(SEP_BYTES, start)
                if end < 0:
                    if start < len(memory):
                        cards.append((path, start, len(memory) - start))
                    break
                if end > start:
                    cards.append((path, start, end - start))
                start = end + len(SEP_BYTES)
    return cards


def write_cards(cards, output_path):
    handles = {}
    try:
        with output_path.open("wb") as destination:
            for path, start, length in cards:
                if path not in handles:
                    handles[path] = path.open("rb")
                source = handles[path]
                source.seek(start)
                destination.write(source.read(length).strip())
                destination.write(SEP_BYTES)
    finally:
        for handle in handles.values():
            handle.close()


def main():
    args = parse_args()
    if not 0 < args.val_fraction < 0.5:
        raise SystemExit("--val-fraction must be between 0 and 0.5")
    corpus_dir = args.corpus_dir
    build_dir = args.build_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    csv_text = corpus_dir / "math_csv_temp.txt"
    convert_csv_to_text(corpus_dir / "MathCSV.csv", csv_text)
    sources = [
        corpus_dir / "synthetic_basic_math_cot.txt",
        corpus_dir / "gsm8k_data.txt",
        csv_text,
        corpus_dir / "numina_math.txt",
    ]
    cards = build_index(sources)
    if not cards:
        raise SystemExit("No corpus documents found. Run the download/generation steps first.")
    random.Random(args.seed).shuffle(cards)
    if args.max_documents:
        cards = cards[: args.max_documents]

    val_count = max(1, round(len(cards) * args.val_fraction))
    val_cards, train_cards = cards[:val_count], cards[val_count:]
    train_text, val_text = corpus_dir / "train.txt", corpus_dir / "val.txt"
    write_cards(train_cards, train_text)
    write_cards(val_cards, val_text)
    print(f"Document split: {len(train_cards):,} train / {len(val_cards):,} validation")
    if args.reuse_tokenizer:
        tokenizer_path = build_dir / "token.model"
        if not tokenizer_path.exists():
            raise SystemExit(f"--reuse-tokenizer requires {tokenizer_path}")
        print(f"Keeping existing tokenizer: {tokenizer_path}")
        return

    special_symbols = [
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "<|thinking|>",
        "<|answer|>",
        SEPARATOR,
        "+",
        "-",
        "*",
        "/",
        "=",
        "^",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "≤",
        "≥",
        "≠",
        "≈",
        "×",
        "÷",
        "±",
        "√",
        "%",
    ]
    print(f"Training {args.vocab_size:,}-piece math tokenizer...")
    spm.SentencePieceTrainer.train(
        input=str(train_text),
        model_prefix=str(build_dir / "token"),
        vocab_size=args.vocab_size,
        model_type="bpe",
        user_defined_symbols=special_symbols,
        input_sentence_size=3_000_000,
        shuffle_input_sentence=True,
        character_coverage=1.0,
        split_digits=True,
        byte_fallback=True,
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    print(f"Saved {build_dir / 'token.model'}")


if __name__ == "__main__":
    main()
