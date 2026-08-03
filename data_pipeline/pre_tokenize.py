"""Stream document files into aligned token and assistant-loss-weight binaries."""

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEPARATOR = b"<|FILE_SEP|>"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "corpus")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "build/token.model")
    return parser.parse_args()


def iter_records(path, chunk_size=1024 * 1024):
    """Split a potentially huge file without loading it into memory."""
    buffer = b""
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            buffer += chunk
            parts = buffer.split(SEPARATOR)
            buffer = parts.pop()
            for part in parts:
                if part.strip():
                    yield part.decode("utf-8")
    if buffer.strip():
        yield buffer.decode("utf-8")


def token_weights(ids, assistant_id, answer_id, end_id):
    if assistant_id not in ids:
        return [1] * len(ids)
    weights = []
    mode = 0
    for token_id in ids:
        if token_id == assistant_id:
            weights.append(0)
            mode = 1
        elif token_id == answer_id and mode:
            weights.append(1)
            mode = 3
        elif token_id == end_id and mode:
            weights.append(1)
            mode = 0
        else:
            weights.append(mode)
    return weights


def encode_split(tokenizer, input_path, token_path, weight_path):
    assistant_id = tokenizer.piece_to_id("<|assistant|>")
    answer_id = tokenizer.piece_to_id("<|answer|>")
    end_id = tokenizer.piece_to_id("<|end|>")
    total = documents = 0
    with token_path.open("wb") as token_file, weight_path.open("wb") as weight_file:
        for documents, record in enumerate(iter_records(input_path), start=1):
            ids = tokenizer.encode_as_ids(record)
            weights = token_weights(ids, assistant_id, answer_id, end_id)
            ids.append(tokenizer.eos_id())
            weights.append(1)
            np.asarray(ids, dtype=np.uint16).tofile(token_file)
            np.asarray(weights, dtype=np.uint8).tofile(weight_file)
            total += len(ids)
            if documents % 100_000 == 0:
                print(f"  {documents:,} documents / {total:,} tokens")
    print(f"{input_path.name}: {documents:,} documents / {total:,} tokens")


def main():
    args = parse_args()
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if tokenizer.get_piece_size() > np.iinfo(np.uint16).max:
        raise SystemExit("Tokenizer is too large for uint16 storage")
    for split in ("train", "val"):
        encode_split(
            tokenizer,
            args.corpus_dir / f"{split}.txt",
            args.corpus_dir / f"{split}.bin",
            args.corpus_dir / f"{split}_weights.bin",
        )


if __name__ == "__main__":
    main()
