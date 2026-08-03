"""Package local NanoMath binaries and source into an upload-ready Kaggle dataset."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

from config import PROFILES


PROJECT_ROOT = Path(__file__).resolve().parent
BINARY_FILES = {
    "train.bin": np.dtype("uint16"),
    "val.bin": np.dtype("uint16"),
    "train_weights.bin": np.dtype("uint8"),
    "val_weights.bin": np.dtype("uint8"),
}
SOURCE_PATHS = (
    "train.py",
    "chat.py",
    "config.py",
    "requirements.txt",
    "README.md",
    "model_architecture",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "corpus")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "build/token.model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "build/kaggle_dataset",
    )
    parser.add_argument(
        "--dataset-id",
        default="agmadaasa/nanomath-kaggle-v2",
        help="Kaggle owner/slug written to dataset-metadata.json",
    )
    parser.add_argument("--title", default="NanoMath Kaggle Training Build v2")
    parser.add_argument("--force", action="store_true", help="Replace an existing package directory")
    return parser.parse_args()


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path, dtype=None):
    record = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if dtype is not None:
        if path.stat().st_size % dtype.itemsize:
            raise ValueError(f"{path} is not aligned to {dtype}")
        record.update(
            dtype=dtype.name,
            tokens=path.stat().st_size // dtype.itemsize,
        )
    return record


def tokenizer_vocab_size(path):
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise SystemExit("Install sentencepiece before packaging the tokenizer") from error
    tokenizer = spm.SentencePieceProcessor(model_file=str(path))
    return tokenizer.get_piece_size()


def reset_output_directory(path, force):
    resolved = path.resolve()
    protected = {PROJECT_ROOT.resolve(), PROJECT_ROOT.parent.resolve(), Path.home().resolve()}
    if resolved in protected:
        raise SystemExit(f"Refusing to use protected directory as package output: {resolved}")
    if path.exists():
        if not force:
            raise SystemExit(f"{path} already exists; pass --force to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def create_source_archive(destination):
    def exclude_build_artifacts(info):
        parts = Path(info.name).parts
        if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
            return None
        return info

    with tarfile.open(destination, "w:gz") as archive:
        for relative_name in SOURCE_PATHS:
            source = PROJECT_ROOT / relative_name
            if not source.exists():
                raise FileNotFoundError(source)
            archive.add(
                source,
                arcname=relative_name,
                recursive=True,
                filter=exclude_build_artifacts,
            )


def source_file_records():
    records = {}
    for relative_name in SOURCE_PATHS:
        source = PROJECT_ROOT / relative_name
        paths = source.rglob("*") if source.is_dir() else (source,)
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            relative_path = path.relative_to(PROJECT_ROOT)
            records[str(Path("nanomath-source") / relative_path)] = file_record(path)
    return records


def main():
    args = parse_args()
    if args.dataset_id.count("/") != 1 or any(not part for part in args.dataset_id.split("/")):
        raise SystemExit("--dataset-id must have the form owner/slug")
    if not args.tokenizer.exists():
        raise FileNotFoundError(args.tokenizer)
    missing = [args.corpus_dir / name for name in BINARY_FILES if not (args.corpus_dir / name).exists()]
    if missing:
        formatted = ", ".join(map(str, missing))
        raise FileNotFoundError(
            f"Missing corpus binaries: {formatted}; run data_pipeline/complete_pipeline.py first"
        )
    for relative_name in SOURCE_PATHS:
        if not (PROJECT_ROOT / relative_name).exists():
            raise FileNotFoundError(PROJECT_ROOT / relative_name)
    reset_output_directory(args.output_dir, args.force)

    records = {}
    for name, dtype in BINARY_FILES.items():
        source = args.corpus_dir / name
        destination = args.output_dir / name
        shutil.copy2(source, destination)
        records[name] = file_record(destination, dtype)

    for split in ("train", "val"):
        tokens = records[f"{split}.bin"]["tokens"]
        weights = records[f"{split}_weights.bin"]["tokens"]
        if tokens != weights:
            raise ValueError(f"{split} tokens and weights are not aligned: {tokens} != {weights}")

    tokenizer_destination = args.output_dir / "token.model"
    shutil.copy2(args.tokenizer, tokenizer_destination)
    records["token.model"] = file_record(tokenizer_destination)

    source_archive = args.output_dir / "nanomath-source.tar.gz"
    create_source_archive(source_archive)
    records[source_archive.name] = file_record(source_archive)

    profile = dict(PROFILES["kaggle"])
    manifest = {
        "format": "nanomath-packed-token-stream",
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {
            "file": "token.model",
            "vocab_size": tokenizer_vocab_size(tokenizer_destination),
            "token_dtype": "uint16",
            "weight_dtype": "uint8",
            "weight_meanings": {"0": "ignored prompt", "1": "reasoning", "3": "final answer"},
        },
        "recommended_training": {"profile": "kaggle", **profile},
        "files": records,
        # Kaggle expands the source archive under this prefix. Recording each
        # resulting file lets the notebook verify either Kaggle's mounted layout
        # or the original archive used by other runtimes.
        "expanded_source_files": source_file_records(),
    }
    manifest_path = args.output_dir / "nanomath-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "title": args.title,
        "id": args.dataset_id,
        "licenses": [{"name": "other"}],
    }
    (args.output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    total_bytes = sum(record["bytes"] for record in records.values())
    print(f"Kaggle dataset ready: {args.output_dir}")
    print(f"Files: {len(records):,}; payload: {total_bytes / 1024**3:.2f} GiB")
    print(f"Vocab: {manifest['tokenizer']['vocab_size']:,} pieces")
    print(f"Upload: python3 -m kaggle datasets create -p {args.output_dir}")
    print(
        f"Update: python3 -m kaggle datasets version -p {args.output_dir} "
        "-m 'refresh training corpus'"
    )


if __name__ == "__main__":
    main()
