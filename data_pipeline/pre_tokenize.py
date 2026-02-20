"""
Pre-tokenizes the entire compiled text dataset into binary (.bin) files (train/val splits).
This allows for highly efficient streaming directly from disk to GPU during training.
"""

import os
import sentencepiece as spm
import numpy as np

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus_dir = os.path.join(PROJECT_ROOT, "corpus")
build_dir = os.path.join(PROJECT_ROOT, "build")
input_file = os.path.join(corpus_dir, "Final_Data.txt")
tokenizer_path = os.path.join(build_dir, "token.model")

# Output files
temp_file = os.path.join(corpus_dir, "temp_all.bin")
train_file = os.path.join(corpus_dir, "train.bin")
val_file = os.path.join(corpus_dir, "val.bin")

# 1. Load the Tokenizer
print(f"Loading tokenizer from {tokenizer_path}...")
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)
vocab_size = sp.get_piece_size()

# Safety check for uint16
if vocab_size > 65535:
    raise ValueError(f"Vocab size {vocab_size} is too large for uint16!")

# 2. Pass 1: Tokenize Everything to a Temp File
print(f"Phase 1: Tokenizing {input_file} into a single binary...")
token_count = 0

with open(input_file, "r", encoding="utf-8") as f_in, \
        open(temp_file, "wb") as f_out:
    for i, line in enumerate(f_in):
        # We assume the line contains valuable newlines/formatting
        ids = sp.encode_as_ids(line)

        # Convert to numpy uint16
        data = np.array(ids, dtype=np.uint16)

        # Write bytes directly to disk
        f_out.write(data.tobytes())

        token_count += len(ids)

        if i % 100000 == 0:
            print(f"Tokenizing line {i}...", end='\r')

print(f"\nTokenization complete. Total tokens: {token_count:,}")

# 3. Pass 2: Split into Train/Val
print("Phase 2: Splitting into Train (90%) and Val (10%)...")

# Calculate the split index
n = int(0.9 * token_count)

# We use memmap to open the huge file as if it were an array in RAM
# mode='r' means read-only, we won't modify the temp file
all_tokens = np.memmap(temp_file, dtype=np.uint16, mode='r', shape=(token_count,))

# Save the slices
# We use .tofile() to save the raw binary data
print(f"Writing {n:,} tokens to {train_file}...")
train_data = all_tokens[:n]
with open(train_file, "wb") as f:
    f.write(train_data.tobytes())

print(f"Writing {token_count - n:,} tokens to {val_file}...")
val_data = all_tokens[n:]
with open(val_file, "wb") as f:
    f.write(val_data.tobytes())

print("\nDone! Binary datasets are ready.")