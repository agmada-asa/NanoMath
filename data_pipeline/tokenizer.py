"""
Trains a custom SentencePiece Byte-Pair Encoding (BPE) tokenizer on the downloaded and
synthetic mathematical datasets. Also compiles the data together and shuffles it.
"""

import sentencepiece as spm
import os
import csv
import random
import mmap

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus_dir = os.path.join(PROJECT_ROOT, "corpus")
build_dir = os.path.join(PROJECT_ROOT, "build")
final_data_path = os.path.join(corpus_dir, "Final_Data.txt")

# Input files
source_files = {
    "synthetic_math": os.path.join(corpus_dir, "synthetic_basic_math_cot.txt"),  # Our synthetic data
    "gsm8k": os.path.join(corpus_dir, "gsm8k_data.txt"),  # High-quality grade school math
    "csv_temp": os.path.join(corpus_dir, "math_csv_temp.txt"),  # Custom CSV data
    "numina_math": os.path.join(corpus_dir, "numina_math.txt"), # Numina Math Dataset
}

csv_source_path = os.path.join(corpus_dir, "MathCSV.csv")

# Ensure build directory exists
os.makedirs(build_dir, exist_ok=True)

# The separator used in download_data.py
SEPARATOR = "<|FILE_SEP|>"
SEP_BYTES = SEPARATOR.encode('utf-8')


# --- Helper Functions ---

def convert_csv_to_text(csv_path, output_txt_path):
    """Reads the CSV and saves it as a text file with separators so it can be indexed."""
    print(f"Pre-processing {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f_in, open(output_txt_path, "w", encoding="utf-8") as f_out:
        reader = csv.reader(f_in)
        next(reader, None)  # skip header

        count = 0
        for row in reader:
            if not row: continue
            # Updated to match our new CoT format
            entry = (
                f"<|user|> {row[0]}<|end|>\n"
                f"<|assistant|>\n"
                f"<|thinking|>\nOperation: {row[2]}\n{row[3]}\n"
                f"<|answer|> {row[1]}<|end|>"
            )
            f_out.write(entry + SEPARATOR)
            count += 1
    print(f"Converted {count} CSV rows to text format.")


def build_index(file_paths):
    """
    Scans files for SEPARATOR and returns a list of 'cards':
    (file_path, start_byte, length_bytes)
    """
    index_cards = []

    for path in file_paths:
        if not os.path.exists(path):
            print(f"WARNING: File not found: {path}")
            continue

        print(f"Indexing {os.path.basename(path)}... (this may take a moment)")

        with open(path, "r+b") as f:
            if os.path.getsize(path) == 0:
                continue

            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                start = 0
                while True:
                    end = mm.find(SEP_BYTES, start)

                    if end == -1:
                        if start < len(mm):
                            index_cards.append((path, start, len(mm) - start))
                        break

                    length = end - start
                    if length > 0:
                        index_cards.append((path, start, length))

                    start = end + len(SEP_BYTES)

    return index_cards


# --- Main Execution ---

# 1. Handle the CSV
convert_csv_to_text(csv_source_path, source_files["csv_temp"])

# 2. Build the Index
files_to_index = [path for path in source_files.values() if os.path.exists(path)]
all_cards = build_index(files_to_index)

print(f"Total chunks found: {len(all_cards)}")

# 3. Shuffle the Deck
print("Shuffling indices...")
random.shuffle(all_cards)

# 4. Write the Final File
print(f"Writing shuffled data to {final_data_path}...")
with open(final_data_path, "wb") as f_out:  # Write in binary mode

    # Open all source files in binary mode
    # We keep handles open to avoid opening/closing 100,000 times
    open_handles = {}
    for path in files_to_index:
        if os.path.exists(path):
            open_handles[path] = open(path, "rb")

    try:
        for i, (path, start, length) in enumerate(all_cards):
            if path not in open_handles: continue

            src = open_handles[path]
            src.seek(start)
            content = src.read(length)

            f_out.write(content)
            f_out.write(b'\n\n')

            if i % 10000 == 0:
                print(f"Processed {i}/{len(all_cards)}...", end='\r')

    finally:
        for h in open_handles.values():
            h.close()

print("\nData preparation complete!")

special_symbols = [
    "<|user|>", "<|assistant|>", "<|end|>", "<|thinking|>", "<|answer|>", # Chat format tags
    "+", "-", "*", "/", "=", "^", # Basic operators
    "(", ")", "[", "]", "{", "}", # Brackets
    "<", ">", "≤", "≥", "≠", "≈", # Inequalities
    "×", "÷", "±", "√", "%", ".", "," # Advanced/Alternate operators
]

# 5. Train Tokenizer
print("Starting tokenizer training...")
spm.SentencePieceTrainer.Train(
    input=final_data_path,
    model_prefix=os.path.join(build_dir, "token"),
    vocab_size=32768,
    model_type="bpe", # Use a deterministic model type
    user_defined_symbols=special_symbols,
    input_sentence_size=3000000, # allow for huge input files
    shuffle_input_sentence=True,
    character_coverage=1.0, # 100% coverage ensures no more '⁇' symbols
    split_digits=True, # Forces '428' to become '4', '2', '8'
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

print("Tokenizer Complete! Saved to build/token.model")