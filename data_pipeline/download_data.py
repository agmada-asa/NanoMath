"""
Downloads the GSM8K and NuminaMath-CoT datasets and saves them into the 'corpus' directory
in a formatted structure ready for tokenization.
"""

import os
from datasets import load_dataset

# Resolve paths correctly from anywhere by referencing the parent directory of this script's directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus_path = os.path.join(PROJECT_ROOT, 'corpus')
os.makedirs(corpus_path, exist_ok=True)

# File Paths
gsm8k_path = os.path.join(corpus_path, 'gsm8k_data.txt')
numina_path = os.path.join(corpus_path, 'numina_math.txt')

SEPARATOR = "<|FILE_SEP|>"

def save_to_file(path, formatted_text):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(formatted_text + SEPARATOR)

# 1. GSM8K - Gold Standard Grade School Math
print("Downloading GSM8K...")
gsm8k = load_dataset("openai/gsm8k", "main", split="train")
for ex in gsm8k:
    text = f"<|user|> {ex['question']}<|end|>\n<|assistant|> {ex['answer']}<|end|>"
    save_to_file(gsm8k_path, text)

# 2. NuminaMath-CoT - 860k Reasoning Problems
print("Downloading NuminaMath-CoT (Streaming)...")
numina = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
for i, ex in enumerate(numina):
    # Numina typically has 'problem' and 'solution' fields
    text = f"<|user|> {ex['problem']}<|end|>\n<|assistant|> {ex['solution']}<|end|>"
    save_to_file(numina_path, text)

print("Done!")