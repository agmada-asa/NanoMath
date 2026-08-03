# NanoMath

NanoMath is a GPT-style maths model with a complete data, training, and inference stack. This version preserves compatibility with the published 136M checkpoint while adding faster Apple-Silicon inference, a local corpus builder, and a self-contained single/dual-GPU Kaggle training build.

## Run it locally

This checkout already has the published files in `build/` (they are intentionally ignored by Git). On a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf download agmadaasa/NanoMath model_weights.pth token.model --local-dir build
```

Ask one question:

```bash
python chat.py "What is 48 / 6?"
```

Or start an interactive session:

```bash
python chat.py
```

The CLI automatically chooses CUDA, Apple MPS, or CPU. It uses deterministic greedy decoding by default because sampling is usually counterproductive for arithmetic. Every answer comes directly from the model weights; there is no calculator, parser, or symbolic-solver fallback.

Useful options:

```bash
python chat.py --show-speed "Explain why 3/4 is larger than 2/3"
python chat.py --temperature 0.5 --top-k 40 "Write a new maths problem"
python chat.py --no-kv-cache --show-speed "What is 48 / 6?"
```

## What was optimized

### Inference

- Per-layer KV caching avoids recomputing the full prompt for every generated token.
- Generation stops on the token ID directly instead of generating all 200 tokens and trimming text afterward.
- Greedy decoding avoids a full softmax and multinomial sample when temperature is zero.
- FP16 inference is selected automatically on MPS/CUDA, roughly halving weight memory; `--dtype float32` remains available.
- Model positions follow the input tensor's actual device, so moving between CPU, MPS, and CUDA is reliable.
- Both raw legacy checkpoints and new metadata-rich resumable checkpoints load automatically.

Low-impact benchmark on an 18 GB M3 Pro, using the full 136.18M architecture in FP16, a 128-token prompt, and 12 generated tokens:

| Decoder | Tokens/second | Relative speed |
|---|---:|---:|
| Full-context recomputation | 41.7 | 1.0x |
| KV cache | 258.5 | 6.2x |

Run the same short benchmark on another machine:

```bash
python benchmark.py --profile legacy --new-tokens 12
```

### Training

`train.py` replaces the notebook-only training loop and adds:

- native MPS, CUDA, and CPU execution;
- vectorized random batches from memory-mapped token files;
- assistant/answer-weighted loss when weight files are present;
- correct AdamW decay groups, gradient clipping, cosine decay, and warmup;
- CUDA BF16/FP16 autocast and fused AdamW where supported;
- resumable checkpoints containing model and training configuration;
- optional `torch.compile` (normally best on CUDA; deliberately off by default on MPS);
- automatic multi-GPU DDP when launched with `torchrun`, without redundant gradient synchronization during accumulation;
- native grouped-query CUDA attention on recent PyTorch builds, with a portable fallback;
- an optional wall-clock cutoff that writes both resumable and inference checkpoints before Kaggle expires;
- small validation runs and infrequent checkpoints to reduce disruption.

Four profiles are available:

| Profile | Parameters with intended vocab | Purpose |
|---|---:|---|
| `legacy` | 136.18M / 32,768 pieces | Published checkpoint compatibility |
| `mac` | 32.0M / 16,384 pieces | Recommended Apple-Silicon training |
| `kaggle` | About 113M / 16,384 pieces | New 16-layer CUDA architecture for one or two 16 GB GPUs |
| `tiny` | Under 5M | Pipeline tests and experimentation |

The `mac` profile uses RoPE, RMSNorm, SwiGLU, grouped-query attention, bias-free projections, tied input/output embeddings, and a 512-token context. It is a new architecture and cannot load the legacy weights; train it from scratch. Its smaller vocabulary and tied embeddings save a large fraction of the original model's output-head compute, which matters on a laptop.

### Mathematical learning signal

- Training and validation are split by complete problems rather than at an arbitrary token boundary.
- Each example ends with EOS, preventing unrelated problems from running together.
- Prompt tokens receive zero loss weight, reasoning tokens weight 1, and final-answer tokens weight 3. This spends capacity on solving rather than memorizing user text.
- The synthetic curriculum now includes exact fractions, percentages, and verified linear equations in addition to arithmetic and word problems.
- Synthetic generation and document shuffling are seeded and reproducible.
- The default tokenizer is reduced from 32K to 16K, keeps digits split, reserves maths symbols, and has byte fallback.
- Downloads overwrite prior corpus files instead of silently duplicating examples on repeated runs.

These changes affect mathematical ability only after training or fine-tuning; inference never substitutes an externally calculated answer.

## Build training data

The quick pipeline creates a practical development corpus: GSM8K, 10,000 streamed NuminaMath examples, and 10,000 synthetic examples.

```bash
python data_pipeline/complete_pipeline.py --quick
```

The complete pipeline is much larger and can take substantial time and disk space:

```bash
python data_pipeline/complete_pipeline.py
```

Individual stages remain configurable:

```bash
python data_pipeline/download_data.py --max-numina 50000
python data_pipeline/generate_math_problems.py --num-samples 100000 --seed 1337
python data_pipeline/tokenizer.py --vocab-size 16384 --seed 1337
python data_pipeline/pre_tokenize.py
```

The resulting files are `corpus/train.bin`, `corpus/val.bin`, and aligned `*_weights.bin` files. All large build artifacts are ignored by Git.

## Train on Kaggle

Build the corpus and tokenizer locally, then create a flat, upload-ready Kaggle dataset:

```bash
python data_pipeline/complete_pipeline.py
python prepare_kaggle_dataset.py \
  --dataset-id agmadaasa/nanomath-kaggle-v2
```

The package is written to `build/kaggle_dataset/`. It contains:

- `train.bin` and `val.bin`: packed little-endian `uint16` token streams;
- aligned `train_weights.bin` and `val_weights.bin`: `uint8` loss weights;
- `token.model` and a manifest containing the vocabulary size, token counts, dtypes, profile, sizes, and SHA-256 checksums;
- `nanomath-source.tar.gz`: the exact trainer and model source needed by the offline notebook;
- `dataset-metadata.json`: metadata consumed by the Kaggle CLI.

Upload it with your configured Kaggle credentials:

```bash
python3 -m kaggle datasets create -p build/kaggle_dataset
```

If that dataset slug already exists, publish the package as a new version instead:

```bash
python3 -m kaggle datasets version -p build/kaggle_dataset -m "refresh training corpus"
```

Then import `kaggle-notebook.ipynb` into Kaggle, attach that private dataset, select a GPU accelerator, and run all cells. The notebook verifies the package, extracts the matching source, and uses one GPU directly or every visible GPU through DDP. Its defaults train the `kaggle` profile with native FP16 on T4/P100 GPUs, BF16 on Ampere or newer GPUs, fused AdamW, fused/memory-efficient SDPA, gradient accumulation, `torch.compile`, checkpointing every 500 steps, and a clean stop after 680 minutes.

Kaggle writes two useful artifacts to `/kaggle/working`:

- `model_weights.pth`: compact metadata-rich model weights for inference;
- `latest_checkpoint.pth`: model, optimizer, scaler, and step state for a resumable Kaggle session.

To continue training, upload `latest_checkpoint.pth` as a private dataset, attach it, and set `RESUME_CHECKPOINT` in the notebook. Architecture settings are checked before resume so an incompatible profile fails immediately.

## Train on a Mac

Start with a short check:

```bash
python train.py --profile tiny --max-iters 20
```

For the recommended model:

```bash
python train.py --profile mac
```

To keep more headroom for video calls and screen sharing, reduce the physical batch and CPU threads. Gradient accumulation preserves a useful effective batch:

```bash
python train.py --profile mac \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --cpu-threads 2
```

Resume an interrupted run:

```bash
python train.py --profile mac --resume build/latest_checkpoint.pth
```

Do not switch profiles when resuming: architecture settings must match the checkpoint. Full training was intentionally not run during optimization; only unit tests, short inference benchmarks, a real-checkpoint query, and a one-step tiny MPS training smoke test were used.

### Fine-tune the published weights for arithmetic

To improve the existing 136M model rather than start a new architecture, compile the verified corpus while retaining its published 32K tokenizer:

```bash
python data_pipeline/generate_math_problems.py --num-samples 100000 --seed 1337
python data_pipeline/tokenizer.py --reuse-tokenizer --seed 1337
python data_pipeline/pre_tokenize.py --tokenizer build/token.model
```

Then fine-tune with short sequences and a conservative learning rate. This changes the weights themselves; no answer-time calculator is involved:

```bash
python train.py \
  --profile legacy \
  --init-from build/model_weights.pth \
  --sequence-length 128 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --max-iters 1000 \
  --output build/model_weights_arithmetic.pth
```

Use fewer iterations for an initial quality check. Fine-tuning the 136M checkpoint is materially heavier than the smoke tests and was not started automatically while the Mac was needed for a meeting.

## Tests

```bash
python -m pytest -q
```

Tests cover legacy and modern cached-attention equivalence, weighted loss, stop tokens, and correctness of the new synthetic generators.

## Project layout

- `model_architecture/`: legacy-compatible and modern Transformer components
- `data_pipeline/`: downloads, verified synthetic generation, tokenizer training, and binary encoding
- `train.py`: local/CUDA training and resumable checkpoints
- `chat.py`: accelerated local CLI
- `benchmark.py`: short cached-versus-uncached inference benchmark
- `config.py`: `legacy`, `mac`, `kaggle`, and `tiny` profiles
- `prepare_kaggle_dataset.py`: validates and packages local binaries plus training source
- `kaggle-notebook.ipynb`: offline, checksum-verified single/dual-GPU Kaggle launcher

The published model is intentionally small and still has neural reasoning limitations. Improving its arithmetic requires fine-tuning the legacy checkpoint or training a new `mac`/`kaggle` checkpoint on the improved corpus.
