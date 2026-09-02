# NanoMath

NanoMath is a compact decoder-only language model for maths reasoning, written from scratch in PyTorch. The project covers the full model lifecycle: data preparation, tokenizer training, single and multi-GPU training, checkpointing, and local inference.

The current Hugging Face release is NanoMath v1.1. It has 113.27 million parameters, a 16,384-piece SentencePiece tokenizer, a 1,024-token context window, and a metadata-rich checkpoint that records the exact model configuration. Despite its size, it produces readable multi-step solutions and handled division, multiplication, a word problem, and a linear equation correctly in the current spot check.

NanoMath is an educational model rather than a calculator. Its correct answers show how much a small purpose-trained model can learn, while its mistakes make the remaining limits easy to inspect. See [observed outputs](#observed-outputs) for examples from the published v1.1 checkpoint.

## Run the published model

```bash
git clone https://github.com/agmada-asa/NanoMath.git
cd NanoMath
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf download agmadaasa/NanoMath model_weights.pth token.model --local-dir build
python chat.py "What is 84 / 7?"
```

Start an interactive session with:

```bash
python chat.py
```

The CLI selects CUDA, Apple MPS, or CPU automatically. Greedy decoding is the default because sampling usually makes arithmetic less consistent.

Useful options:

```bash
python chat.py --show-speed "Explain why 3/4 is larger than 2/3"
python chat.py --temperature 0.5 --top-k 40 "Write a new maths problem"
python chat.py --no-kv-cache --show-speed "What is 48 / 6?"
```

## Published versions

| Version | Parameters | Architecture | Tokenizer | Status |
|---|---:|---|---:|---|
| v1.1 | 113.27M | 16 layers, RoPE, RMSNorm, SwiGLU, grouped-query attention | 16,384 | Current |
| v1.0 | 136.18M | 12 layers, learned positions, LayerNorm, ReLU | 32,768 | Retained in Hugging Face history |

To fetch a specific release, add `--revision v1.1` or `--revision v1.0` to the `hf download` command. A checkpoint and tokenizer must come from the same release.

The v1.1 checkpoint completed optimizer step 5,999 on two NVIDIA T4 GPUs. Its training build contains 641,276,817 train tokens and 13,076,780 validation tokens. The checkpoint SHA-256 is:

```text
f13f762b64ec710221575b6f5d4dae05f6105d8234b8a53a35ea7b40680f6dca
```

## Observed outputs

These results came from the published v1.1 checkpoint with greedy decoding and a 200-token limit. It solved four of the six mixed prompts correctly. This is a reproducible spot check, not an accuracy benchmark.

| Prompt | Model's final answer | Expected | Result |
|---|---|---:|---|
| `What is 84 / 7?` | `12` | 12 | Correct |
| `Calculate 57 + 68.` | `89` | 125 | Incorrect |
| `What is 17 * 24?` | `408` | 408 | Correct |
| `A box has 15 pencils. Six are removed and four are added. How many pencils are in the box?` | `13` | 13 | Correct |
| `Solve 3x + 5 = 20.` | `5` | 5 | Correct |
| `Which is larger, 5/8 or 2/3?` | Lost the original numerators and did not finish within 200 tokens | 2/3 | Incorrect |

For the incorrect addition, the model produced fluent but invalid subtraction steps:

```text
<|thinking|> To solve 57 + 68, we align the numbers by place value and add from right to left. Aligning 57 over 68: - In the ones column, 7 is less than 8, so we must borrow. ... Combining the columns, the final answer is 89. <|answer|> 89
```

For a 113M model, producing consistent solution structure across these different problem types is a useful result. The remaining limitation is arithmetic reliability: the reasoning format makes a mistake easy to inspect, but it cannot guarantee that the result is correct.

## Model architecture

The v1.1 `kaggle` profile uses:

- 16 decoder blocks with width 768
- 12 query heads and 4 key-value heads
- rotary position embeddings
- RMSNorm and SwiGLU
- tied token and output embeddings
- bias-free projections
- a 1,024-token context window

Inference uses a per-layer key-value cache, direct stop-token detection, and FP16 on MPS or CUDA. On older PyTorch builds and non-CUDA devices, grouped-query attention expands key and value tensors to a compatible view.

The loader also understands the raw v1.0 checkpoint. It infers the legacy dimensions from the saved tensors when checkpoint metadata is absent.

## Training data and loss

The corpus combines GSM8K, NuminaMath, and generated arithmetic problems. Each example uses role and reasoning markers such as `<|user|>`, `<|thinking|>`, `<|answer|>`, and `<|end|>`.

Data preparation splits complete problems between training and validation before tokenization. It writes packed `uint16` token streams and aligned `uint8` loss weights. Prompt tokens receive weight 0, reasoning tokens weight 1, and final-answer tokens weight 3.

The generated curriculum includes arithmetic, exact fractions, percentages, linear equations, and word problems. Generation and document shuffling use explicit seeds.

## Build the data

The quick pipeline downloads a smaller development corpus and generates 10,000 synthetic examples:

```bash
python data_pipeline/complete_pipeline.py --quick
```

Build the full corpus with:

```bash
python data_pipeline/complete_pipeline.py
```

Run individual stages when you need more control:

```bash
python data_pipeline/download_data.py --max-numina 50000
python data_pipeline/generate_math_problems.py --num-samples 100000 --seed 1337
python data_pipeline/tokenizer.py --vocab-size 16384 --seed 1337
python data_pipeline/pre_tokenize.py
```

The resulting files are `corpus/train.bin`, `corpus/val.bin`, `corpus/train_weights.bin`, and `corpus/val_weights.bin`.

## Train on Kaggle

Package the corpus, tokenizer, and matching source code:

```bash
python prepare_kaggle_dataset.py --dataset-id agmadaasa/nanomath-kaggle-v2
```

The command writes an upload-ready directory to `build/kaggle_dataset/`. Its manifest records file sizes, token counts, dtypes, profile settings, and SHA-256 checksums.

Create the Kaggle dataset or publish a new version:

```bash
python3 -m kaggle datasets create -p build/kaggle_dataset
python3 -m kaggle datasets version -p build/kaggle_dataset -m "refresh training corpus"
```

Import `kaggle-notebook.ipynb`, attach the dataset, select a GPU accelerator, and run the notebook. It verifies the packaged files before training and starts Distributed Data Parallel when Kaggle exposes more than one GPU.

Kaggle writes two checkpoints to `/kaggle/working`:

- `model_weights.pth` contains the model, model configuration, and training metadata needed for inference.
- `latest_checkpoint.pth` also contains optimizer and scaler state for resuming training.

To continue in another session, attach `latest_checkpoint.pth`, set `RESUME_CHECKPOINT` in the notebook, and raise `MAX_ITERS` above the saved step.

## Train locally

Run a short pipeline check first:

```bash
python train.py --profile tiny --max-iters 20
```

The `mac` profile is sized for Apple Silicon:

```bash
python train.py --profile mac
```

Reduce the physical batch and CPU use when the machine needs more headroom:

```bash
python train.py --profile mac \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --cpu-threads 2
```

Resume an interrupted run with:

```bash
python train.py --profile mac --resume build/latest_checkpoint.pth
```

Do not change architecture profiles when resuming. The trainer checks the saved model configuration and rejects incompatible settings.

## Fine-tune v1.0

The legacy profile is available for experiments with the original 136M model. Download both v1.0 files first:

```bash
hf download agmadaasa/NanoMath \
  model_weights.pth token.model \
  --revision v1.0 \
  --local-dir build
```

Reuse that tokenizer when compiling data, then initialize the legacy profile from the downloaded weights:

```bash
python data_pipeline/tokenizer.py --reuse-tokenizer --seed 1337
python data_pipeline/pre_tokenize.py --tokenizer build/token.model
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

## Web app

[NanoMathWeb](https://github.com/agmada-asa/NanoMathWeb) sends requests to the [NanoMath API Space](https://huggingface.co/spaces/agmadaasa/NanoMath-api). The frontend does not load this repository or the Hugging Face model files directly.

Updating the model repository alone does not trigger a Space rebuild. The API Space must use code compatible with the checkpoint and restart before the web app serves the new model. Once the Space is running the new release, NanoMathWeb needs no code change because its API URL stays the same.

## Tests

```bash
python -m pytest -q
```

The tests cover legacy and modern model construction, cached-attention equivalence, weighted loss, stop tokens, and generated-data checks.

Run a short cache benchmark with:

```bash
python benchmark.py --profile kaggle --vocab-size 16384 --new-tokens 12
```

## Repository layout

- `model_architecture/` contains the decoder blocks, attention, and feed-forward layers.
- `data_pipeline/` downloads source datasets, generates synthetic problems, trains the tokenizer, and packs token streams.
- `train.py` handles local training, CUDA training, DDP, and resumable checkpoints.
- `chat.py` loads v1.0 and v1.1 checkpoints for local inference.
- `benchmark.py` compares cached and full-context decoding.
- `config.py` defines the `legacy`, `mac`, `kaggle`, and `tiny` profiles.
- `prepare_kaggle_dataset.py` builds the checksum-verified Kaggle package.
- `kaggle-notebook.ipynb` runs the packaged trainer on one or more Kaggle GPUs.
