# NanoMath: A 136M Parameter Math-Reasoning Language Model

NanoMath is a custom-built, GPT-style Large Language Model designed specifically to solve mathematical problems using Chain-of-Thought (CoT) reasoning. Built entirely from scratch in PyTorch, this project encompasses the entire lifecycle of an LLM: from synthetic data generation and custom tokenization to mixed-precision training and inference.


## 🚀 Project Overview

The goal of this project was to build an end-to-end language modeling pipeline to better understand transformer architectures, data processing bottlenecks, and training optimizations. 

- **Architecture:** Decoder-only Transformer (GPT-style) built from scratch.
- **Size:** 136 Million Parameters.
- **Dataset:** ~1 Billion tokens (Synthetic basic math + GSM8K + NuminaMath).
- **Training Compute:** NVIDIA T4 GPU (12 Hours on Kaggle).

## 🛠️ Data Pipeline & Training Flow

To optimize compute costs and training efficiency, the data pipeline was split between local processing and cloud training:

1. **Local Data Processing (Mac):**
   - **Data Collection & Generation:** Combined high-quality math datasets (GSM8K, NuminaMath) with a custom Python script generating hundreds of thousands of synthetic arithmetic problems. The synthetic data enforces a strict `<|thinking|>` and `<|answer|>` format to teach the model step-by-step reasoning.
   - **Custom Tokenizer:** Trained a custom Byte-Pair Encoding (BPE) tokenizer using `sentencepiece`. The vocabulary was explicitly tailored to recognize mathematical operators and digits efficiently.
   - **Pre-Tokenization:** To prevent data-loading bottlenecks during training, the entire dataset was pre-tokenized and serialized into highly efficient binary (`.bin`) files locally.

2. **Cloud Training (Kaggle):**
   - The compiled tokenizer (`.model`) and massive `.bin` files were uploaded as a Kaggle Dataset.
   - The model was trained using `numpy.memmap` to stream the binary data directly from disk to the GPU, entirely bypassing RAM limitations.
   - **Optimizations:** Implemented PyTorch 2.0+ `torch.compile`, Flash Attention (`F.scaled_dot_product_attention`), and Mixed Precision Training (FP16) via `torch.cuda.amp` to maximize the T4 GPU's throughput.

3. **Local Inference (Mac):**
   - After 12 hours of training, the model weights (`.pth`) were downloaded back to my local machine for inference and evaluation via a custom CLI chat interface (`chat.py`).

## 🧠 Model Performance & Limitations

The model was trained to "think before it speaks" by generating intermediate reasoning steps. It performs well on basic arithmetic but, given its small parameter size and limited training time, it exhibits known LLM hallucinations on complex operations.

### ✅ Successes: Basic Arithmetic & Step-by-Step Logic
The model successfully learned how to execute algorithms like long division and partial products:

**Example 1: Long Division**
> **You:** What is 48 / 6?  
> **LLM:** `<|thinking|>` To solve 48 / 6, we use long division from left to right. - Bring down '4' to make 4. 6 goes into 4 0 times. - Bring down '8' to make 48. 6 goes into 48 exactly 8 times (6 * 8 = 48). Subtract 48 from 48 to leave 0. Putting the quotient digits together gives us 8. `<|answer|>` 8

**Example 2: Partial Products**
> **You:** What is 3 * 12?  
> **LLM:** `<|thinking|>` To solve 3 * 12, we can use the partial products method by breaking down each number. Break down 3 into 3. Break down 12 into 10 + 2. Multiply each part of the first number by each part of the second number: - 3 * 10 = 30 - 3 * 2 = 6 Finally, add all the partial products together: 30 + 6 = 36. `<|answer|>` 36

### ⚠️ Limitations: Hallucinations & Complex Multi-Step Routing
Because of the model's small scale, it sometimes struggles to carry values perfectly across complex borrowing steps or loses track of logical routing in word problems.

**Example 3: Logic Routing Hallucination**
> **You:** If I have 4 apples and sell 3 and buy 2 more, how many do I have?  
> **LLM:** If I have 4 apples and sell 3 apples, then you would have: 4 (yuan) + 3 (second) = $\boxed{8}$ apples.  


**Example 4: Partial Math Breakdown**
> **You:** 435 + 47
> **LLM:**  <|thinking|> To solve 435 + 47, we align the numbers by place value and add from right to left. Aligning 435 and 0047: Step 1 (ones): 5 + 7 = 12. We write down 2 and carry over 1 to the tens. Step 2 (tens): 3 + 4 + 1 (carry) = 8. We write down 8. Step 3 (hundreds): 3 + 0 = 3. We write down 3. Step 4 (thousands): 4 + 0 = 4. We write down 4. Reading the resulting digits gives us 4382. <|answer|> 4382

## 💻 How to Run (Inference)

1. Download the model weights (`model_weights.pth`) and the tokenizer (`token.model`) from [HuggingFace](https://huggingface.co/agmadaasa/NanoMath/tree/main).
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the trained weights (`model_weights.pth`) and the tokenizer (`token.model`) in the build/ directory.
4. Run the interactive chat:
   ```bash
   python chat.py
   ```

## 🏗️ Repo Structure
`model_architecture`: Contains the PyTorch implementations of the Transformer Block, MultiHeadAttention, FeedForward, and the core GPTLanguageModel.

`data_pipeline/`: Directory containing all data processing scripts:
- `download_data.py`: Downloads GSM8K and NuminaMath datasets.
- `generate_math_problems.py`: Script to procedurally generate diverse, CoT-formatted synthetic math problems.
- `tokenizer.py` & `pre_tokenize.py`: Scripts for training the SentencePiece BPE tokenizer and serializing text data into raw binary sequences.
- `complete_pipeline.py`: Orchestrates the full data pipeline.

`config.py`: Centralized hyperparameters for easy tuning.

`kaggle-notebook.ipynb`: The notebook used for the 12-hour cloud training run. Note: Because Kaggle is running in a notebook environment and its structure is restricted, there is some code duplication between this notebook and the Python scripts in this repo.

`chat.py`: The CLI application for running inference locally.