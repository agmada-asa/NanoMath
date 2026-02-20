import os
import torch
import sentencepiece as spm
from model_architecture.gpt_language_model import GPTLanguageModel
from config import get_hyperparams

# --- 1. Hyperparameters & Setup ---
CONFIG = get_hyperparams()
block_size = CONFIG["block_size"]
n_embd = CONFIG["n_embd"]
n_head = CONFIG["n_head"]
n_layer = CONFIG["n_layer"]

if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

MAX_TOKENS = 200

print(f"Loading model on {device}...")

# --- 2. Load Tokenizer & Apply Vocab Padding ---
tokenizer_path = os.path.join("build", "token.model")
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)
vocab_size = sp.get_piece_size()

# We must pad the vocab exactly like we did in training
if vocab_size % 128 != 0:
    vocab_size = ((vocab_size // 128) + 1) * 128

# --- 3. Initialize Model ---
# We force dropout to 0.0 for inference
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    device=device,
    dropout=0.0
)

# --- 4. Load the Trained Weights ---
# Depending on how long you trained for, you might need to change this to:
# "latest_checkpoint.pth" or "model_wights.pth", since Kaggle limits to 12 hours of training time
weights_path = os.path.join("build", "model_weights.pth") 

try:
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Clean up prefixes if they exist (from torch.compile or accelerate)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        elif k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    print("Successfully loaded model weights!")
except FileNotFoundError:
    print(f"ERROR: Could not find '{weights_path}'. Make sure your training finished and saved the file.")
    exit()

model.to(device)
model.eval()

print(f"Model Size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

# --- 5. The Chat Loop ---
print("\n" + "=" * 30)
print("🤖 NanoMath is Ready")
print("Type 'quit' to exit.")
print("=" * 30 + "\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    # Format the prompt exactly how the model saw it in training
    prompt = f"<|user|> {user_input} <|end|>\n<|assistant|>"

    # Encode and send to device
    context = torch.tensor([sp.encode_as_ids(prompt)], dtype=torch.long, device=device)

    print("NanoMath: ", end="", flush=True)

    # Generate the full response at once (much faster!)
    with torch.no_grad():
        # Adjust temperature: 0.7 is good, lower (0.5) is more strictly mathematical
        generated_ids = model.generate(context, max_new_tokens=MAX_TOKENS, temperature=0.7)

    # Extract only the newly generated tokens (ignore the prompt we fed it)
    new_tokens = generated_ids[0][len(context[0]):].tolist()

    # Decode to text
    response_text = sp.decode(new_tokens)

    # Clean up the output (stop at the end token if it generated one)
    if "<|end|>" in response_text:
        response_text = response_text.split("<|end|>")[0].strip()

    print(response_text)
    print("\n" + "-" * 20)