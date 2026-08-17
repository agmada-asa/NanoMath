import torch

from model_architecture.gpt_language_model import GPTLanguageModel


def make_model(**overrides):
    config = {
        "vocab_size": 128,
        "n_embd": 64,
        "block_size": 64,
        "n_head": 4,
        "n_layer": 2,
        "dropout": 0.0,
    }
    config.update(overrides)
    return GPTLanguageModel(**config).eval()


def test_cached_logits_match_full_forward_for_legacy_and_modern_models():
    torch.manual_seed(7)
    configurations = [
        {},
        {
            "n_kv_head": 2,
            "norm_type": "rmsnorm",
            "mlp_type": "swiglu",
            "position_encoding": "rope",
            "tie_embeddings": True,
            "bias": False,
        },
    ]
    tokens = torch.randint(0, 128, (2, 12))
    for configuration in configurations:
        model = make_model(**configuration)
        full_logits, _ = model(tokens)
        _, _, cache = model(tokens[:, :8], use_cache=True)
        cached_logits = []
        for index in range(8, 12):
            logits, _, cache = model(
                tokens[:, index : index + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cached_logits.append(logits)
        torch.testing.assert_close(
            torch.cat(cached_logits, dim=1), full_logits[:, 8:], atol=2e-5, rtol=1e-5
        )


def test_weighted_loss_ignores_prompt_and_emphasizes_answer():
    model = make_model()
    tokens = torch.randint(0, 128, (1, 8))
    targets = torch.randint(0, 128, (1, 8))
    weights = torch.tensor([[0, 0, 0, 1, 1, 3, 3, 1]], dtype=torch.float32)
    logits, loss = model(tokens, targets, loss_weights=weights)
    raw = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    )
    expected = (raw * weights.reshape(-1)).sum() / weights.sum()
    torch.testing.assert_close(loss, expected)


def test_generation_stops_on_requested_token():
    model = make_model()
    model._sample = lambda logits, temperature, top_k: torch.full(
        (logits.size(0), 1), 5, device=logits.device
    )
    prompt = torch.tensor([[1, 2, 3]])
    result = model.generate(prompt, max_new_tokens=20, stop_token_ids=5)
    assert result.shape[1] == prompt.shape[1] + 1
