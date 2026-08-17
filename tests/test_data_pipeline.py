from fractions import Fraction
import re

from data_pipeline.generate_math_problems import (
    generate_fraction_addition,
    generate_linear_equation,
    generate_percentage,
)
from data_pipeline.pre_tokenize import token_weights
from train import BinaryDataset

import numpy as np


def test_new_math_generators_return_verified_answers():
    for _ in range(100):
        question, _, answer = generate_fraction_addition()
        a, b, c, d = map(int, re.findall(r"\d+", question)[:4])
        assert Fraction(answer) == Fraction(a, b) + Fraction(c, d)

        question, _, answer = generate_percentage()
        percent, base = map(int, re.findall(r"\d+", question)[:2])
        assert int(answer) == base * percent // 100

        question, _, answer = generate_linear_equation()
        match = re.fullmatch(r"Solve for x: (\d+)x ([+-]) (\d+) = (-?\d+)\.", question)
        coefficient, sign, magnitude, rhs = match.groups()
        coefficient, magnitude, rhs = map(int, (coefficient, magnitude, rhs))
        offset = magnitude if sign == "+" else -magnitude
        assert coefficient * int(answer) + offset == rhs


def test_assistant_and_answer_weights():
    assistant, answer, end = 10, 11, 12
    ids = [1, 2, end, assistant, 4, answer, 6, 7, end]
    assert token_weights(ids, assistant, answer, end) == [0, 0, 0, 0, 1, 1, 3, 3, 1]


def test_binary_dataset_keeps_tokens_and_weights_aligned(tmp_path):
    tokens = np.arange(40, dtype=np.uint16)
    weights = (np.arange(40) % 4).astype(np.uint8)
    token_path = tmp_path / "train.bin"
    weight_path = tmp_path / "train_weights.bin"
    tokens.tofile(token_path)
    weights.tofile(weight_path)

    dataset = BinaryDataset(token_path, block_size=8, weights_path=weight_path)
    x, y, loss_weights = dataset.batch(
        batch_size=3,
        device="cpu",
        rng=np.random.default_rng(7),
    )

    assert x.shape == y.shape == loss_weights.shape == (3, 8)
    assert (y == x + 1).all()
    assert (loss_weights == (y % 4)).all()
