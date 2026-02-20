"""
Generates synthetic, Chain-of-Thought (CoT) basic math problems (addition, subtraction,
multiplication, division, and basic word problems). The output is written to the 'corpus' directory.
"""

import os
import random

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus_path = os.path.join(PROJECT_ROOT, 'corpus')
os.makedirs(corpus_path, exist_ok=True)
output_file = os.path.join(corpus_path, 'synthetic_basic_math_cot.txt')
SEPARATOR = "<|FILE_SEP|>"

# --- Helper Lists for Diversity ---
NAMES = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah", "Ian", "Julia"]
ITEMS = ["apples", "bananas", "oranges", "books", "marbles", "coins", "toys", "cards", "stickers", "gems", "candies", "tickets"]
BUY_VERBS = ["buys", "finds", "receives", "collects", "is given"]
SELL_VERBS = ["sells", "loses", "gives away", "drops", "spends"]


def generate_addition():
    """Generates an addition problem with a step-by-step column addition reasoning string."""
    a = random.randint(1, 9999)
    b = random.randint(1, 9999)
    total = a + b

    # Dynamic CoT generation for column addition
    max_len = max(len(str(a)), len(str(b)))
    str_a, str_b = str(a).zfill(max_len), str(b).zfill(max_len)
    places = ["ones", "tens", "hundreds", "thousands", "ten thousands"]

    steps = [
        f"To solve {a} + {b}, we align the numbers by place value and add from right to left.",
        f"Aligning {str_a} and {str_b}:"
    ]

    carry = 0
    for i in range(max_len - 1, -1, -1):
        place_idx = max_len - 1 - i
        place_name = places[place_idx]
        d1 = int(str_a[i])
        d2 = int(str_b[i])
        col_sum = d1 + d2 + carry
        new_digit = col_sum % 10
        new_carry = col_sum // 10

        step_str = f"Step {place_idx + 1} ({place_name}): {d1} + {d2}"
        if carry > 0:
            step_str += f" + {carry} (carry)"
        step_str += f" = {col_sum}."

        if new_carry > 0:
            step_str += f" We write down {new_digit} and carry over {new_carry} to the {places[place_idx + 1]}."
        else:
            step_str += f" We write down {new_digit}."

        steps.append(step_str)
        carry = new_carry

    if carry > 0:
        steps.append(f"Step {max_len + 1}: Bring down the final carried {carry}.")

    steps.append(f"Reading the resulting digits gives us {total}.")

    q = f"What is {a} + {b}?"
    t = "\n".join(steps)
    return q, t, total


def generate_subtraction():
    """Generates a subtraction problem with a step-by-step column borrowing reasoning string."""
    a = random.randint(10, 9999)
    b = random.randint(1, a)
    result = a - b

    # Dynamic CoT generation for column subtraction with borrowing
    max_len = max(len(str(a)), len(str(b)))
    str_a_list = list(map(int, str(a).zfill(max_len)))
    str_b_list = list(map(int, str(b).zfill(max_len)))
    places = ["ones", "tens", "hundreds", "thousands"]

    steps = [
        f"To calculate {a} - {b}, we align the numbers and subtract column by column, right to left.",
        f"Aligning {str(a).zfill(max_len)} over {str(b).zfill(max_len)}:"
    ]

    for i in range(max_len - 1, -1, -1):
        place_idx = max_len - 1 - i
        place_name = places[place_idx]
        top = str_a_list[i]
        bot = str_b_list[i]

        if top < bot:
            # Borrowing logic
            j = i - 1
            while j >= 0 and str_a_list[j] == 0:
                j -= 1

            steps.append(f"- In the {place_name} column, {top} is less than {bot}, so we must borrow.")
            str_a_list[j] -= 1

            # Explain the cascade if we had to borrow across zeros
            for k in range(j, i):
                if k == j:
                    steps.append(
                        f"  Borrow 1 from the {places[max_len - 1 - k]} column, reducing it to {str_a_list[k]}.")
                else:
                    str_a_list[k] = 9
                    steps.append(
                        f"  Pass the borrowed value through the {places[max_len - 1 - k]} column (it becomes 9).")

            top += 10
            str_a_list[i] = top

        diff = top - bot
        steps.append(f"Step {place_idx + 1} ({place_name}): {top} - {bot} = {diff}.")

    steps.append(f"Combining the columns, the final answer is {result}.")

    q = f"Calculate {a} - {b}."
    t = "\n".join(steps)
    return q, t, result


def generate_multiplication():
    """Generates a multiplication problem using the partial products reasoning method."""
    if random.random() < 0.5:
        a = random.randint(2, 9)
        b = random.randint(10, 999)
    else:
        a = random.randint(10, 99)
        b = random.randint(10, 99)

    result = a * b

    # Dynamic CoT using partial products (Grid method logic, highly effective for LLMs)
    def break_down(n):
        s = str(n)
        return [int(d + '0' * (len(s) - idx - 1)) for idx, d in enumerate(s) if d != '0']

    parts_a = break_down(a)
    parts_b = break_down(b)

    joined_a = " + ".join(map(str, parts_a))
    joined_b = " + ".join(map(str, parts_b))

    steps = [f"To solve {a} * {b}, we can use the partial products method by breaking down each number.",
             f"Break down {a} into {joined_a}.",
             f"Break down {b} into {joined_b}.",
             "Multiply each part of the first number by each part of the second number:"]

    sub_results = []
    for pa in parts_a:
        for pb in parts_b:
            prod = pa * pb
            sub_results.append(prod)
            steps.append(f"- {pa} * {pb} = {prod}")

    steps.append("Finally, add all the partial products together:")
    equation = " + ".join(map(str, sub_results))
    steps.append(f"{equation} = {sum(sub_results)}.")

    q = f"Solve {a} * {b}."
    t = "\n".join(steps)
    return q, t, result


def generate_division():
    """Generates a division problem using traditional left-to-right long division logic."""
    b = random.randint(2, 100)
    result = random.randint(2, 100)
    a = b * result

    # Dynamic CoT using left-to-right long division logic
    steps = [f"To solve {a} / {b}, we use long division from left to right."]

    str_a = str(a)
    current_val = 0
    quotient_str = ""

    for digit in str_a:
        current_val = current_val * 10 + int(digit)
        if current_val < b:
            if quotient_str != "":  # Prevent leading zeros
                quotient_str += "0"
            steps.append(f"- Bring down '{digit}' to make {current_val}. {b} goes into {current_val} 0 times.")
        else:
            q_digit = current_val // b
            rem = current_val % b
            quotient_str += str(q_digit)
            steps.append(
                f"- Bring down '{digit}' to make {current_val}. {b} goes into {current_val} exactly {q_digit} times ({b} * {q_digit} = {b * q_digit}).")
            if rem == 0:
                steps.append(f"  Subtract {b * q_digit} from {current_val} to leave 0.")
            else:
                steps.append(f"  Subtract {b * q_digit} from {current_val} to leave a remainder of {rem}.")
            current_val = rem

    steps.append(f"Putting the quotient digits together gives us {int(quotient_str)}.")

    q = f"What is {a} / {b}?"
    t = "\n".join(steps)
    return q, t, result


def generate_word_problem():
    """Generates a basic math word problem involving addition and subtraction of items."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    start = random.randint(10, 100)

    lose_verb = random.choice(SELL_VERBS)
    lose_amt = random.randint(1, start - 1)

    gain_verb = random.choice(BUY_VERBS)
    gain_amt = random.randint(5, 50)

    step1 = start - lose_amt
    final = step1 + gain_amt

    q = f"{name} starts with {start} {item}. {name} {lose_verb} {lose_amt} of them, and later {gain_verb} {gain_amt} more. How many {item} does {name} have now?"

    t = (
        f"Let's break down the events to find the final number of {item}.\n"
        f"1. Initial State: {name} starts with {start} {item}.\n"
        f"2. First Event: {name} {lose_verb} {lose_amt} {item}. This is a subtraction.\n"
        f"   {start} - {lose_amt} = {step1} {item} remaining.\n"
        f"3. Second Event: Later, {name} {gain_verb} {gain_amt} {item}. This is an addition.\n"
        f"   We add this to our current total: {step1} + {gain_amt} = {final}.\n"
        f"4. Conclusion: {name} now has a final count of {final} {item}."
    )
    return q, t, final


# --- Data Builder ---

def build_dataset(num_samples=1000000):
    """Main loop building the full synthetic dataset with the requested number of samples."""
    print(f"Generating {num_samples:,} diverse synthetic math problems...")

    generators = [
        generate_addition,
        generate_subtraction,
        generate_multiplication,
        generate_division,
        generate_word_problem
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Randomly pick a problem type
            func = random.choice(generators)
            q, t, a = func()

            # Format using the new <|thinking|> and <|answer|> tags
            entry = (
                f"<|user|> {q} <|end|>\n"
                f"<|assistant|>\n"
                f"<|thinking|>\n{t}\n"
                f"<|answer|> {a} <|end|>"
            )

            f.write(entry + SEPARATOR)

            if (i + 1) % 50000 == 0:
                print(f"Generated {i + 1:,} problems...")

    print(f"\nSuccess! Dataset saved to {output_file}")


if __name__ == "__main__":
    # Generates 1 million samples. You can adjust this number.
    build_dataset(1000000)