import json
import pandas as pd
from transformers import AutoTokenizer

input_file = "/home/msx2020/code-r1/data/verified/codea1_verify.jsonl"

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
)

data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)


def compute_prompt_tokens(prompt):
    if not isinstance(prompt, str):
        return 0

    try:
        prompt_json = json.loads(prompt)

        formatted = tokenizer.apply_chat_template(
            prompt_json,
            tokenize=False,
            add_generation_prompt=True
        )

        tokens = tokenizer(formatted).input_ids
        return len(tokens)

    except Exception:
        return 0


df["prompt_tokens"] = df["prompt"].apply(compute_prompt_tokens)

print(df["prompt_tokens"].describe())