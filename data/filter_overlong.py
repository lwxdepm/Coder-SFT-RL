import pandas as pd
from transformers import AutoTokenizer


def filter_long_samples(
    parquet_path: str,
    tokenizer_path: str,
    text_column: str = "prompt",
    max_length: int = 2048,
    output_path: str = None,
):
    """
    过滤 tokenizer 后 token 长度超过 max_length 的样本
    """

    print("Loading parquet...")
    df = pd.read_parquet(parquet_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print("Tokenizing and filtering...")


    def get_len(messages):

        # 如果是字符串，转为chat格式
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # 应用 chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return len(tokenizer.encode(text))

    df["token_len"] = df[text_column].apply(get_len)

    print("原始样本数:", len(df))

    df_filtered = df[df["token_len"] <= max_length]

    print("过滤后样本数:", len(df_filtered))

    df_filtered = df_filtered.drop(columns=["token_len"])

    if output_path is None:
        output_path = parquet_path

    print("Saving to:", output_path)
    df_filtered.to_parquet(output_path)

    return df_filtered
filter_long_samples(
    '/home/featurize/code-r1/data/final/codea1_verify.parquet' , 
    'Qwen/Qwen2.5-Coder-1.5B-Instruct' ,
    "prompt",
    500 , 
    '/home/featurize/code-r1/data/final/codea1_verify.parquet' 
)
