"""
Dataset class for verl GRPO training.
Handles prompt formatting and test case packaging.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Optional
from transformers import PreTrainedTokenizer


class CodeRLDataset(Dataset):
    """
    Dataset for Code-RL training.

    Each sample contains:
        - input_ids: tokenized prompt
        - attention_mask
        - tests: list of assert statements (non-tensor)
        - metadata: source, difficulty, etc.

    verl requires non-tensor fields to be stored separately.
    """

    SYSTEM_PROMPT = (
        "You are an expert Python programmer. "
        "Solve the given programming problem. "
        "Return ONLY the complete Python code. "
        "Do NOT include explanation or markdown unless asked."
    )

    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 512,
        max_response_length: int = 512,
        chat_template: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.chat_template = chat_template

        self.samples = self._load(data_file)
        print(f"[CodeRLDataset] Loaded {len(self.samples)} samples from {data_file}")

    def _load(self, path: str) -> list[dict]:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _format_prompt(self, raw_question: str) -> str:
        """Format prompt using chat template if available."""
        if self.chat_template and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": raw_question.strip()},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"### Problem\n{raw_question.strip()}\n\n"
                f"### Solution\n"
            )
        return prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Format prompt
        question = sample.get("raw_question", sample.get("prompt", ""))
        prompt_str = self._format_prompt(question)

        # Tokenize
        encoding = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            # Tensor fields (for verl batch)
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),

            # Non-tensor fields (for reward manager)
            "tests": sample.get("tests", []),
            "prompt_str": prompt_str,
            "id": sample.get("id", str(idx)),
            "source": sample.get("source", "unknown"),
            "difficulty": sample.get("difficulty", "unknown"),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Custom collate for verl DataProto.
        Separates tensor fields from non-tensor fields.
        """
        tensor_keys = ["input_ids", "attention_mask"]
        non_tensor_keys = ["tests", "prompt_str", "id", "source", "difficulty"]

        tensor_batch = {
            k: torch.stack([b[k] for b in batch])
            for k in tensor_keys
        }

        non_tensor_batch = {
            k: [b[k] for b in batch]
            for k in non_tensor_keys
        }

        return {
            "tensor_batch": tensor_batch,
            "non_tensor_batch": non_tensor_batch,
        }