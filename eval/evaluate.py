"""
Evaluation script.
Benchmarks: HumanEval+, MBPP+
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from reward.executor import CodeExecutor

console = Console()


def evaluate_on_dataset(
    model,
    tokenizer,
    dataset: list[dict],
    executor: CodeExecutor,
    temperature: float = 0.2,
    n_samples: int = 1,
    max_new_tokens: int = 512,
) -> dict:
    """
    Evaluate model on a dataset.
    Returns: pass@1, pass@k metrics.
    """
    import torch

    results = []
    batch_executor = CodeExecutor(sandbox_type="subprocess", timeout=5.0)

    for sample in tqdm(dataset, desc="Evaluating"):
        prompt = sample["prompt"]
        tests = sample["tests"]

        # Generate n_samples responses
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        responses = [
            tokenizer.decode(out[input_len:], skip_special_tokens=True)
            for out in outputs
        ]

        # Execute
        exec_results = batch_executor.run_batch(
            responses=responses,
            tests_list=[tests] * n_samples,
        )

        passed_flags = [r.passed == r.total for r in exec_results]

        results.append({
            "id": sample.get("id", ""),
            "passed_flags": passed_flags,
            "pass_rate": np.mean(passed_flags),
        })

    # Compute pass@k
    pass_at_1 = np.mean([r["passed_flags"][0] for r in results])
    pass_at_k = np.mean([any(r["passed_flags"]) for r in results])

    return {
        "pass@1": float(pass_at_1),
        f"pass@{n_samples}": float(pass_at_k),
        "n_samples": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--benchmark", default="mbpp",
                       choices=["mbpp", "humaneval", "all"])
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    console.print(f"[cyan]Loading model: {args.model_path}[/cyan]")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    executor = CodeExecutor(sandbox_type="subprocess", timeout=5.0)

    all_results = {}

    # Load benchmarks
    benchmarks = {}

    if args.benchmark in ["mbpp", "all"]:
        from datasets import load_dataset
        ds = load_dataset("evalplus/mbppplus", split="test")
        benchmarks["mbpp+"] = [
            {
                "id": str(item.get("task_id", i)),
                "prompt": item["text"],
                "tests": item.get("test_list", [])[:5],
            }
            for i, item in enumerate(ds)
            if item.get("test_list")
        ]

    if args.benchmark in ["humaneval", "all"]:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        benchmarks["humaneval+"] = [
            {
                "id": item.get("task_id", str(i)),
                "prompt": item["prompt"],
                "tests": [
                    line.strip()
                    for line in item.get("test", "").split("\n")
                    if line.strip().startswith("assert ")
                ][:5],
            }
            for i, item in enumerate(ds)
        ]

    # Evaluate
    for bench_name, dataset in benchmarks.items():
        console.print(f"\n[bold]Evaluating on {bench_name}...[/bold]")

        result = evaluate_on_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            executor=executor,
            temperature=args.temperature,
            n_samples=args.n_samples,
        )

        all_results[bench_name] = result

    # Print results table
    table = Table(title="Evaluation Results")
    table.add_column("Benchmark", style="cyan")
    table.add_column("pass@1", style="green")
    table.add_column(f"pass@{args.n_samples}", style="yellow")

    for bench, res in all_results.items():
        table.add_row(
            bench,
            f"{res['pass@1']:.1%}",
            f"{res.get(f'pass@{args.n_samples}', 0):.1%}",
        )

    console.print(table)

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[green]Results saved to {args.output}[/green]")


if __name__ == "__main__":
    main()