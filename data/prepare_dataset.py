"""
Prepare RL dataset from multiple sources.
Supports: KodCode, LiveCodeBench, MBPP+, HumanEval+, RLVR Code Data Python

Fix: drop invalid Unicode surrogate characters before saving (丢弃).
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional, Any

from datasets import load_dataset
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import ast
import re

console = Console()

# ============================================================
# Unicode Cleaning (DROP surrogates)
# ============================================================

# Match Unicode surrogate range: U+D800..U+DFFF
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

def drop_surrogates_in_str(s: str) -> str:
    """Drop any Unicode surrogate code points from a string."""
    if not s:
        return s
    # Remove surrogate code points that break UTF-8 encoding
    return _SURROGATE_RE.sub("", s)

def sanitize_obj(obj: Any) -> Any:
    """
    Recursively sanitize nested structures by dropping surrogate characters in all strings.
    """
    if isinstance(obj, str):
        return drop_surrogates_in_str(obj)
    if isinstance(obj, list):
        return [sanitize_obj(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_obj(x) for x in obj)
    if isinstance(obj, dict):
        # keys can also be strings
        return {sanitize_obj(k): sanitize_obj(v) for k, v in obj.items()}
    return obj


# ============================================================
# Test Parsing
# ============================================================

def parse_tests(tests) -> list[str]:
    """Parse test cases from various formats into a list of assert statements."""
    if isinstance(tests, list):
        return [t.strip() for t in tests if isinstance(t, str) and t.strip().startswith("assert")]

    if isinstance(tests, str):
        tests = tests.strip()

        # Try to parse as JSON/Python list literal first
        if tests.startswith("[") and tests.endswith("]"):
            try:
                parsed = ast.literal_eval(tests)
                if isinstance(parsed, list):
                    return [t.strip() for t in parsed if isinstance(t, str) and t.strip().startswith("assert")]
            except (ValueError, SyntaxError):
                pass

        # Fallback: split by newlines only
        return [t.strip() for t in tests.split("\n") if t.strip().startswith("assert")]

    return []


# ============================================================
# Dataset Loaders
# ============================================================

def load_rlvr_code_data_python(max_samples: Optional[int] = None) -> list[dict]:
    """Load RLVR Code Data Python - high quality with verified tests."""
    console.print("[cyan]Loading RLVR Code Data Python...[/cyan]")

    ds = load_dataset("saurabh5/rlvr-code-data-python", split="train")
    samples: list[dict] = []

    for item in tqdm(ds, desc="RLVR Code Data Python"):
        if max_samples and len(samples) >= max_samples:
            break

        if not item.get("translated_test_cases"):
            continue

        # Note: dataset fields may vary; keep robust fallbacks.
        prompt = item.get("translated_problem") or item.get("rewritten_input") or item.get("prompt") or ""
        solution = item.get("translated_solution") or item.get("rewritten_solution") or item.get("solution") or ""

        # basic length filters
        if len(prompt) > 1000:
            continue
        if solution and len(solution) > 4000:
            continue

        difficulty = item.get("difficulty", 0)
        if isinstance(difficulty, (int, float)) and difficulty > 3:
            continue

        tests = parse_tests(item.get("translated_test_cases", []))
        if len(tests) < 2:
            continue

        sample = {
            "id": f"rlvr_python_{len(samples)}",
            "source": "rlvr_code_data_python",
            "prompt": prompt,
            "raw_question": prompt,
            "tests": tests[:10],  # cap at 10 tests
            "difficulty": difficulty,
            "solution": solution,
        }

        samples.append(sample)

    console.print(f"[green]RLVR Code Data Python: {len(samples)} samples[/green]")
    return samples


# ============================================================
# Dataset Verification (optional)
# ============================================================

def verify_sample(sample: dict, executor) -> bool:
    """Verify that the reference solution passes all tests."""
    if not sample.get("solution"):
        return True  # no solution to verify, keep it

    code = sample["solution"]
    tests = sample["tests"]

    result = executor.run(code, tests, timeout=10)
    return result["passed"] == result["total"]


def verify_dataset(samples: list[dict], executor, n_workers: int = 8) -> tuple[list[dict], dict]:
    """
    Verify all samples and filter out bad ones.
    Returns: (verified_samples, stats)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    verified: list[dict] = []
    stats = {"total": len(samples), "passed": 0, "failed": 0, "no_solution": 0}

    console.print(f"[cyan]Verifying {len(samples)} samples...[/cyan]")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(verify_sample, s, executor): s for s in samples}

        for future in tqdm(as_completed(futures), total=len(futures)):
            sample = futures[future]
            try:
                ok = future.result()
                if ok:
                    verified.append(sample)
                    if sample.get("solution"):
                        stats["passed"] += 1
                    else:
                        stats["no_solution"] += 1
                else:
                    stats["failed"] += 1
            except Exception:
                stats["failed"] += 1

    return verified, stats


# ============================================================
# Saving
# ============================================================

def save_jsonl(samples: list[dict], path: Path) -> None:
    """
    Save JSONL with UTF-8 encoding.
    Drops Unicode surrogate characters recursively (丢弃).
    """
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            s2 = sanitize_obj(s)
            f.write(json.dumps(s2, ensure_ascii=False) + "\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["rlvr"])
    parser.add_argument("--max_per_source", type=int, default=2000)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "rlvr": load_rlvr_code_data_python,
        # "kodcode": load_kodcode,
        # "mbpp": load_mbpp_plus,
        # "humaneval": load_humaneval_plus,
    }

    all_samples: list[dict] = []
    for source in args.sources:
        if source in loaders:
            samples = loaders[source](args.max_per_source)
            all_samples.extend(samples)
        else:
            console.print(f"[yellow]Unknown source: {source} (skipped)[/yellow]")

    console.print(f"\n[bold]Total samples: {len(all_samples)}[/bold]")

    # Optional verification
    if args.verify:
        from reward.executor import CodeExecutor
        executor = CodeExecutor(sandbox="subprocess")
        all_samples, stats = verify_dataset(all_samples, executor)

        table = Table(title="Verification Stats")
        table.add_column("Metric")
        table.add_column("Value")
        for k, v in stats.items():
            table.add_row(k, str(v))
        console.print(table)

    # Shuffle and split
    random.shuffle(all_samples)
    n_train = int(len(all_samples) * args.train_ratio)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]

    # Save
    save_jsonl(train_samples, output_dir / "train.jsonl")
    save_jsonl(val_samples, output_dir / "val.jsonl")

    # Print summary
    table = Table(title="Dataset Summary")
    table.add_column("Split")
    table.add_column("Size")
    table.add_row("Train", str(len(train_samples)))
    table.add_row("Val", str(len(val_samples)))
    console.print(table)

    # Source distribution
    from collections import Counter
    source_dist = Counter(s["source"] for s in all_samples)
    table2 = Table(title="Source Distribution")
    table2.add_column("Source")
    table2.add_column("Count")
    for src, cnt in source_dist.most_common():
        table2.add_row(src, str(cnt))
    console.print(table2)


if __name__ == "__main__":
    main()