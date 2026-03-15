"""
Dataset verification tool.
Run before training to eliminate noisy samples.
"""

import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import sys
# 添加项目根目录到 path 
project_root = Path(__file__).parent.parent 
sys.path.insert(0, str(project_root))
from reward.executor import CodeExecutor

console = Console()


def verify_sample(
    sample: dict,
    executor: CodeExecutor,
    n_runs: int = 3,
) -> dict:
    """
    Verify one sample by running its reference solution.
    """

    solution = sample.get("solution", "")
    tests = sample.get("tests", [])
    sid = sample.get("id")

    if not solution:
        return {
            "id": sid,
            "status": "no_solution",
            "pass_rate": None,
            "stable": None,
        }

    if not tests:
        return {
            "id": sid,
            "status": "no_tests",
            "pass_rate": None,
            "stable": None,
        }

    pass_rates = []

    for _ in range(n_runs):

        passed = 0

        for t in tests:
            result = executor.execute(solution, t)

            if result["status"] == "success":
                passed += 1

        pass_rate = passed / len(tests)
        pass_rates.append(pass_rate)

    mean_pass = sum(pass_rates) / len(pass_rates)
    stable = max(pass_rates) - min(pass_rates) < 0.1

    if mean_pass >= 1.0 and stable:
        status = "pass"
    elif mean_pass >= 0.5:
        status = "partial"
    elif not stable:
        status = "flaky"
    else:
        status = "fail"

    return {
        "id": sid,
        "status": status,
        "pass_rate": mean_pass,
        "stable": stable,
        "pass_rates": pass_rates,
    }


def analyze_failures(
    failures: list[dict],
    samples_by_id: dict,
    executor: CodeExecutor,
    n_examples: int = 5,
):
    console.print(f"\n[bold red]Failure Analysis ({n_examples} examples)[/bold red]")

    for i, fail in enumerate(failures[:n_examples]):

        sample = samples_by_id.get(fail["id"], {})
        solution = sample.get("solution", "")
        tests = sample.get("tests", [])

        console.print(f"\n[red]--- Failed Sample {i+1} ---[/red]")
        console.print(f"ID: {fail['id']}")
        console.print(f"Status: {fail['status']}")
        console.print(f"Pass rate: {fail.get('pass_rate', 'N/A')}")

        if solution and tests:

            for t in tests[:3]:
                result = executor.execute(solution, t)
                console.print(f"[yellow]Test result:[/yellow] {result}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Code-RL dataset quality"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--n_runs", type=int, default=4)
    parser.add_argument("--n_workers", type=int, default=64)
    parser.add_argument("--sandbox", default="subprocess",
                       choices=["subprocess", "firejail"])
    parser.add_argument("--filter_status", nargs="+",
                       default=["pass"])
    parser.add_argument("--analyze_failures", action="store_true")
    args = parser.parse_args()

    # load dataset
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    console.print(f"[cyan]Loaded {len(samples)} samples[/cyan]")

    executor = CodeExecutor(
        sandbox_type=args.sandbox,
        timeout=8.0,
    )

    verification_results = []
    samples_by_id = {
        s.get("id", str(i)): s
        for i, s in enumerate(samples)
    }

    # parallel verify
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:

        futures = {
            pool.submit(
                verify_sample,
                s,
                executor,
                args.n_runs
            ): s
            for s in samples
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Verifying",
        ):
            try:
                result = future.result()
                verification_results.append(result)

            except Exception as e:
                sample = futures[future]
                verification_results.append({
                    "id": sample.get("id"),
                    "status": "error",
                    "pass_rate": 0.0,
                    "stable": False,
                    "error": str(e),
                })

    # stats
    status_counts = defaultdict(int)
    for r in verification_results:
        status_counts[r["status"]] += 1

    table = Table(title="Verification Results")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percentage", style="yellow")

    total = len(verification_results)

    for status, count in sorted(status_counts.items()):
        pct = 100 * count / total
        color = "green" if status == "pass" else "red"
        table.add_row(
            f"[{color}]{status}[/{color}]",
            str(count),
            f"{pct:.1f}%"
        )

    table.add_row("[bold]Total[/bold]", str(total), "100%")
    console.print(table)

    # failure analysis
    if args.analyze_failures:
        failures = [
            r for r in verification_results
            if r["status"] not in args.filter_status
        ]
        analyze_failures(failures, samples_by_id, executor)

    # save filtered dataset
    if args.output:

        passed_ids = {
            r["id"]
            for r in verification_results
            if r["status"] in args.filter_status
        }

        filtered = [
            s for s in samples
            if s.get("id") in passed_ids
        ]

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for s in filtered:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        console.print(
            f"\n[green]Saved {len(filtered)} verified samples "
            f"to {output_path}[/green]"
        )

        report_path = output_path.with_suffix(".verify_report.json")

        with open(report_path, "w") as f:
            json.dump({
                "total": total,
                "status_counts": dict(status_counts),
                "kept": len(filtered),
                "filter_status": args.filter_status,
                "results": verification_results,
            }, f, indent=2)

        console.print(f"[cyan]Report saved to {report_path}[/cyan]")


if __name__ == "__main__":
    main()