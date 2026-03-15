"""
Benchmark utilities for Code-RL.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class BenchmarkResult:
    name: str
    pass_at_1: float
    pass_at_k: float
    n_samples: int
    temperature: float
    model_name: str
    timestamp: str


class BenchmarkSuite:
    """Manages multiple benchmark evaluations."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def save(self, name: str = "benchmark_results.json"):
        """Save all results to JSON."""
        output_path = self.output_dir / name
        with open(output_path, "w") as f:
            json.dump([r.__dict__ for r in self.results], f, indent=2)
        console.print(f"[green]Results saved to {output_path}[/green]")

    def print_summary(self):
        """Print summary table of all results."""
        table = Table(title="Benchmark Summary")
        table.add_column("Model", style="cyan")
        table.add_column("Benchmark", style="magenta")
        table.add_column("pass@1", style="green")
        table.add_column(f"pass@k", style="yellow")
        table.add_column("Temp", style="blue")

        for result in self.results:
            table.add_row(
                Path(result.model_name).name,
                result.name,
                f"{result.pass_at_1:.1%}",
                f"{result.pass_at_k:.1%}",
                f"{result.temperature:.2f}",
            )

        console.print(table)

    def compare_models(self, model_names: list[str]):
        """Compare multiple models across benchmarks."""
        from collections import defaultdict

        model_results = defaultdict(list)

        for result in self.results:
            if result.model_name in model_names:
                model_results[result.model_name].append(result)

        # Create comparison table
        table = Table(title="Model Comparison")
        table.add_column("Benchmark", style="cyan")

        for model_name in model_names:
            table.add_column(model_name, style="green")

        # Group by benchmark name
        benchmark_names = sorted(set(r.name for r in self.results))

        for bench in benchmark_names:
            row = [bench]
            for model_name in model_names:
                # Find result for this model and benchmark
                matching = [
                    r for r in model_results[model_name]
                    if r.name == bench
                ]
                if matching:
                    row.append(f"{matching[0].pass_at_1:.1%}")
                else:
                    row.append("N/A")
            table.add_row(*row)

        console.print(table)