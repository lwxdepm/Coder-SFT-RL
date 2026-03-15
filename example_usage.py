#!/usr/bin/env python3
"""
Example usage of Code-RL components.
"""

import json
from pathlib import Path
from rich.console import Console

from reward.executor import CodeExecutor, BatchExecutor
from reward.metrics import RolloutMetrics
from utils.temperature import TemperatureScheduler, TemperatureSchedulerConfig

console = Console()


def example_executor():
    """Example of using the code executor."""
    console.print("[bold cyan]Example 1: Code Execution[/bold cyan]")

    executor = CodeExecutor(sandbox="firejail", timeout=3.0)

    # Example code and tests
    code = """
def add(a, b):
    return a + b
"""

    tests = [
        "assert add(1, 2) == 3",
        "assert add(-1, 1) == 0",
        "assert add(0, 0) == 0",
    ]

    result = executor.run(code, tests)

    console.print(f"Passed: {result.passed}/{result.total}")
    console.print(f"Reward: {executor.compute_reward(result):.3f}")
    console.print(f"Runnable: {result.is_runnable}")
    console.print(f"Runtime: {result.runtime:.3f}s")


def example_temperature_scheduler():
    """Example of using temperature scheduler."""
    console.print("\n[bold cyan]Example 2: Temperature Scheduling[/bold cyan]")

    config = TemperatureSchedulerConfig(
        schedule_type="cosine_annealing",
        t_max=1.0,
        t_min=0.4,
        warmup_steps=5,
        total_steps=20,
    )

    scheduler = TemperatureScheduler(config)

    console.print("Temperature schedule:")
    for step in range(1, 11):
        temp = scheduler.step()
        console.print(f"  Step {step:2d}: {temp:.3f}")


def example_metrics():
    """Example of using metrics."""
    console.print("\n[bold cyan]Example 3: Rollout Metrics[/bold cyan]")

    # Simulated rollout results
    metrics = RolloutMetrics(
        rewards=[0.0, 0.5, 1.0, 0.8, 0.2],
        runnable_flags=[True, True, True, True, False],
        responses=[
            "def add(a, b): return a + b",
            "def add(x, y): return x + y",
            "def sum_two(a, b): return a + b",
            "def add(a, b): return a + b",
            "def add(a, b) return a + b",  # syntax error
        ]
    )

    console.print(f"Reward mean: {metrics.reward_mean:.3f}")
    console.print(f"Reward std: {metrics.reward_std:.3f}")
    console.print(f"Pass rate: {metrics.pass_rate:.1%}")
    console.print(f"Runnable rate: {metrics.runnable_rate:.1%}")
    console.print(f"Is degenerate: {metrics.is_degenerate}")

    diversity = metrics.compute_diversity()
    console.print(f"Distinct-2: {diversity['distinct_2']:.3f}")
    console.print(f"Avg edit distance: {diversity['avg_edit_distance']:.3f}")


def main():
    console.print("[bold]Code-RL Example Usage[/bold]")
    console.print("=" * 50)

    example_executor()
    example_temperature_scheduler()
    example_metrics()

    console.print("\n[green]✅ All examples completed successfully![/green]")


if __name__ == "__main__":
    main()