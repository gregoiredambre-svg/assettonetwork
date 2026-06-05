"""Single entry point for the simplified road-maintenance modelling pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


@dataclass(frozen=True)
class Step:
    name: str
    description: str
    command: list[str]


STEPS: list[Step] = [
    Step(
        name="materials",
        description="Build first-construction pavement material summaries.",
        command=[PYTHON, "load_materials.py"],
    ),
    Step(
        name="graph",
        description="Build the section-level interdependency graph.",
        command=[PYTHON, "graph_construction.py"],
    ),
    Step(
        name="temporal",
        description="Train the temporal full_refined GCN and local baselines.",
        command=[
            PYTHON,
            "graph_model_temporal.py",
            "--graph-variant",
            "full_refined",
            "--output-tag",
            "full_refined",
            "--treatment-mode",
            "experiment",
        ],
    ),
    Step(
        name="rgcn",
        description="Train relation-aware temporal R-GCN variants.",
        command=[PYTHON, "part1_extensions.py", "--stage", "rgcn"],
    ),
    Step(
        name="network",
        description="Train the full_refined network-impact surrogate.",
        command=[
            PYTHON,
            "graph_model.py",
            "--graph-variant",
            "full_refined",
            "--output-tag",
            "full_refined",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the simplified modelling pipeline.")
    parser.add_argument(
        "steps",
        nargs="*",
        choices=[step.name for step in STEPS],
        help="Specific steps to run. By default, all steps run in pipeline order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = set(args.steps) if args.steps else {step.name for step in STEPS}
    selected = [step for step in STEPS if step.name in requested]

    for step in selected:
        command_text = " ".join(step.command)
        print(f"\n[{step.name}] {step.description}")
        print(f"$ {command_text}")
        if args.dry_run:
            continue
        subprocess.run(step.command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
