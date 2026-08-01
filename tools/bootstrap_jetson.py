#!/usr/bin/env python3
"""Dry-run-by-default bootstrap for the provisional Jetson baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

import check_jetson_compatibility as compatibility


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def bootstrap_commands(
    baseline: Mapping[str, Any], venv_path: Path
) -> list[list[str]]:
    bootstrap = baseline["bootstrap"]
    python_command = str(bootstrap["python_command"])
    pip = str(venv_path / "bin" / "python")
    commands = [
        ["sudo", "apt-get", "update"],
        [
            "sudo",
            "apt-get",
            "install",
            "--yes",
            *[str(package) for package in bootstrap["apt_packages"]],
        ],
        [python_command, "-m", "venv", str(venv_path)],
        [pip, "-m", "pip", "install", "--editable", str(REPOSITORY_ROOT)],
    ]
    for name, value in bootstrap.get("cmake_defines", {}).items():
        commands[-1].extend(
            ["--config-settings", f"cmake.define.{name}={value}"]
        )
    return commands


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a Jetson bootstrap plan; make changes only with --apply"
    )
    parser.add_argument(
        "--baseline", type=Path, default=compatibility.DEFAULT_BASELINE_PATH
    )
    parser.add_argument("--venv", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="install apt packages, create the venv, and install the project",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser_for().parse_args(argv)
    try:
        baseline = compatibility.load_baseline(arguments.baseline.resolve())
    except compatibility.BaselineError as error:
        print(f"baseline error: {error}", file=sys.stderr)
        return 2

    configured_venv = Path(str(baseline["bootstrap"]["default_venv"]))
    venv_path = (arguments.venv or configured_venv).resolve()
    commands = bootstrap_commands(baseline, venv_path)
    print("Jetson bootstrap plan:")
    for command in commands:
        print(f"  {shlex.join(command)}")

    if not arguments.apply:
        print("Dry run only. Re-run with --apply on the Jetson to execute this plan.")
        return 0

    inventory = compatibility.collect_inventory(baseline)
    report = compatibility.evaluate_inventory(
        baseline, inventory, strict_target=True
    )
    if not report["target_match"]:
        print(
            "refusing --apply: this host does not match the Jetson target",
            file=sys.stderr,
        )
        return 1

    environment = os.environ.copy()
    for command in commands:
        subprocess.run(
            command,
            check=True,
            cwd=REPOSITORY_ROOT,
            env=environment,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
