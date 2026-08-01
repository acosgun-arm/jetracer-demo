#!/usr/bin/env python3
"""Headless target-aware model deployment preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=REPOSITORY_ROOT / "configs/off_the_shelf_models.json",
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/off_the_shelf_model_benchmarks.json",
    )
    parser.add_argument(
        "--policy", type=Path, default=sim.DEFAULT_DEPLOYMENT_POLICY_PATH
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    policy = sim.load_deployment_policy(arguments.policy)
    capabilities = sim.collect_runtime_capabilities(policy)
    result = sim.evaluate_deployment(
        arguments.models,
        arguments.benchmarks,
        policy,
        capabilities,
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **result.to_dict(),
        "gui_opened": False,
        "camera_opened": False,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
