#!/usr/bin/env python3
"""Manage staged JetRacer bring-up records; this tool never drives the car."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=sim.DEFAULT_BRINGUP_PLAN_PATH)
    parser.add_argument("--platform-id", default="jetracer-pro-elp-dry-run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--state", type=Path, required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--state", type=Path, required=True)
    begin = subparsers.add_parser("begin")
    begin.add_argument("--state", type=Path, required=True)
    begin.add_argument("--stage", required=True)
    begin.add_argument("--preflight", type=Path)
    begin.add_argument(
        "--preflight-config", type=Path, default=sim.DEFAULT_PREFLIGHT_CONFIG_PATH
    )
    record = subparsers.add_parser("record")
    record.add_argument("--state", type=Path, required=True)
    record.add_argument("--stage", required=True)
    record.add_argument("--outcome", choices=("pass", "fail"), required=True)
    record.add_argument("--evidence", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser_for().parse_args(argv)
    plan = sim.load_bringup_plan(arguments.plan)
    if arguments.command == "init":
        state = sim.initialize_bringup_state(
            arguments.state, plan, platform_id=arguments.platform_id
        )
    elif arguments.command == "status":
        state = sim.load_bringup_state(
            arguments.state, plan, platform_id=arguments.platform_id
        )
    elif arguments.command == "begin":
        state = sim.begin_bringup_stage(
            arguments.state,
            plan,
            platform_id=arguments.platform_id,
            stage_id=arguments.stage,
            preflight_path=arguments.preflight,
            preflight_configuration=sim.load_preflight_configuration(
                arguments.preflight_config
            ),
        )
    else:
        state = sim.record_bringup_stage(
            arguments.state,
            plan,
            platform_id=arguments.platform_id,
            stage_id=arguments.stage,
            outcome=arguments.outcome,
            evidence_path=arguments.evidence,
        )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
