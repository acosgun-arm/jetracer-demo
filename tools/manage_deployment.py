#!/usr/bin/env python3
"""Create, prepare, verify, promote, and roll back JetRacer releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jetracer_sim as sim


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=sim.DEFAULT_DEPLOYMENT_CONFIGURATION_PATH,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create an immutable release")
    create.add_argument("--release-id", required=True)
    create.add_argument("--wheelhouse", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="install a release offline")
    prepare.add_argument("--release-id", required=True)
    verify = commands.add_parser("verify", help="verify release hashes")
    verify.add_argument("--release-id", required=True)
    verify.add_argument("--require-prepared", action="store_true")
    promote = commands.add_parser("promote", help="make a release current")
    promote.add_argument("--release-id", required=True)
    commands.add_parser("rollback", help="swap current and previous releases")
    commands.add_parser("status", help="show deployment link status")
    return parser.parse_args()


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    configuration = sim.load_deployment_configuration(arguments.config)
    if arguments.command == "create":
        return sim.create_release(
            configuration, arguments.release_id, arguments.wheelhouse
        )
    if arguments.command == "prepare":
        return sim.prepare_release(configuration, arguments.release_id)
    if arguments.command == "verify":
        return sim.verify_release(
            configuration,
            arguments.release_id,
            require_prepared=arguments.require_prepared,
        )
    if arguments.command == "promote":
        return sim.promote_release(configuration, arguments.release_id)
    if arguments.command == "rollback":
        return sim.rollback_release(configuration)
    return sim.deployment_status(configuration)


if __name__ == "__main__":
    print(json.dumps(run(parse_arguments()), indent=2, sort_keys=True))
