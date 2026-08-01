#!/usr/bin/env python3
"""Run verified standby, explicitly armed driving, or graceful shutdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jetracer_sim as sim


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=sim.DEFAULT_DEPLOYMENT_CONFIGURATION_PATH,
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--standby", action="store_true")
    modes.add_argument("--drive", action="store_true")
    modes.add_argument("--safe-stop", action="store_true")
    parser.add_argument(
        "--watch", action="store_true", help="recheck standby continuously"
    )
    parser.add_argument(
        "--explicit-arm",
        action="store_true",
        help="required separate acknowledgement for drive mode",
    )
    arguments = parser.parse_args()
    if arguments.watch and not arguments.standby:
        parser.error("--watch is valid only with --standby")
    if arguments.explicit_arm and not arguments.drive:
        parser.error("--explicit-arm is valid only with --drive")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    configuration = sim.load_deployment_configuration(arguments.config)
    if arguments.standby:
        return sim.run_deployment_standby(configuration, watch=arguments.watch)
    if arguments.drive:
        return sim.run_deployed_drive(
            configuration, explicit_arm=arguments.explicit_arm
        )
    result = sim.safe_stop_deployed_runtime(configuration)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
