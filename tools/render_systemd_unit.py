#!/usr/bin/env python3
"""Render, but never install or start, the standby-only systemd service."""

from __future__ import annotations

import argparse
from pathlib import Path

import jetracer_sim as sim


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=sim.DEFAULT_DEPLOYMENT_CONFIGURATION_PATH,
    )
    parser.add_argument("--service-user", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    configuration = sim.load_deployment_configuration(arguments.config)
    unit = sim.render_deployment_systemd_unit(
        configuration, arguments.service_user
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(unit, encoding="utf-8")
    print(arguments.output)


if __name__ == "__main__":
    main()
