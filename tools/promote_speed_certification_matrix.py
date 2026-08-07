#!/usr/bin/env python3
"""Promote a completed speed-certification matrix for UI and CI use."""

from __future__ import annotations

import argparse
from pathlib import Path

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "benchmarks"
        / "speed_certification_results.json",
    )
    arguments = parser.parse_args()
    catalog = sim.promote_speed_certification_matrix(
        arguments.summary, arguments.output
    )
    print(
        f"catalog={arguments.output.resolve()} cases={len(catalog['cases'])} "
        f"certified={catalog['counts']['certified']} "
        f"uncertified={catalog['counts']['uncertified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
