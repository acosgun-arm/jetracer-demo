"""Regression tests for process shutdown signal handling."""

from __future__ import annotations

from pathlib import Path
from select import select
from signal import SIGINT, SIGTERM, getsignal
import subprocess
import sys
from threading import Thread
from time import sleep
from time import perf_counter

import jetracer_sim as sim


def test_signal_monitor_requests_shutdown_and_restores_handlers() -> None:
    previous_interrupt = getsignal(SIGINT)
    previous_terminate = getsignal(SIGTERM)
    monitor = sim.ShutdownSignalMonitor()
    with monitor:
        terminate_handler = getsignal(SIGTERM)
        assert callable(terminate_handler)
        terminate_handler(SIGTERM, None)
        assert monitor.requested
        assert monitor.reason == "received SIGTERM"
        monitor.request("later request does not replace first reason")
        assert monitor.reason == "received SIGTERM"
    assert getsignal(SIGINT) == previous_interrupt
    assert getsignal(SIGTERM) == previous_terminate


def test_manual_shutdown_request() -> None:
    monitor = sim.ShutdownSignalMonitor()
    monitor.request("test shutdown")
    assert monitor.requested
    assert monitor.reason == "test shutdown"


def test_shutdown_wait_is_interruptible() -> None:
    monitor = sim.ShutdownSignalMonitor()
    requester = Thread(
        target=lambda: (sleep(0.02), monitor.request("thread request")),
        daemon=True,
    )
    started = perf_counter()
    requester.start()
    assert monitor.wait(1.0)
    requester.join()
    assert perf_counter() - started < 0.5
    try:
        monitor.wait(-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative shutdown wait was accepted")


def test_realtime_demo_handles_sigterm_headlessly() -> None:
    repository = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(repository / "examples" / "realtime_demo.py"),
            "--platform-config",
            str(repository / "configs" / "platforms" / "sim.json"),
            "--model-config",
            str(repository / "configs" / "demo_models.json"),
            "--benchmark-registry",
            str(repository / "benchmarks" / "demo_model_benchmarks.json"),
            "--headless",
            "--duration",
            "5",
            "--no-log",
        ],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    output_lines: list[str] = []
    deadline_s = perf_counter() + 5.0
    ready = False
    while perf_counter() < deadline_s and process.poll() is None:
        readable, _, _ = select([process.stdout], [], [], 0.1)
        if not readable:
            continue
        line = process.stdout.readline()
        output_lines.append(line)
        if line.startswith("platform="):
            ready = True
            break
    if not ready:
        process.kill()
        remainder, _ = process.communicate()
        raise AssertionError(
            "real-time demo did not become ready:\n"
            + "".join(output_lines)
            + remainder
        )
    process.send_signal(SIGTERM)
    remainder, _ = process.communicate(timeout=5.0)
    output = "".join(output_lines) + remainder
    assert process.returncode == 0, output
    assert "shutdown_reason=received SIGTERM" in output
    assert "actuator_watchdog_expirations=0" in output


def main() -> None:
    test_signal_monitor_requests_shutdown_and_restores_handlers()
    test_manual_shutdown_request()
    test_shutdown_wait_is_interruptible()
    test_realtime_demo_handles_sigterm_headlessly()


if __name__ == "__main__":
    main()
