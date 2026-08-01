"""Process-level shutdown requests for safe vehicle applications."""

from __future__ import annotations

from signal import SIGINT, SIGTERM, Signals, getsignal, signal
from threading import Event, current_thread, main_thread
from types import FrameType
from typing import Any


class ShutdownSignalMonitor:
    """Convert SIGINT/SIGTERM into a polled shutdown request.

    The signal handler only records intent. The control loop remains responsible
    for stopping its platform so neutral output happens in normal Python code.
    """

    def __init__(self) -> None:
        self._event = Event()
        self._reason: str | None = None
        self._previous_handlers: dict[Signals, Any] = {}
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("shutdown signal monitors cannot be restarted")
        if current_thread() is not main_thread():
            raise RuntimeError("shutdown signals must be installed on the main thread")
        for handled_signal in (SIGINT, SIGTERM):
            self._previous_handlers[handled_signal] = getsignal(handled_signal)
            signal(handled_signal, self._handle_signal)
        self._started = True

    def request(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("shutdown reason must not be empty")
        if self._reason is None:
            self._reason = reason
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def wait(self, timeout_s: float) -> bool:
        """Wait for shutdown, returning early when SIGINT/SIGTERM arrives."""

        if timeout_s < 0.0:
            raise ValueError("shutdown wait timeout must not be negative")
        return self._event.wait(timeout_s)

    def close(self) -> None:
        if not self._started or self._closed:
            return
        for handled_signal, previous in self._previous_handlers.items():
            signal(handled_signal, previous)
        self._closed = True

    def __enter__(self) -> ShutdownSignalMonitor:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _handle_signal(
        self,
        signum: int,
        _frame: FrameType | None,
    ) -> None:
        self.request(f"received {Signals(signum).name}")
