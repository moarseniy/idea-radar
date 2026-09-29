"""Small thread-safe retry and circuit-breaker primitives for source ingestion."""
from __future__ import annotations

import threading
import time


class CircuitOpenError(RuntimeError):
    """The connector is paused after repeated transient failures."""


class CircuitBreaker:
    """Consecutive-failure breaker with one half-open probe after its cooldown."""

    def __init__(self, threshold: int = 3, reset_after: float = 30.0):
        self.threshold = threshold
        self.reset_after = reset_after
        self.lock = threading.Lock()
        self.failures = 0
        self.open_until = 0.0
        self.probe_in_flight = False

    def before_call(self) -> None:
        with self.lock:
            now = time.monotonic()
            if self.open_until > now:
                raise CircuitOpenError("Коннектор временно приостановлен после повторных ошибок")
            if self.open_until:
                if self.probe_in_flight:
                    raise CircuitOpenError("Коннектор ожидает проверочный запрос")
                self.probe_in_flight = True

    def success(self) -> None:
        with self.lock:
            self.failures = 0
            self.open_until = 0.0
            self.probe_in_flight = False

    def failure(self) -> None:
        with self.lock:
            self.failures += 1
            self.probe_in_flight = False
            if self.failures >= self.threshold:
                self.open_until = time.monotonic() + self.reset_after
