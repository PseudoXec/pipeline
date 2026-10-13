"""Small rolling-average inference timers used by the pipeline heartbeat."""
from collections import deque
import threading
import time
from typing import Dict, Optional


class RollingAverageTimer:
    def __init__(self, window_size: int = 30):
        self._samples = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, elapsed_ms: float) -> None:
        with self._lock:
            self._samples.append(elapsed_ms)

    def measure(self):
        return _TimerContext(self)

    @property
    def average_ms(self) -> Optional[float]:
        with self._lock:
            return sum(self._samples) / len(self._samples) if self._samples else None


class _TimerContext:
    def __init__(self, timer: RollingAverageTimer):
        self.timer = timer

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.timer.record((time.perf_counter() - self.started) * 1000.0)


_timers: Dict[str, RollingAverageTimer] = {}
_timers_lock = threading.Lock()


def timer(name: str) -> RollingAverageTimer:
    with _timers_lock:
        return _timers.setdefault(name, RollingAverageTimer())


def averages() -> Dict[str, Optional[float]]:
    with _timers_lock:
        return {name: item.average_ms for name, item in _timers.items()}
