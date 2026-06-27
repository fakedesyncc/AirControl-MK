"""Телеметрия производительности: FPS, задержка кадра, состояние системы.

Пишет метрики в CSV для последующего анализа (например, сравнение нагрузки при
разных фильтрах или распознавателях). Скользящее среднее FPS считается онлайн.
"""

import csv
import os
import time
from collections import deque
from typing import Optional


class FPSMeter:
    """Скользящий счётчик FPS по таймстемпам последних кадров."""

    def __init__(self, window: int = 30):
        self._ts = deque(maxlen=window)

    def tick(self, timestamp: Optional[float] = None) -> None:
        self._ts.append(timestamp if timestamp is not None else time.time())

    @property
    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


class TelemetryLogger:
    """Периодическая запись метрик в CSV."""

    FIELDS = ["timestamp", "fps", "frame_latency_ms", "detect_latency_ms",
              "pose", "mode", "filter", "recognizer"]

    def __init__(self, cfg, filter_type: str, recognizer: str):
        self.cfg = cfg
        self.filter_type = filter_type
        self.recognizer = recognizer
        self._last_sample = 0.0
        self._writer = None
        self._file = None
        if cfg.enabled and cfg.log_to_csv:
            os.makedirs(cfg.log_dir, exist_ok=True)
            path = os.path.join(cfg.log_dir,
                                f"telemetry_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            self._file = open(path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
            self._writer.writeheader()
            self.path = path

    def maybe_log(self, fps: float, frame_latency_ms: float,
                  detect_latency_ms: float, pose: str, mode: str) -> None:
        if not (self.cfg.enabled and self._writer):
            return
        now = time.time()
        if now - self._last_sample < self.cfg.sample_interval:
            return
        self._last_sample = now
        self._writer.writerow({
            "timestamp": round(now, 3),
            "fps": round(fps, 2),
            "frame_latency_ms": round(frame_latency_ms, 2),
            "detect_latency_ms": round(detect_latency_ms, 2),
            "pose": pose,
            "mode": mode,
            "filter": self.filter_type,
            "recognizer": self.recognizer,
        })
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
