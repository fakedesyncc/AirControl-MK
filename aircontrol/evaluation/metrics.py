"""Телеметрия производительности: FPS, задержка кадра, состояние системы.

Пишет метрики в CSV для последующего анализа (например, сравнение нагрузки при
разных фильтрах или распознавателях). Скользящее среднее FPS считается онлайн.

Помимо записи здесь есть чистая функция :func:`summarize_telemetry`, которая
считает сводную статистику (FPS, задержки кадра и детекции) по уже записанному
CSV. Она не требует pandas — только стандартные ``csv`` и ``statistics``.

Сводку из командной строки можно получить так::

    python -c "from aircontrol.evaluation.metrics import summarize_telemetry; \\
print(summarize_telemetry('data/logs/telemetry_20260629_120000.csv')[1])"

Первый элемент кортежа — словарь со статистикой, второй — готовый текст.
"""

import csv
import os
import statistics
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Union


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


def _percentile(values: List[float], pct: float) -> float:
    """Перцентиль методом линейной интерполяции (тип «inclusive», как np.percentile).

    ``pct`` задаётся в долях единицы (0.95 для p95). Для пустого списка
    возвращает 0.0, для одного элемента — сам элемент.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = pct * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def _iter_rows(source: Union[str, List[dict]]) -> List[dict]:
    """Приводит источник (путь к CSV или список строк-словарей) к списку строк."""
    if isinstance(source, str):
        if not os.path.exists(source):
            return []
        with open(source, "r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    return list(source)


def summarize_telemetry(source: Union[str, List[dict]]) -> Tuple[Dict[str, float], str]:
    """Считает сводную статистику по телеметрии и возвращает (словарь, текст).

    ``source`` — путь к CSV-файлу телеметрии (с колонками
    :attr:`TelemetryLogger.FIELDS`) либо уже разобранный список строк-словарей.

    В словаре: ``samples`` (число строк), ``duration_s`` (диапазон timestamp),
    ``fps_mean`` / ``fps_median`` / ``fps_p95``, ``frame_latency_ms_mean`` /
    ``frame_latency_ms_p95``, ``detect_latency_ms_mean`` /
    ``detect_latency_ms_p95``. Пустой или содержащий только заголовок CSV даёт
    нулевую статистику и не падает.

    Нечисловые/пустые ячейки в числовых колонках просто пропускаются, поэтому
    частично испорченные строки не ломают расчёт.
    """
    rows = _iter_rows(source)

    def column(name: str) -> List[float]:
        out = []
        for row in rows:
            raw = row.get(name, "")
            if raw is None or raw == "":
                continue
            try:
                out.append(float(raw))
            except (TypeError, ValueError):
                continue
        return out

    fps = column("fps")
    frame = column("frame_latency_ms")
    detect = column("detect_latency_ms")
    times = column("timestamp")

    duration = (max(times) - min(times)) if len(times) >= 2 else 0.0

    stats: Dict[str, float] = {
        "samples": float(len(rows)),
        "duration_s": round(duration, 3),
        "fps_mean": round(statistics.fmean(fps), 2) if fps else 0.0,
        "fps_median": round(statistics.median(fps), 2) if fps else 0.0,
        "fps_p95": round(_percentile(fps, 0.95), 2),
        "frame_latency_ms_mean": round(statistics.fmean(frame), 2) if frame else 0.0,
        "frame_latency_ms_p95": round(_percentile(frame, 0.95), 2),
        "detect_latency_ms_mean": round(statistics.fmean(detect), 2) if detect else 0.0,
        "detect_latency_ms_p95": round(_percentile(detect, 0.95), 2),
    }

    text = (
        f"Телеметрия: {int(stats['samples'])} замеров за {stats['duration_s']} с\n"
        f"  FPS:   среднее {stats['fps_mean']}, медиана {stats['fps_median']}, "
        f"p95 {stats['fps_p95']}\n"
        f"  Кадр:  среднее {stats['frame_latency_ms_mean']} мс, "
        f"p95 {stats['frame_latency_ms_p95']} мс\n"
        f"  Детект: среднее {stats['detect_latency_ms_mean']} мс, "
        f"p95 {stats['detect_latency_ms_p95']} мс"
    )
    return stats, text
