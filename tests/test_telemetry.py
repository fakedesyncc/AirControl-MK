"""Тесты сводной статистики телеметрии и счётчика FPS (stdlib unittest).

Проверяют ЧИСТУЮ логику без железа: ни камеры, ни MediaPipe, ни Tk.
Синтетический CSV с известными строками прогоняется через summarize_telemetry,
ожидаемые mean/p95/count/duration сверяются вручную. Отдельно проверяется, что
FPSMeter даёт вменяемый FPS по таймстемпам, а пустой/только-заголовок CSV не
ломает расчёт.

Запуск:  python -m unittest tests.test_telemetry -v
"""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.evaluation.metrics import (
    FPSMeter,
    TelemetryLogger,
    summarize_telemetry,
)


def _write_csv(path, rows):
    """Пишет синтетическую телеметрию с каноническими колонками логгера."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TelemetryLogger.FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(timestamp, fps, frame, detect):
    return {
        "timestamp": timestamp,
        "fps": fps,
        "frame_latency_ms": frame,
        "detect_latency_ms": detect,
        "pose": "open",
        "mode": "cursor",
        "filter": "oneeuro",
        "recognizer": "heuristic",
    }


class SummarizeTelemetryTest(unittest.TestCase):
    """Сводка по CSV: mean/median/p95/count/duration сверены вручную."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "telemetry_test.csv")

    def tearDown(self):
        self._tmp.cleanup()

    def test_known_rows_match_expected(self):
        # FPS: 10,20,30,40,50 -> mean 30, median 30, p95 (lin interp) = 48.0
        # frame: 5,10,15,20,25 -> mean 15, p95 = 24.0
        # detect: 1,2,3,4,5 -> mean 3, p95 = 4.8
        # timestamps 100..104 -> duration 4.0
        rows = [
            _row(100.0, 10, 5, 1),
            _row(101.0, 20, 10, 2),
            _row(102.0, 30, 15, 3),
            _row(103.0, 40, 20, 4),
            _row(104.0, 50, 25, 5),
        ]
        _write_csv(self.path, rows)

        stats, text = summarize_telemetry(self.path)

        self.assertEqual(stats["samples"], 5.0)
        self.assertAlmostEqual(stats["duration_s"], 4.0, places=3)
        self.assertAlmostEqual(stats["fps_mean"], 30.0, places=2)
        self.assertAlmostEqual(stats["fps_median"], 30.0, places=2)
        self.assertAlmostEqual(stats["fps_p95"], 48.0, places=2)
        self.assertAlmostEqual(stats["frame_latency_ms_mean"], 15.0, places=2)
        self.assertAlmostEqual(stats["frame_latency_ms_p95"], 24.0, places=2)
        self.assertAlmostEqual(stats["detect_latency_ms_mean"], 3.0, places=2)
        self.assertAlmostEqual(stats["detect_latency_ms_p95"], 4.8, places=2)
        self.assertIn("5 замеров", text)

    def test_accepts_list_of_rows_directly(self):
        rows = [_row(0.0, 25, 8, 2), _row(1.0, 35, 12, 4)]
        stats, _ = summarize_telemetry(rows)
        self.assertEqual(stats["samples"], 2.0)
        self.assertAlmostEqual(stats["fps_mean"], 30.0, places=2)
        self.assertAlmostEqual(stats["duration_s"], 1.0, places=3)

    def test_corrupt_cells_are_skipped(self):
        rows = [_row(0.0, 30, 10, 3), _row(1.0, "", "n/a", 5)]
        stats, _ = summarize_telemetry(rows)
        # вторая строка по fps/frame испорчена -> в расчёт идёт только первая
        self.assertAlmostEqual(stats["fps_mean"], 30.0, places=2)
        self.assertAlmostEqual(stats["frame_latency_ms_mean"], 10.0, places=2)
        # detect валиден в обеих строках: 3 и 5 -> среднее 4
        self.assertAlmostEqual(stats["detect_latency_ms_mean"], 4.0, places=2)


class SummarizeEmptyTest(unittest.TestCase):
    """Граничные случаи: пустой файл, только заголовок, отсутствующий путь."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_headers_only_csv_is_zeroed(self):
        path = os.path.join(self._tmp.name, "headers_only.csv")
        _write_csv(path, [])
        stats, text = summarize_telemetry(path)
        self.assertEqual(stats["samples"], 0.0)
        self.assertEqual(stats["duration_s"], 0.0)
        self.assertEqual(stats["fps_mean"], 0.0)
        self.assertEqual(stats["fps_p95"], 0.0)
        self.assertIn("0 замеров", text)

    def test_completely_empty_file(self):
        path = os.path.join(self._tmp.name, "empty.csv")
        open(path, "w", encoding="utf-8").close()
        stats, _ = summarize_telemetry(path)
        self.assertEqual(stats["samples"], 0.0)

    def test_missing_path_returns_zeroed_stats(self):
        path = os.path.join(self._tmp.name, "does_not_exist.csv")
        stats, _ = summarize_telemetry(path)
        self.assertEqual(stats["samples"], 0.0)
        self.assertEqual(stats["duration_s"], 0.0)

    def test_empty_list_input(self):
        stats, _ = summarize_telemetry([])
        self.assertEqual(stats["samples"], 0.0)


class FPSMeterTest(unittest.TestCase):
    """FPSMeter: онлайн-FPS по таймстемпам последних кадров."""

    def test_returns_zero_before_two_ticks(self):
        meter = FPSMeter()
        self.assertEqual(meter.fps, 0.0)
        meter.tick(0.0)
        self.assertEqual(meter.fps, 0.0)

    def test_sane_fps_from_uniform_timestamps(self):
        # кадры через 1/30 с -> ~30 FPS
        meter = FPSMeter(window=30)
        for i in range(10):
            meter.tick(i / 30.0)
        self.assertAlmostEqual(meter.fps, 30.0, places=4)

    def test_window_drops_old_timestamps(self):
        meter = FPSMeter(window=3)
        # окно 3 -> учитываются только последние три таймстемпа
        for i in range(6):
            meter.tick(i / 60.0)  # шаг 1/60 c -> 60 FPS
        self.assertAlmostEqual(meter.fps, 60.0, places=4)

    def test_zero_span_does_not_divide_by_zero(self):
        meter = FPSMeter()
        meter.tick(5.0)
        meter.tick(5.0)
        self.assertEqual(meter.fps, 0.0)


if __name__ == "__main__":
    unittest.main()
