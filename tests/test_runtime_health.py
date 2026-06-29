"""Тесты устойчивой деградации захвата кадра (stdlib unittest, без железа).

Проверяют ЧИСТУЮ логику принятия решений: decide_camera_health() и тонкую
обёртку-накопитель CameraHealthMonitor. Ни камеры, ни Tk, ни MediaPipe —
импортируются только pure-функции из aircontrol.app.

Запуск:  python -m unittest tests.test_runtime_health -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.app import (
    CAMERA_LOST_STATUS,
    CameraHealthMonitor,
    build_runtime_health_lines,
    decide_camera_health,
)


class DecideCameraHealthTest(unittest.TestCase):
    """Чистая функция: счётчик неудач + время → (показать статус, бэкофф)."""

    def test_no_failures_no_status_no_backoff(self):
        show, backoff = decide_camera_health(
            consecutive_failures=0, elapsed_since_last_ok=None)
        self.assertFalse(show)
        self.assertEqual(backoff, 0.0)

    def test_few_failures_back_off_but_no_status(self):
        # Пара неудач — статус ещё не поднимаем, но уже просим бэкофф (не busy-loop).
        show, backoff = decide_camera_health(
            consecutive_failures=2, elapsed_since_last_ok=0.1)
        self.assertFalse(show)
        self.assertGreater(backoff, 0.0)

    def test_many_consecutive_failures_trigger_status(self):
        show, backoff = decide_camera_health(
            consecutive_failures=5, elapsed_since_last_ok=0.0, fail_threshold=5)
        self.assertTrue(show)
        self.assertGreater(backoff, 0.0)

    def test_elapsed_time_triggers_status_even_with_few_failures(self):
        # Даже при малом счётчике долгая пауза без кадров = камера потеряна.
        show, _ = decide_camera_health(
            consecutive_failures=1, elapsed_since_last_ok=2.0, fail_seconds=1.0)
        self.assertTrue(show)

    def test_lost_backoff_is_larger_than_transient_backoff(self):
        _, transient = decide_camera_health(
            consecutive_failures=1, elapsed_since_last_ok=0.0)
        _, lost = decide_camera_health(
            consecutive_failures=10, elapsed_since_last_ok=5.0)
        self.assertGreater(lost, transient)


class CameraHealthMonitorTest(unittest.TestCase):
    """Накопитель состояния поверх чистой логики."""

    def test_fresh_monitor_is_healthy(self):
        m = CameraHealthMonitor()
        self.assertFalse(m.lost)
        self.assertEqual(m.consecutive_failures, 0)
        self.assertIsNone(m.status_line)

    def test_no_consecutive_failures_no_status(self):
        m = CameraHealthMonitor()
        m.record_success(now=100.0)
        self.assertFalse(m.lost)
        self.assertIsNone(m.status_line)

    def test_n_consecutive_failures_set_camera_lost(self):
        m = CameraHealthMonitor(fail_threshold=5, fail_seconds=999.0)
        m.record_success(now=0.0)
        backoff = 0.0
        for i in range(5):
            backoff = m.record_failure(now=0.0 + i * 0.001)
        self.assertTrue(m.lost)
        self.assertEqual(m.status_line, CAMERA_LOST_STATUS)
        self.assertGreater(backoff, 0.0)
        self.assertEqual(m.consecutive_failures, 5)

    def test_failures_request_backoff_before_status(self):
        # Даже первая неудача обязана вернуть бэкофф > 0, иначе цикл крутит вхолостую.
        m = CameraHealthMonitor()
        backoff = m.record_failure(now=1.0)
        self.assertGreater(backoff, 0.0)

    def test_elapsed_time_marks_lost(self):
        m = CameraHealthMonitor(fail_threshold=999, fail_seconds=1.0)
        m.record_success(now=0.0)
        m.record_failure(now=2.0)  # прошло 2с с последнего кадра > порога 1с
        self.assertTrue(m.lost)

    def test_recovery_resets_status_and_counter(self):
        m = CameraHealthMonitor(fail_threshold=3, fail_seconds=999.0)
        m.record_success(now=0.0)
        for i in range(3):
            m.record_failure(now=i * 0.001)
        self.assertTrue(m.lost)
        # Камера вернулась.
        m.record_success(now=10.0)
        self.assertFalse(m.lost)
        self.assertEqual(m.consecutive_failures, 0)
        self.assertIsNone(m.status_line)


class HealthLineSurfacingTest(unittest.TestCase):
    """Статус «камера потеряна» должен попадать в строки оверлея."""

    def _base_kwargs(self):
        return dict(
            mode="control",
            input_status="INPUT OK",
            fps=30.0,
            detect_ms=10.0,
            auto_tuned=False,
            last_frame_age=0.0,
            hand_detected=True,
            mode_age=10.0,
        )

    def test_camera_lost_line_present_when_flag_set(self):
        lines = build_runtime_health_lines(camera_lost=True, **self._base_kwargs())
        self.assertIn(CAMERA_LOST_STATUS, lines)
        # Самое важное сообщение — первым.
        self.assertEqual(lines[0], CAMERA_LOST_STATUS)

    def test_no_camera_lost_line_when_healthy(self):
        lines = build_runtime_health_lines(camera_lost=False, **self._base_kwargs())
        self.assertNotIn(CAMERA_LOST_STATUS, lines)


if __name__ == "__main__":
    unittest.main()
