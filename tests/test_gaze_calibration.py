"""Тесты чистых помощников калибровки взгляда (stdlib unittest).

Покрывают только логику без Tk/камеры/MediaPipe: раскладку точек-целей, сборку
собранных (raw, target) пар в формат GazeCalibration.fit() и проверку
качества/достаточности данных. Подгонка проверяется через gaze_math.GazeCalibration
(чистый numpy-независимый код).

Запуск:  python -m unittest tests.test_gaze_calibration -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.tracking.gaze_math import GazeCalibration
from aircontrol.ui.calibration import (
    GAZE_MIN_TARGETS,
    average_raw,
    build_gaze_calibration_samples,
    clamp_unit,
    gaze_samples_quality_ok,
    gaze_target_points,
)


class TargetLayoutTests(unittest.TestCase):
    def test_default_layout_is_corners_plus_center(self):
        pts = gaze_target_points()
        self.assertEqual(len(pts), 5)
        m = 0.15
        expected = [
            (m, m), (1.0 - m, m), (1.0 - m, 1.0 - m), (m, 1.0 - m), (0.5, 0.5),
        ]
        for got, exp in zip(pts, expected):
            self.assertAlmostEqual(got[0], exp[0], places=6)
            self.assertAlmostEqual(got[1], exp[1], places=6)

    def test_all_points_inside_unit_square(self):
        for x, y in gaze_target_points():
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)

    def test_margin_is_clamped_to_avoid_degenerate_layout(self):
        # Слишком большой отступ ограничивается, центр не совпадает с углами.
        pts = gaze_target_points(margin=0.9)
        corners = pts[:4]
        center = pts[4]
        self.assertEqual(center, (0.5, 0.5))
        for cx, cy in corners:
            self.assertNotEqual((cx, cy), center)
            self.assertTrue(0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0)

    def test_negative_margin_clamped_to_zero(self):
        pts = gaze_target_points(margin=-1.0)
        # Углы прижаты к (0,0)/(1,1), но остаются внутри [0..1].
        self.assertEqual(pts[0], (0.0, 0.0))
        self.assertEqual(pts[2], (1.0, 1.0))


class ClampUnitTests(unittest.TestCase):
    def test_clamps_range(self):
        self.assertEqual(clamp_unit(-0.5), 0.0)
        self.assertEqual(clamp_unit(1.5), 1.0)
        self.assertEqual(clamp_unit(0.42), 0.42)


class AverageRawTests(unittest.TestCase):
    def test_averages_pairs(self):
        avg = average_raw([(0.0, 0.0), (1.0, 0.5), (0.5, 0.25)])
        self.assertIsNotNone(avg)
        self.assertAlmostEqual(avg[0], 0.5, places=6)
        self.assertAlmostEqual(avg[1], 0.25, places=6)

    def test_empty_returns_none(self):
        self.assertIsNone(average_raw([]))

    def test_skips_invalid_entries(self):
        avg = average_raw([None, (0.2, 0.4), (0.4, 0.6)])
        self.assertIsNotNone(avg)
        self.assertAlmostEqual(avg[0], 0.3, places=6)
        self.assertAlmostEqual(avg[1], 0.5, places=6)


class BuildSamplesTests(unittest.TestCase):
    def test_assembles_valid_pairs(self):
        collected = [
            ((0.20, 0.20), (0.15, 0.15)),
            ((0.80, 0.20), (0.85, 0.15)),
            ((0.80, 0.80), (0.85, 0.85)),
            ((0.20, 0.80), (0.15, 0.85)),
            ((0.50, 0.50), (0.50, 0.50)),
        ]
        samples = build_gaze_calibration_samples(collected)
        self.assertEqual(len(samples), 5)
        for (raw, tgt) in samples:
            self.assertEqual(len(raw), 2)
            self.assertEqual(len(tgt), 2)

    def test_drops_targets_without_raw(self):
        collected = [
            ((0.20, 0.20), (0.15, 0.15)),
            (None, (0.85, 0.15)),
            ((0.80, 0.80), (0.85, 0.85)),
        ]
        samples = build_gaze_calibration_samples(collected)
        self.assertEqual(len(samples), 2)

    def test_target_coords_are_clamped(self):
        collected = [((0.2, 0.2), (-0.5, 1.5))]
        samples = build_gaze_calibration_samples(collected)
        self.assertEqual(samples[0][1], (0.0, 1.0))


class QualityCheckTests(unittest.TestCase):
    def _good_collected(self):
        return [
            ((0.20, 0.20), (0.15, 0.15)),
            ((0.80, 0.20), (0.85, 0.15)),
            ((0.80, 0.80), (0.85, 0.85)),
            ((0.20, 0.80), (0.15, 0.85)),
            ((0.50, 0.50), (0.50, 0.50)),
        ]

    def test_good_samples_pass(self):
        samples = build_gaze_calibration_samples(self._good_collected())
        self.assertTrue(gaze_samples_quality_ok(samples))

    def test_too_few_targets_fail(self):
        collected = self._good_collected()[: GAZE_MIN_TARGETS - 1]
        samples = build_gaze_calibration_samples(collected)
        self.assertFalse(gaze_samples_quality_ok(samples))

    def test_degenerate_all_same_raw_fails(self):
        # Пользователь смотрел в одну точку: сырые значения не различаются.
        collected = [((0.5, 0.5), t) for t in gaze_target_points()]
        samples = build_gaze_calibration_samples(collected)
        self.assertFalse(gaze_samples_quality_ok(samples))

    def test_no_variation_on_one_axis_fails(self):
        # Разброс только по x, по y всё одинаково → подгонка y вырождена.
        collected = [
            ((0.10, 0.50), (0.15, 0.15)),
            ((0.50, 0.50), (0.50, 0.50)),
            ((0.90, 0.50), (0.85, 0.85)),
        ]
        samples = build_gaze_calibration_samples(collected)
        self.assertFalse(gaze_samples_quality_ok(samples))


class FittedMapTests(unittest.TestCase):
    """Сквозная проверка: собранные пары дают карту, верно отображающую углы."""

    def test_fitted_map_recovers_targets(self):
        # Сырой взгляд линейно связан с экраном: target = 0.8*raw + 0.1 по обеим
        # осям. Подгонка по углам+центру должна восстановить это отображение.
        def to_target(rx, ry):
            return (0.8 * rx + 0.1, 0.8 * ry + 0.1)

        raw_corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
        collected = [(raw, to_target(*raw)) for raw in raw_corners]
        samples = build_gaze_calibration_samples(collected)
        self.assertTrue(gaze_samples_quality_ok(samples))

        cal = GazeCalibration()
        self.assertTrue(cal.fit(samples))
        self.assertAlmostEqual(cal.ax, 0.8, places=4)
        self.assertAlmostEqual(cal.bx, 0.1, places=4)
        self.assertAlmostEqual(cal.ay, 0.8, places=4)
        self.assertAlmostEqual(cal.by, 0.1, places=4)

        # Карта переводит сырые углы в ожидаемые целевые точки.
        for raw in raw_corners:
            sx, sy = cal.apply(*raw)
            exp_x, exp_y = to_target(*raw)
            self.assertAlmostEqual(sx, exp_x, places=4)
            self.assertAlmostEqual(sy, exp_y, places=4)

    def test_too_few_samples_does_not_fit(self):
        # Деградация: одной пары мало → fit() сообщает об отказе.
        samples = build_gaze_calibration_samples([((0.5, 0.5), (0.5, 0.5))])
        cal = GazeCalibration()
        self.assertFalse(cal.fit(samples))
        # Карта осталась тождественной (калибровка не сломана).
        self.assertEqual((cal.ax, cal.bx, cal.ay, cal.by), (1.0, 0.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
