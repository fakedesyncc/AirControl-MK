"""Тесты миграции/валидации персистентного конфига (stdlib unittest).

Запуск:  python -m unittest tests.test_config_migration

Конфиг хранится в JSON и переживает смену версий приложения и ручную правку.
Здесь проверяем, что AppConfig.load устойчив к таким файлам:
  * старый/минимальный конфиг без новых секций → недостающее берётся из дефолтов;
  * лишние/неизвестные ключи → игнорируются без ошибки (forward-compat);
  * недопустимое значение enum-поля → откат к безопасному дефолту, без падения;
  * round-trip save→load сохраняет корректные значения без искажений.

Реальная камера/трекинг/модели не нужны — работаем только с (де)сериализацией
во временном файле.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.config import AppConfig


class ConfigMigrationTest(unittest.TestCase):
    def setUp(self):
        # Отдельный временный каталог на каждый тест: load() при отсутствии файла
        # создаёт его, поэтому изоляция важна.
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "config.json")

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, data: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # ---- forward-compat: старый/минимальный конфиг -------------------------

    def test_old_minimal_config_fills_missing_sections_with_defaults(self):
        """Старый файл без новых секций (fusion.gaze, scan_keyboard, swipe_*)
        загружается, а отсутствующее берётся из значений по умолчанию."""
        self._write({"profile_name": "legacy", "voice": {"language": "en-US"}})

        cfg = AppConfig.load(self.path)
        defaults = AppConfig()

        # Сохранилось то, что было в старом файле.
        self.assertEqual(cfg.profile_name, "legacy")
        self.assertEqual(cfg.voice.language, "en-US")

        # Новые/отсутствующие секции и поля — ровно дефолтные.
        self.assertEqual(cfg.scan_keyboard.scan_interval,
                         defaults.scan_keyboard.scan_interval)
        self.assertEqual(cfg.scan_keyboard.select_key,
                         defaults.scan_keyboard.select_key)
        self.assertEqual(cfg.gestures.swipe_backend,
                         defaults.gestures.swipe_backend)
        self.assertEqual(cfg.gestures.swipe_sequence_length,
                         defaults.gestures.swipe_sequence_length)
        self.assertEqual(cfg.fusion.gaze_enabled, defaults.fusion.gaze_enabled)
        self.assertEqual(cfg.fusion.gaze.smoothing_alpha,
                         defaults.fusion.gaze.smoothing_alpha)
        # Не указанное поле внутри присутствовавшей секции — тоже дефолт.
        self.assertEqual(cfg.voice.engine, defaults.voice.engine)

    # ---- неизвестные/лишние ключи ------------------------------------------

    def test_unknown_keys_are_ignored(self):
        """Незнакомые ключи (верхнего уровня и вложенные) не ломают загрузку."""
        self._write({
            "profile_name": "default",
            "totally_unknown_key": 123,
            "filter": {"type": "ema", "future_param": "whatever"},
            "another_future_section": {"a": 1, "b": [1, 2, 3]},
        })

        cfg = AppConfig.load(self.path)

        self.assertEqual(cfg.profile_name, "default")
        self.assertEqual(cfg.filter.type, "ema")
        self.assertFalse(hasattr(cfg, "totally_unknown_key"))
        self.assertFalse(hasattr(cfg.filter, "future_param"))

    # ---- недопустимые enum-значения ----------------------------------------

    def test_invalid_enum_values_fall_back_to_safe_defaults(self):
        """Мусорные значения enum-полей откатываются к безопасным дефолтам,
        приложение не падает."""
        self._write({
            "start_mode": "nope",
            "tracking": {"running_mode": "weird"},
            "filter": {"type": "bogus"},
            "cursor": {"dwell_profile": "??"},
            "gestures": {"recognizer": "nn", "swipe_backend": "transformer"},
            "voice": {"engine": "whisper"},
            "fusion": {"gaze_mode": "nope", "gaze": {"running_mode": "stream"}},
        })

        cfg = AppConfig.load(self.path)
        defaults = AppConfig()

        self.assertEqual(cfg.start_mode, defaults.start_mode)
        self.assertEqual(cfg.tracking.running_mode, defaults.tracking.running_mode)
        self.assertEqual(cfg.filter.type, defaults.filter.type)
        self.assertEqual(cfg.cursor.dwell_profile, defaults.cursor.dwell_profile)
        self.assertEqual(cfg.gestures.recognizer, defaults.gestures.recognizer)
        self.assertEqual(cfg.gestures.swipe_backend, defaults.gestures.swipe_backend)
        self.assertEqual(cfg.voice.engine, defaults.voice.engine)
        self.assertEqual(cfg.fusion.gaze_mode, defaults.fusion.gaze_mode)
        self.assertEqual(cfg.fusion.gaze.running_mode,
                         defaults.fusion.gaze.running_mode)

    def test_valid_enum_values_are_preserved(self):
        """Корректные (не дефолтные) значения enum-полей не трогаются."""
        self._write({
            "filter": {"type": "kalman"},
            "gestures": {"swipe_backend": "lstm"},
            "voice": {"engine": "vosk"},
            "fusion": {"gaze_mode": "cursor"},
            "tracking": {"running_mode": "image"},
        })

        cfg = AppConfig.load(self.path)

        self.assertEqual(cfg.filter.type, "kalman")
        self.assertEqual(cfg.gestures.swipe_backend, "lstm")
        self.assertEqual(cfg.voice.engine, "vosk")
        self.assertEqual(cfg.fusion.gaze_mode, "cursor")
        self.assertEqual(cfg.tracking.running_mode, "image")

    # ---- round-trip --------------------------------------------------------

    def test_round_trip_preserves_valid_values(self):
        """save → load сохраняет корректно настроенный конфиг без искажений."""
        cfg = AppConfig()
        cfg.profile_name = "roundtrip"
        cfg.filter.type = "one_euro"
        cfg.filter.ema_alpha = 0.42
        cfg.gestures.swipe_backend = "tcn"
        cfg.gestures.swipe_min_confidence = 0.71
        cfg.voice.engine = "vosk"
        cfg.fusion.gaze_mode = "cursor"
        cfg.cursor.dwell_profile = "fast"
        cfg.scan_keyboard.scan_interval = 1.7

        cfg.save(self.path)
        loaded = AppConfig.load(self.path)

        self.assertEqual(loaded.profile_name, "roundtrip")
        self.assertEqual(loaded.filter.type, "one_euro")
        self.assertAlmostEqual(loaded.filter.ema_alpha, 0.42)
        self.assertEqual(loaded.gestures.swipe_backend, "tcn")
        self.assertAlmostEqual(loaded.gestures.swipe_min_confidence, 0.71)
        self.assertEqual(loaded.voice.engine, "vosk")
        self.assertEqual(loaded.fusion.gaze_mode, "cursor")
        self.assertEqual(loaded.cursor.dwell_profile, "fast")
        self.assertAlmostEqual(loaded.scan_keyboard.scan_interval, 1.7)

    def test_round_trip_of_pristine_defaults_is_stable(self):
        """Дефолтный конфиг после save→load остаётся дефолтным (нет дрейфа)."""
        AppConfig().save(self.path)
        loaded = AppConfig.load(self.path)
        defaults = AppConfig()

        # Пути могут чиниться _repair_runtime_paths под текущее окружение,
        # поэтому сравниваем enum-поля, а не абсолютные пути к моделям.
        self.assertEqual(loaded.filter.type, defaults.filter.type)
        self.assertEqual(loaded.voice.engine, defaults.voice.engine)
        self.assertEqual(loaded.gestures.swipe_backend, defaults.gestures.swipe_backend)
        self.assertEqual(loaded.fusion.gaze_mode, defaults.fusion.gaze_mode)
        self.assertEqual(loaded.start_mode, defaults.start_mode)


if __name__ == "__main__":
    unittest.main()
