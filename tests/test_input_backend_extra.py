"""Дополнительные тесты обнаружения/кэширования backend ввода.

Запуск:  python -m unittest tests.test_input_backend_extra -v

Покрывают чистую логику без реального ввода в ОС: определение дисплей-сервера,
кэширование дорогих проверок (shutil.which, дисплей-сервер) и конкретность
текстов предупреждений/подсказок по платформам. Все внешние зависимости
(окружение, shutil.which) подменяются через unittest.mock.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.control import input_backend as ib


class TestDisplayServerDetection(unittest.TestCase):
    """Определение типа дисплей-сессии по переменным окружения."""

    def setUp(self):
        ib.reset_detection_cache()
        self.addCleanup(ib.reset_detection_cache)

    def test_xdg_session_type_takes_priority(self):
        for value, expected in (("wayland", "wayland"), ("x11", "x11")):
            ib.reset_detection_cache()
            with patch.dict(os.environ, {"XDG_SESSION_TYPE": value}, clear=True):
                self.assertEqual(ib._linux_display_server(), expected)

    def test_wayland_display_implies_wayland(self):
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True):
            self.assertEqual(ib._linux_display_server(), "wayland")

    def test_display_implies_x11(self):
        with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True):
            self.assertEqual(ib._linux_display_server(), "x11")

    def test_no_env_is_headless(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ib._linux_display_server(), "headless")


class TestDisplayServerCache(unittest.TestCase):
    """Дисплей-сервер кэшируется по сигнатуре окружения."""

    def setUp(self):
        ib.reset_detection_cache()
        self.addCleanup(ib.reset_detection_cache)

    def test_stable_env_returns_same_result(self):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True):
            first = ib._linux_display_server()
            second = ib._linux_display_server()
        self.assertEqual(first, "wayland")
        self.assertEqual(first, second)

    def test_cache_is_populated_and_reused(self):
        with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True):
            sig = ib._display_env_signature()
            self.assertNotIn(sig, ib._DISPLAY_SERVER_CACHE)
            first = ib._linux_display_server()
            self.assertIn(sig, ib._DISPLAY_SERVER_CACHE)
            self.assertEqual(ib._DISPLAY_SERVER_CACHE[sig], "x11")
            # Повторный вызов возвращает то же значение из кэша.
            self.assertEqual(ib._linux_display_server(), first)
            self.assertEqual(len(ib._DISPLAY_SERVER_CACHE), 1)

    def test_cache_recomputes_when_env_changes(self):
        with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True):
            self.assertEqual(ib._linux_display_server(), "x11")
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True):
            self.assertEqual(ib._linux_display_server(), "wayland")


class TestWhichCache(unittest.TestCase):
    """Мемоизация shutil.which для инструментов ввода."""

    def setUp(self):
        ib.reset_detection_cache()
        self.addCleanup(ib.reset_detection_cache)

    def test_which_result_is_stable(self):
        with patch.object(ib.shutil, "which", return_value="/usr/bin/ydotool"):
            self.assertEqual(ib._which_cached("ydotool"), "/usr/bin/ydotool")
            self.assertEqual(ib._which_cached("ydotool"), "/usr/bin/ydotool")

    def test_which_called_once_for_same_tool(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
            with patch.object(ib.shutil, "which",
                              return_value="/usr/bin/xdotool") as which:
                ib._which_cached("xdotool")
                ib._which_cached("xdotool")
                ib._which_cached("xdotool")
                self.assertEqual(which.call_count, 1)

    def test_reset_clears_which_cache(self):
        with patch.object(ib.shutil, "which", return_value="/x") as which:
            ib._which_cached("xdotool")
            ib.reset_detection_cache()
            ib._which_cached("xdotool")
            self.assertEqual(which.call_count, 2)

    def test_distinct_which_patches_do_not_leak(self):
        # Подмена which другим объектом даёт другой ключ кэша: устаревший путь
        # не возвращается (важно для совместимости с существующими тестами).
        with patch.dict(os.environ, {"PATH": ""}, clear=True):
            with patch.object(ib.shutil, "which", return_value="/usr/bin/ydotool"):
                self.assertEqual(ib._which_cached("ydotool"), "/usr/bin/ydotool")
            with patch.object(ib.shutil, "which", return_value=None):
                self.assertIsNone(ib._which_cached("ydotool"))


class TestWaylandWarningText(unittest.TestCase):
    """Предупреждение на Wayland указывает конкретную меру (ydotoold/uinput/Xorg)."""

    def setUp(self):
        ib.reset_detection_cache()
        self._orig_active = set(ib._ACTIVE_BACKENDS)
        ib._ACTIVE_BACKENDS.clear()
        ib._ACTIVE_BACKENDS.add("pynput")

        def _restore():
            ib._ACTIVE_BACKENDS.clear()
            ib._ACTIVE_BACKENDS.update(self._orig_active)
            ib.reset_detection_cache()

        self.addCleanup(_restore)

    def test_warns_to_use_xorg_or_ydotool_when_tool_missing(self):
        with patch.object(ib.sys, "platform", "linux"), \
             patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True), \
             patch.object(ib.shutil, "which", return_value=None):
            warning = ib.input_backend_warning()
        self.assertIsNotNone(warning)
        self.assertIn("Xorg", warning)
        self.assertIn("ydotool", warning)
        self.assertIn("uinput", warning)

    def test_warns_about_ydotoold_uinput_when_tool_present(self):
        with patch.object(ib.sys, "platform", "linux"), \
             patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True), \
             patch.object(ib.shutil, "which", return_value="/usr/bin/ydotool"):
            warning = ib.input_backend_warning()
        self.assertIsNotNone(warning)
        self.assertIn("ydotoold", warning)
        self.assertIn("uinput", warning)

    def test_headless_warning_mentions_display_session(self):
        with patch.object(ib.sys, "platform", "linux"), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(ib.shutil, "which", return_value=None):
            warning = ib.input_backend_warning()
        self.assertIsNotNone(warning)
        self.assertIn("DISPLAY", warning)


class TestFallbackHintRemediation(unittest.TestCase):
    """Подсказка при полном отказе backend конкретна по платформам."""

    def setUp(self):
        ib.reset_detection_cache()
        self.addCleanup(ib.reset_detection_cache)

    def test_macos_hint_mentions_accessibility(self):
        with patch.object(ib.sys, "platform", "darwin"):
            hint = ib._fallback_hint()
        self.assertIn("Accessibility", hint)

    def test_windows_hint_mentions_antivirus_or_smartscreen(self):
        with patch.object(ib.sys, "platform", "win32"):
            hint = ib._fallback_hint()
        self.assertTrue("SmartScreen" in hint or "антивирус" in hint)

    def test_linux_wayland_hint_mentions_ydotoold_or_xorg(self):
        with patch.object(ib.sys, "platform", "linux"), \
             patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True), \
             patch.object(ib.shutil, "which", return_value=None):
            hint = ib._fallback_hint()
        self.assertTrue("Xorg" in hint or "ydotoold" in hint)

    def test_linux_missing_tools_hint(self):
        with patch.object(ib.sys, "platform", "linux"), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(ib.shutil, "which", return_value=None):
            hint = ib._fallback_hint()
        self.assertIn("xdotool/ydotool", hint)


if __name__ == "__main__":
    unittest.main()
