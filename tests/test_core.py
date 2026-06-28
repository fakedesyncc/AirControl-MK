"""Тесты ключевой логики AirControl (stdlib unittest, без внешних зависимостей).

Запуск:  python -m unittest discover -s tests
Покрывают чистую логику, не требующую камеры/GUI: конфиг, фильтры, признаки,
движок жестов, тест Фиттса, ML-классификатор на синтетике.
"""

import math
import os
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.config import (
    AppConfig,
    DEFAULT_LOG_DIR,
    DEFAULT_ML_DATASET_PATH,
    DEFAULT_ML_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    FilterConfig,
    apply_assistive_profile,
)
from aircontrol.control.actions import ActionExecutor
from aircontrol.control.cursor import CursorController
from aircontrol.evaluation.fitts import FittsTest
from aircontrol.gestures import features as F
from aircontrol.gestures.engine import FrameGestures, GestureEngine
from aircontrol.gestures.synthetic import build_hand, generate_synthetic_dataset
from aircontrol.gestures.ml import MLPoseClassifier, train_from_dataset
from aircontrol.tracking.filters import create_filter
from aircontrol.tracking.hand_tracker import HandResult


class TestConfig(unittest.TestCase):
    def test_roundtrip(self):
        cfg = AppConfig()
        cfg.cursor.sensitivity = 1.7
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            cfg.save(path)
            loaded = AppConfig.load(path)
        self.assertAlmostEqual(loaded.cursor.sensitivity, 1.7)
        self.assertEqual(loaded.gestures.mapping["pinch_index"], "left_click")

    def test_forward_compatible(self):
        # Лишние/отсутствующие ключи не ломают загрузку.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            with open(path, "w") as f:
                f.write('{"profile_name": "x", "unknown_field": 1}')
            loaded = AppConfig.load(path)
        self.assertEqual(loaded.profile_name, "x")
        self.assertEqual(loaded.filter.type, "one_euro")  # дефолт
        self.assertEqual(loaded.camera.backend, "auto")
        self.assertFalse(loaded.input.dry_run)

    def test_relocatable_model_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            with open(path, "w") as f:
                f.write('{"tracking": {"model_path": "/missing/hand_landmarker.task"}}')
            loaded = AppConfig.load(path)
        self.assertEqual(loaded.tracking.model_path, DEFAULT_MODEL_PATH)

    def test_relocatable_data_paths(self):
        old = "/tmp/old/Hand Mouse Controller"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            with open(path, "w") as f:
                f.write(
                    "{"
                    f'"gestures": {{"ml_model_path": "{old}/aircontrol/data/gesture_model.npz", '
                    f'"ml_dataset_path": "{old}/aircontrol/data/gesture_dataset.npz"}},'
                    f'"evaluation": {{"log_dir": "{old}/aircontrol/data/logs"}},'
                    f'"telemetry": {{"log_dir": "{old}/aircontrol/data/logs"}}'
                    "}"
                )
            loaded = AppConfig.load(path)
        self.assertEqual(loaded.gestures.ml_model_path, DEFAULT_ML_MODEL_PATH)
        self.assertEqual(loaded.gestures.ml_dataset_path, DEFAULT_ML_DATASET_PATH)
        self.assertEqual(loaded.evaluation.log_dir, DEFAULT_LOG_DIR)
        self.assertEqual(loaded.telemetry.log_dir, DEFAULT_LOG_DIR)

    def test_assistive_profile(self):
        cfg = apply_assistive_profile(AppConfig())
        self.assertEqual(cfg.profile_name, "assistive")
        self.assertEqual(cfg.start_mode, "control")
        self.assertEqual((cfg.camera.width, cfg.camera.height), (480, 360))
        self.assertTrue(cfg.cursor.dwell_enabled)
        self.assertLess(cfg.cursor.active_region, 0.55)
        self.assertFalse(cfg.gestures.dynamic_enabled)
        self.assertFalse(cfg.gestures.bimanual_enabled)
        self.assertEqual(cfg.performance.detect_downscale, 0.5)
        self.assertEqual(cfg.performance.detect_max_fps, 24)


class TestInputSafety(unittest.TestCase):
    def test_dry_input_records_without_executing(self):
        with tempfile.TemporaryDirectory() as d:
            act = ActionExecutor(d, dry_run=True)
            act.execute("left_click")
            self.assertEqual(act.last_action, "left_click")
            self.assertEqual(act.input_status(), "DRY INPUT")
            act.scroll(2)
            self.assertEqual(act.last_action, "scroll:2")

    def test_can_toggle_dry_input_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            act = ActionExecutor(d, dry_run=True)
            act.set_dry_run(False)
            self.assertNotEqual(act.input_status(), "DRY INPUT")
            act.set_dry_run(True)
            self.assertEqual(act.input_status(), "DRY INPUT")

    def test_dispatch_keeps_user_action_name(self):
        with tempfile.TemporaryDirectory() as d:
            act = ActionExecutor(d, dry_run=True)
            act.execute("copy")
            self.assertEqual(act.last_action, "copy")

    def test_xdotool_backend_fallback(self):
        from aircontrol.control import input_backend as ib
        orig_keyboard = ib._KeyboardController
        orig_mouse = ib._MouseController
        orig_import_error = ib._IMPORT_ERROR
        orig_create_error = ib._CREATE_ERROR
        orig_active = set(ib._ACTIVE_BACKENDS)
        orig_ydotool_ready = ib._YDOTOOL_READY
        try:
            ib._KeyboardController = None
            ib._MouseController = None
            ib._IMPORT_ERROR = "pynput unavailable"
            ib._CREATE_ERROR = None
            ib._YDOTOOL_READY = False
            ib._ACTIVE_BACKENDS.clear()
            with patch.object(ib, "_xdotool_available", return_value=True):
                self.assertIsInstance(ib.create_keyboard_controller(), ib.XDoToolKeyboardController)
                self.assertIsInstance(ib.create_mouse_controller(), ib.XDoToolMouseController)
            self.assertEqual(ib.input_backend_name(), "xdotool")
            self.assertIsNone(ib.input_backend_error())
        finally:
            ib._KeyboardController = orig_keyboard
            ib._MouseController = orig_mouse
            ib._IMPORT_ERROR = orig_import_error
            ib._CREATE_ERROR = orig_create_error
            ib._YDOTOOL_READY = orig_ydotool_ready
            ib._ACTIVE_BACKENDS.clear()
            ib._ACTIVE_BACKENDS.update(orig_active)

    def test_ydotool_backend_preferred_on_wayland(self):
        from aircontrol.control import input_backend as ib
        orig_keyboard = ib._KeyboardController
        orig_mouse = ib._MouseController
        orig_import_error = ib._IMPORT_ERROR
        orig_create_error = ib._CREATE_ERROR
        orig_active = set(ib._ACTIVE_BACKENDS)
        orig_ydotool_ready = ib._YDOTOOL_READY
        try:
            ib._KeyboardController = object
            ib._MouseController = object
            ib._IMPORT_ERROR = None
            ib._CREATE_ERROR = None
            ib._YDOTOOL_READY = False
            ib._ACTIVE_BACKENDS.clear()
            with patch.object(ib.sys, "platform", "linux"), \
                 patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True), \
                 patch.object(ib.shutil, "which", return_value="/usr/bin/ydotool"), \
                 patch.object(ib.subprocess, "run") as run:
                run.return_value.returncode = 0
                self.assertIsInstance(ib.create_keyboard_controller(), ib.YDoToolKeyboardController)
                self.assertIsInstance(ib.create_mouse_controller(), ib.YDoToolMouseController)
            self.assertEqual(ib.input_backend_name(), "ydotool")
            self.assertIsNone(ib.input_backend_error())
        finally:
            ib._KeyboardController = orig_keyboard
            ib._MouseController = orig_mouse
            ib._IMPORT_ERROR = orig_import_error
            ib._CREATE_ERROR = orig_create_error
            ib._YDOTOOL_READY = orig_ydotool_ready
            ib._ACTIVE_BACKENDS.clear()
            ib._ACTIVE_BACKENDS.update(orig_active)

    def test_ydotool_command_encoding(self):
        from aircontrol.control import input_backend as ib
        self.assertEqual(ib._ydotool_button(ib.Button.left, click=True), "0xC0")
        self.assertEqual(ib._ydotool_button(ib.Button.right, down=True), "0x41")
        self.assertEqual(ib._ydotool_button(ib.Button.middle, up=True), "0x82")
        self.assertEqual(ib._linux_key_code("c"), 46)
        self.assertEqual(ib._linux_key_code(ib.Key.ctrl), 29)

    def test_xdotool_nonzero_exit_is_visible(self):
        from aircontrol.control import input_backend as ib
        with patch.object(ib.subprocess, "run") as run:
            run.return_value.returncode = 7
            with self.assertRaisesRegex(RuntimeError, "xdotool click 1 failed"):
                ib._run_xdotool(["click", "1"])

    def test_ydotool_nonzero_exit_is_visible(self):
        from aircontrol.control import input_backend as ib
        with patch.object(ib.subprocess, "run") as run:
            run.return_value.returncode = 9
            with self.assertRaisesRegex(RuntimeError, "ydotool click 0xC0 failed"):
                ib._run_ydotool(["click", "0xC0"])

    def test_ydotool_soft_probe_keeps_boolean_result(self):
        from aircontrol.control import input_backend as ib
        with patch.object(ib.subprocess, "run") as run:
            run.return_value.returncode = 9
            self.assertFalse(ib._run_ydotool(["click", "0x00"], raise_on_error=False))

    def test_wayland_pynput_backend_reports_risk(self):
        from aircontrol.control import input_backend as ib
        orig_active = set(ib._ACTIVE_BACKENDS)
        try:
            ib._ACTIVE_BACKENDS.clear()
            ib._ACTIVE_BACKENDS.add("pynput")
            with patch.object(ib.sys, "platform", "linux"), \
                 patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True), \
                 patch.object(ib.shutil, "which", return_value=None):
                warning = ib.input_backend_warning()
            self.assertIsNotNone(warning)
            self.assertIn("Wayland", warning)
        finally:
            ib._ACTIVE_BACKENDS.clear()
            ib._ACTIVE_BACKENDS.update(orig_active)

    def test_input_probe_moves_mouse_and_restores_position(self):
        from aircontrol.control import input_backend as ib

        class Mouse:
            def __init__(self):
                self.position = (20, 30)

        class Keyboard:
            pass

        orig_keyboard = ib._KeyboardController
        orig_mouse = ib._MouseController
        orig_import_error = ib._IMPORT_ERROR
        orig_create_error = ib._CREATE_ERROR
        orig_active = set(ib._ACTIVE_BACKENDS)
        try:
            ib._KeyboardController = Keyboard
            ib._MouseController = Mouse
            ib._IMPORT_ERROR = None
            ib._CREATE_ERROR = None
            ib._ACTIVE_BACKENDS.clear()
            with patch.object(ib.sys, "platform", "darwin"):
                probe = ib.probe_input_backend(move_mouse=True)
            self.assertEqual(probe["status"], "OK")
            self.assertTrue(probe["mouse_move"])
            self.assertIn("restored", probe["mouse_detail"])
        finally:
            ib._KeyboardController = orig_keyboard
            ib._MouseController = orig_mouse
            ib._IMPORT_ERROR = orig_import_error
            ib._CREATE_ERROR = orig_create_error
            ib._ACTIVE_BACKENDS.clear()
            ib._ACTIVE_BACKENDS.update(orig_active)

    def test_input_probe_reports_failed_mouse_move(self):
        from aircontrol.control import input_backend as ib

        class Mouse:
            def __init__(self):
                self._position = (20, 30)

            @property
            def position(self):
                return self._position

            @position.setter
            def position(self, value):
                pass

        class Keyboard:
            pass

        orig_keyboard = ib._KeyboardController
        orig_mouse = ib._MouseController
        orig_import_error = ib._IMPORT_ERROR
        orig_create_error = ib._CREATE_ERROR
        orig_active = set(ib._ACTIVE_BACKENDS)
        try:
            ib._KeyboardController = Keyboard
            ib._MouseController = Mouse
            ib._IMPORT_ERROR = None
            ib._CREATE_ERROR = None
            ib._ACTIVE_BACKENDS.clear()
            with patch.object(ib.sys, "platform", "darwin"):
                probe = ib.probe_input_backend(move_mouse=True)
            self.assertEqual(probe["status"], "FAIL")
            self.assertFalse(probe["mouse_move"])
            self.assertIn("did not change", probe["mouse_detail"])
        finally:
            ib._KeyboardController = orig_keyboard
            ib._MouseController = orig_mouse
            ib._IMPORT_ERROR = orig_import_error
            ib._CREATE_ERROR = orig_create_error
            ib._ACTIVE_BACKENDS.clear()
            ib._ACTIVE_BACKENDS.update(orig_active)

    def test_hotkey_releases_pressed_keys_after_backend_error(self):
        class Keyboard:
            def __init__(self):
                self.pressed = []
                self.released = []

            def press(self, key):
                self.pressed.append(key)
                if key == "x":
                    raise RuntimeError("press failed")

            def release(self, key):
                self.released.append(key)

        with tempfile.TemporaryDirectory() as d:
            act = ActionExecutor(d, dry_run=True)
            act.dry_run = False
            act.keyboard = Keyboard()
            act.hotkey("ctrl", "x")
            self.assertEqual(act.keyboard.pressed, ["ctrl", "x"])
            self.assertEqual(act.keyboard.released, ["ctrl"])

    def test_zoom_releases_modifier_after_mouse_error(self):
        class Keyboard:
            def __init__(self):
                self.pressed = []
                self.released = []

            def press(self, key):
                self.pressed.append(key)

            def release(self, key):
                self.released.append(key)

        class Mouse:
            def scroll(self, dx, dy):
                raise RuntimeError("scroll failed")

        with tempfile.TemporaryDirectory() as d:
            act = ActionExecutor(d, dry_run=True)
            act.dry_run = False
            act.keyboard = Keyboard()
            act.mouse = Mouse()
            act.mod = "ctrl"
            act.zoom(1)
            self.assertEqual(act.keyboard.pressed, ["ctrl"])
            self.assertEqual(act.keyboard.released, ["ctrl"])
            with patch("aircontrol.control.actions.input_backend_error", return_value=None), \
                 patch("aircontrol.control.actions.input_backend_warning", return_value=None):
                self.assertEqual(act.input_status(), "INPUT ERROR")
            self.assertIn("zoom", act.last_input_error)
            self.assertEqual(act.input_error_count, 1)

    def test_click_error_is_reported_in_input_status(self):
        class Mouse:
            def click(self, button, count=1):
                raise RuntimeError("permission denied")

        with tempfile.TemporaryDirectory() as d:
            act = ActionExecutor(d, dry_run=True)
            act.dry_run = False
            act.mouse = Mouse()
            act.left_click()
            with patch("aircontrol.control.actions.input_backend_error", return_value=None), \
                 patch("aircontrol.control.actions.input_backend_warning", return_value=None):
                self.assertEqual(act.input_status(), "INPUT ERROR")
            self.assertIn("left_click: permission denied", act.last_input_error)
            self.assertEqual(act.input_error_count, 1)

    def test_backend_controller_error_is_reported_in_input_status(self):
        from aircontrol.control.input_backend import YDoToolMouseController

        with tempfile.TemporaryDirectory() as d:
            act = ActionExecutor(d, dry_run=True)
            act.dry_run = False
            act.mouse = YDoToolMouseController()
            with patch("aircontrol.control.input_backend._run_ydotool",
                       side_effect=RuntimeError("ydotool denied")), \
                 patch("aircontrol.control.actions.input_backend_error", return_value=None), \
                 patch("aircontrol.control.actions.input_backend_warning", return_value=None):
                act.left_click()
                self.assertEqual(act.input_status(), "INPUT ERROR")
            self.assertIn("ydotool denied", act.last_input_error)
            self.assertEqual(act.input_error_count, 1)

    def test_voice_flac_converter_path_requires_existing_executable(self):
        from aircontrol.voice.recognizer import flac_converter_path
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "flac")
            with open(exe, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(exe, 0o755)
            self.assertEqual(flac_converter_path(lambda: exe), exe)
            self.assertIsNone(flac_converter_path(lambda: os.path.join(d, "missing-flac")))


class TestCursorBackend(unittest.TestCase):
    class DummyMouse:
        def __init__(self, pos=(0, 0)):
            self.position = pos

    def test_cursor_rewires_mouse_backend_after_safe_toggle(self):
        cfg = AppConfig()
        first = self.DummyMouse((1, 1))
        second = self.DummyMouse((10, 10))
        cursor = CursorController(cfg.cursor, cfg.filter, first, (100, 100))
        cursor.set_mouse(second)
        fg = FrameGestures(hand_detected=True, cursor_norm=(0.5, 0.5))
        cursor.update(fg, 1.0)
        cursor.step()
        self.assertEqual(first.position, (1, 1))
        self.assertNotEqual(second.position, (10, 10))


class TestLauncherConfig(unittest.TestCase):
    def test_preview_forces_view_and_dry_input(self):
        from aircontrol.launcher import prepare_launch_config
        cfg = AppConfig()
        cfg.start_mode = "control"
        cfg.input.dry_run = False
        prepare_launch_config(cfg, assistive=False, dry_input=True, start_mode="view")
        self.assertEqual(cfg.start_mode, "view")
        self.assertTrue(cfg.input.dry_run)

    def test_assistive_control_uses_control_mode(self):
        from aircontrol.launcher import prepare_launch_config
        cfg = prepare_launch_config(AppConfig(), assistive=True, dry_input=False)
        self.assertEqual(cfg.profile_name, "assistive")
        self.assertEqual(cfg.start_mode, "control")
        self.assertFalse(cfg.input.dry_run)

    def test_control_preflight_required_only_for_real_control(self):
        from aircontrol.launcher import _requires_control_preflight

        cfg = AppConfig()
        cfg.start_mode = "control"
        cfg.input.dry_run = False
        self.assertTrue(_requires_control_preflight(cfg))

        cfg.input.dry_run = True
        self.assertFalse(_requires_control_preflight(cfg))

        cfg.start_mode = "view"
        cfg.input.dry_run = False
        self.assertFalse(_requires_control_preflight(cfg))

    def test_launcher_status_from_readiness_summary(self):
        from aircontrol.launcher import _launcher_status_from_summary
        self.assertIn("проблему", _launcher_status_from_summary([
            "=== AirControl readiness summary ===",
            "Status: needs attention before assistive control.",
        ]))
        self.assertIn("Камера проверится", _launcher_status_from_summary([
            "Status: ready for safe training.",
            "- Camera was not opened during this check.",
        ]))

    def test_control_preflight_message_only_when_attention_needed(self):
        from aircontrol.launcher import _control_preflight_message

        self.assertIsNone(_control_preflight_message([
            "=== AirControl readiness summary ===",
            "Status: ready for safe training.",
        ]))
        message = _control_preflight_message([
            "=== AirControl readiness summary ===",
            "Status: needs attention before assistive control.",
            "- OS input backend is unavailable: gestures may be detected but cannot control the computer.",
        ])
        self.assertIn("не подтвердил", message)
        self.assertIn("Безопасная тренировка", message)

    def test_control_preflight_opens_diagnostics_when_user_declines(self):
        from aircontrol.launcher import _confirm_control_preflight

        class Root:
            def after(self, _delay, callback):
                callback()

        class Status:
            value = ""

            def set(self, value):
                self.value = value

        opened = []
        status = Status()
        ok = _confirm_control_preflight(
            Root(),
            status_var=status,
            diagnostics_callback=lambda: opened.append(True),
            report_builder=lambda **_kwargs: "doctor",
            summary_builder=lambda _report: [
                "=== AirControl readiness summary ===",
                "Status: needs attention before assistive control.",
                "- OS input backend is unavailable: gestures may be detected but cannot control the computer.",
            ],
            ask_yes_no=lambda _title, _message: False,
        )

        self.assertFalse(ok)
        self.assertEqual(opened, [True])
        self.assertIn("не запущено", status.value)

    def test_control_preflight_continues_when_ready(self):
        from aircontrol.launcher import _confirm_control_preflight

        ok = _confirm_control_preflight(
            object(),
            report_builder=lambda **_kwargs: "doctor",
            summary_builder=lambda _report: [
                "=== AirControl readiness summary ===",
                "Status: ready for safe training.",
            ],
            ask_yes_no=lambda _title, _message: (_ for _ in ()).throw(AssertionError("should not ask")),
        )

        self.assertTrue(ok)

    def test_first_run_status_ready_path(self):
        from aircontrol.launcher import build_first_run_status

        statuses = build_first_run_status(
            "OpenCV: OK\nMediaPipe: OK\nHand model: OK (/tmp/model)\nTkinter: OK\n"
            "Camera scan: 0..1\ncamera[0]: OK frame=(360, 480, 3)\n"
            "input backend: OK (pynput)\n"
            "input probe: OK (backend=pynput; mouse_move=requested; backend initialized)\n"
            "input mouse move probe: OK (mouse moved and was restored near the original position)\n",
            ["Status: ready for safe training."],
        )

        by_id = {item["id"]: item for item in statuses}
        self.assertEqual(by_id["camera"]["status"], "ok")
        self.assertEqual(by_id["input"]["status"], "ok")
        self.assertEqual(by_id["performance"]["status"], "pending")
        self.assertEqual(by_id["next"]["status"], "ok")

    def test_first_run_status_blocks_when_camera_missing(self):
        from aircontrol.launcher import build_first_run_status

        statuses = build_first_run_status(
            "OpenCV: OK\nMediaPipe: OK\nHand model: OK (/tmp/model)\nTkinter: OK\n"
            "Camera scan: 0..1\ncamera[0]: not available\n"
            "input backend: OK (pynput)\ninput probe: OK (backend=pynput)\n",
            ["Status: needs attention before assistive control."],
        )

        by_id = {item["id"]: item for item in statuses}
        self.assertEqual(by_id["camera"]["status"], "fail")
        self.assertEqual(by_id["next"]["status"], "fail")
        self.assertIn("ZIP", by_id["next"]["message"])

    def test_first_run_status_warns_when_input_probe_skipped(self):
        from aircontrol.launcher import build_first_run_status

        statuses = build_first_run_status(
            "OpenCV: OK\nMediaPipe: OK\nHand model: OK (/tmp/model)\nTkinter: OK\n"
            "Camera scan: 0..1\ncamera[0]: OK frame=(360, 480, 3)\n"
            "input backend: OK (ydotool)\n"
            "input probe: WARN (backend=ydotool; mouse_move=requested; visible movement skipped)\n"
            "input mouse move probe: SKIPPED (ydotool cannot safely read the current cursor position)\n",
            ["Status: ready for safe training."],
        )

        by_id = {item["id"]: item for item in statuses}
        self.assertEqual(by_id["camera"]["status"], "ok")
        self.assertEqual(by_id["input"]["status"], "warn")
        self.assertEqual(by_id["next"]["status"], "warn")

    def test_first_run_report_contains_summary_and_doctor(self):
        from aircontrol.launcher import format_first_run_report

        report = format_first_run_report(
            [{"id": "camera", "title": "Камера", "status": "ok", "message": "готова"}],
            ["Status: ready for safe training."],
            "OpenCV: OK",
        )

        self.assertIn("AirControl first-run wizard", report)
        self.assertIn("Status: ready for safe training.", report)
        self.assertIn("OpenCV: OK", report)

    def test_launcher_launch_success_destroys_root_and_runs_app(self):
        from aircontrol.launcher import _launch_aircontrol_from_launcher

        class Root:
            destroyed = False

            def destroy(self):
                self.destroyed = True

        class App:
            def __init__(self, cfg):
                self.cfg = cfg

            def run(self):
                calls.append(self.cfg)

        calls = []
        root = Root()
        ok = _launch_aircontrol_from_launcher(root, AppConfig(), App)
        self.assertTrue(ok)
        self.assertTrue(root.destroyed)
        self.assertEqual(len(calls), 1)

    def test_launcher_launch_failure_shows_support_error(self):
        from aircontrol.launcher import _launch_aircontrol_from_launcher

        class Root:
            destroyed = False

            def destroy(self):
                self.destroyed = True

        def broken_app(_cfg):
            raise RuntimeError("camera unavailable")

        root = Root()
        with patch("aircontrol.launcher._show_launcher_startup_error",
                   return_value=("/tmp/crash.log", "/tmp/support.zip", "msg")) as show:
            ok = _launch_aircontrol_from_launcher(root, AppConfig(), broken_app)
        self.assertFalse(ok)
        self.assertTrue(root.destroyed)
        show.assert_called_once()

    def test_launcher_calibration_failure_shows_support_error(self):
        from aircontrol.launcher import _run_calibration_from_launcher

        class Root:
            destroyed = False

            def destroy(self):
                self.destroyed = True

        def broken_calibration(_cfg):
            raise RuntimeError("tk display failed")

        root = Root()
        with patch("aircontrol.launcher._show_launcher_startup_error",
                   return_value=("/tmp/crash.log", "/tmp/support.zip", "msg")) as show:
            ok = _run_calibration_from_launcher(root, AppConfig(), broken_calibration)
        self.assertFalse(ok)
        self.assertTrue(root.destroyed)
        show.assert_called_once()


class TestVoiceAvailability(unittest.TestCase):
    def test_voice_status_without_microphone_backend(self):
        from aircontrol.voice import recognizer as vr
        with patch.object(vr, "SPEECH_AVAILABLE", True), \
             patch.object(vr, "MICROPHONE_BACKEND_AVAILABLE", False):
            self.assertEqual(vr._initial_status(AppConfig().voice), "microphone backend unavailable")

    def test_voice_recognizer_keeps_recognize_method(self):
        from aircontrol.voice.recognizer import VoiceRecognizer
        self.assertTrue(hasattr(VoiceRecognizer, "_recognize"))


class TestCalibrationMath(unittest.TestCase):
    def test_active_region_from_motion_span(self):
        from aircontrol.ui.calibration import compute_active_region
        samples = [(0.2 + i * 0.02, 0.4) for i in range(20)]
        self.assertEqual(compute_active_region(samples), 0.38)

    def test_active_region_rejects_too_few_samples(self):
        from aircontrol.ui.calibration import compute_active_region
        self.assertIsNone(compute_active_region([(0.5, 0.5)] * 5))

    def test_pinch_thresholds_from_separated_samples(self):
        from aircontrol.ui.calibration import compute_pinch_thresholds
        thresholds = compute_pinch_thresholds([0.8] * 8, [0.25] * 8)
        self.assertEqual(thresholds, (0.443, 0.608))

    def test_pinch_thresholds_reject_small_gap(self):
        from aircontrol.ui.calibration import compute_pinch_thresholds
        self.assertIsNone(compute_pinch_thresholds([0.5] * 8, [0.48] * 8))


class TestFilters(unittest.TestCase):
    def _jitter(self, ftype):
        cfg = FilterConfig(); cfg.type = ftype
        f = create_filter(cfg)
        rng = np.random.default_rng(0)
        out, t = [], 0.0
        for _ in range(200):
            t += 1 / 60
            v = 0.5 + rng.normal(0, 0.01)
            out.append(f.filter(v, v, t)[0])
        return float(np.std(out[50:]))

    def test_filters_reduce_jitter(self):
        raw = self._jitter("none")
        for ftype in ("ema", "one_euro", "kalman"):
            self.assertLess(self._jitter(ftype), raw,
                            f"{ftype} должен снижать jitter относительно none")

    def test_one_euro_beats_ema(self):
        self.assertLess(self._jitter("one_euro"), self._jitter("ema"))


class TestFeatures(unittest.TestCase):
    def test_invariance_to_translation_and_scale(self):
        rng = np.random.default_rng(1)
        lm = build_hand([0.05, 0.05, 0.95, 0.95], 0.2, rng, noise=0.0)
        feat1 = F.extract_features(lm)
        moved = lm.copy(); moved[:, :2] += 0.1          # сдвиг
        moved[:, :2] *= 1.0
        feat2 = F.extract_features(moved)
        # Сдвиг не должен менять признаки (инвариантность к трансляции).
        self.assertLess(np.abs(feat1 - feat2).max(), 1e-4)

    def test_feature_dim(self):
        rng = np.random.default_rng(2)
        lm = build_hand([0, 0, 0, 0], 1.0, rng)
        self.assertEqual(F.extract_features(lm).shape, (42,))


class TestEngine(unittest.TestCase):
    def test_release_on_lost_hand(self):
        eng = GestureEngine(AppConfig().gestures)
        # Принудительно «зажат» индекс — потеря руки должна дать left_up.
        eng.index.pinched = True
        eng.index.ignore_release = False
        fg = eng.process(None)
        self.assertIn("left_up", [e.action for e in fg.events])

    def test_open_palm_freezes(self):
        rng = np.random.default_rng(3)
        eng = GestureEngine(AppConfig().gestures)
        lm = build_hand([0.05, 0.05, 0.05, 0.05], 1.0, rng, noise=0.0)
        fg = eng.process(HandResult(landmarks=lm, handedness="Right", score=0.9))
        self.assertEqual(fg.pose, "open_palm")
        self.assertTrue(fg.frozen)


class TestPreviewMode(unittest.TestCase):
    def test_keyboard_shortcuts_work_in_latin_and_cyrillic_layouts(self):
        from aircontrol.app import resolve_key_command
        pairs = [
            (("f", "f"), ("Cyrillic_a", "а"), "cycle_filter"),
            (("g", "g"), ("Cyrillic_pe", "п"), "toggle_recognizer"),
            (("d", "d"), ("Cyrillic_ve", "в"), "toggle_dwell"),
            (("l", "l"), ("Cyrillic_de", "д"), "toggle_landmarks"),
            (("h", "h"), ("Cyrillic_er", "р"), "toggle_hud"),
        ]
        for latin, cyrillic, command in pairs:
            self.assertEqual(resolve_key_command(*latin), command)
            self.assertEqual(resolve_key_command(*cyrillic), command)

    def test_keyboard_shortcuts_cover_mode_and_service_keys(self):
        from aircontrol.app import resolve_key_command
        self.assertEqual(resolve_key_command("1", "1"), "mode_view")
        self.assertEqual(resolve_key_command("2", "2"), "mode_control")
        self.assertEqual(resolve_key_command("KP_Add", ""), "sensitivity_up")
        self.assertEqual(resolve_key_command("minus", "-"), "sensitivity_down")
        self.assertEqual(resolve_key_command("Escape", "\x1b"), "close")
        self.assertEqual(resolve_key_command("F2", ""), "fitts_gesture")
        self.assertEqual(resolve_key_command("F3", ""), "fitts_mouse")

    def test_preview_classifies_without_input_events(self):
        from aircontrol.app import build_preview_gestures
        from aircontrol.gestures.heuristic import HeuristicPoseClassifier
        rng = np.random.default_rng(31)
        lm = build_hand([0.05, 0.05, 0.05, 0.05], 1.0, rng, noise=0.0)
        hand = HandResult(landmarks=lm, handedness="Right", score=0.9)
        fg = build_preview_gestures(HeuristicPoseClassifier(), hand)
        self.assertTrue(fg.hand_detected)
        self.assertEqual(fg.pose, "open_palm")
        self.assertTrue(fg.frozen)
        self.assertEqual(fg.events, [])
        self.assertFalse(fg.listening_requested)

    def test_preview_without_hand_is_empty(self):
        from aircontrol.app import build_preview_gestures
        from aircontrol.gestures.heuristic import HeuristicPoseClassifier
        fg = build_preview_gestures(HeuristicPoseClassifier(), None)
        self.assertFalse(fg.hand_detected)
        self.assertEqual(fg.pose, "none")

    def test_view_mode_reports_low_fps_without_hand_warning(self):
        from aircontrol.app import build_runtime_health_lines
        lines = build_runtime_health_lines(
            mode="view",
            input_status="DRY INPUT",
            fps=15.2,
            detect_ms=82.0,
            auto_tuned=False,
            last_frame_age=0.2,
            hand_detected=False,
            mode_age=5.0,
        )
        self.assertIn("Low FPS 15.2: automatic light mode pending", lines)
        self.assertIn("Slow detection 82ms: reduce resolution or lighting load", lines)
        self.assertFalse(any("Hand not found" in line for line in lines))

    def test_control_mode_reports_missing_hand(self):
        from aircontrol.app import build_runtime_health_lines
        lines = build_runtime_health_lines(
            mode="control",
            input_status="INPUT pynput",
            fps=30.0,
            detect_ms=20.0,
            auto_tuned=False,
            last_frame_age=0.1,
            hand_detected=False,
            mode_age=4.0,
        )
        self.assertIn("Hand not found: show the full palm inside the camera frame", lines)

    def test_runtime_health_reports_input_risk(self):
        from aircontrol.app import build_runtime_health_lines
        lines = build_runtime_health_lines(
            mode="control",
            input_status="INPUT RISK",
            fps=30.0,
            detect_ms=20.0,
            auto_tuned=False,
            last_frame_age=0.1,
            hand_detected=True,
            mode_age=4.0,
        )
        self.assertIn("Input RISK: this Linux session may block clicks and keys", lines)

    def test_runtime_health_reports_input_error(self):
        from aircontrol.app import build_runtime_health_lines
        lines = build_runtime_health_lines(
            mode="control",
            input_status="INPUT ERROR",
            last_input_error="left_click: permission denied",
            input_error_age=0.5,
            fps=30.0,
            detect_ms=20.0,
            auto_tuned=False,
            last_frame_age=0.1,
            hand_detected=True,
            mode_age=4.0,
        )
        self.assertIn("Input ERROR: left_click: permission denied", lines)

    def test_runtime_health_reports_capped_light_mode(self):
        from aircontrol.app import build_runtime_health_lines
        lines = build_runtime_health_lines(
            mode="control",
            input_status="INPUT pynput",
            fps=15.4,
            detect_ms=30.0,
            auto_tuned=True,
            detect_max_fps=16,
            low_perf_reason="FPS 15.2",
            last_frame_age=0.1,
            hand_detected=True,
            mode_age=8.0,
        )
        self.assertIn("Light mode ON: detection capped at 16 FPS", lines)
        self.assertFalse(any(line.startswith("Low FPS") for line in lines))

    def test_runtime_tune_is_more_aggressive_for_assistive_profile(self):
        from aircontrol.app import apply_runtime_performance_tune
        cfg = apply_assistive_profile(AppConfig())
        apply_runtime_performance_tune(cfg, deep=False)
        self.assertEqual(cfg.performance.detect_downscale, 0.4)
        self.assertEqual(cfg.performance.detect_max_fps, 16)
        self.assertFalse(cfg.ui.show_landmarks)
        self.assertFalse(cfg.ui.show_particles)

    def test_runtime_tune_keeps_default_first_step_moderate(self):
        from aircontrol.app import apply_runtime_performance_tune
        cfg = AppConfig()
        apply_runtime_performance_tune(cfg, deep=False)
        self.assertEqual(cfg.performance.detect_downscale, 0.5)
        self.assertEqual(cfg.performance.detect_max_fps, 20)
        self.assertTrue(cfg.ui.show_landmarks)


class TestFitts(unittest.TestCase):
    def test_throughput_positive(self):
        cfg = AppConfig().evaluation
        cfg.num_targets = 9; cfg.target_widths = [50]; cfg.ring_amplitudes = [300]
        cfg.repetitions = 1
        test = FittsTest(cfg, 1920, 1080)
        rng = np.random.default_rng(0)
        t = 0.0
        while not test.finished:
            tx, ty = test.current_target["pos"]
            t += 0.5
            test.register_click(tx + rng.normal(0, 5), ty + rng.normal(0, 5), t)
        s = test.summary()
        self.assertGreater(s["throughput_mean"], 0)
        self.assertLessEqual(s["error_rate_mean"], 1.0)


class TestML(unittest.TestCase):
    def test_train_predict_synthetic(self):
        ds = generate_synthetic_dataset(per_pose=80, seed=5)
        with tempfile.TemporaryDirectory() as d:
            ds_path = os.path.join(d, "ds.npz")
            model_path = os.path.join(d, "model.npz")
            ds.save(ds_path)
            metrics = train_from_dataset(ds_path, model_path, backend="knn")
            self.assertEqual(metrics["n_samples"], 80 * 5)
            clf = MLPoseClassifier.load(model_path)
            self.assertIsNotNone(clf)
            rng = np.random.default_rng(6)
            lm = build_hand([0.05, 0.95, 0.95, 0.95], 0.15, rng)  # point
            label, conf = clf.predict(lm)
            self.assertIn(label, ["fist", "none", "open_palm", "peace", "point"])


class TestDynamic(unittest.TestCase):
    def _swipe(self, dx, dy, steps=8, dt=0.04):
        from aircontrol.gestures.dynamic import DynamicGestureRecognizer
        r = DynamicGestureRecognizer()
        t, x, y, res = 0.0, 0.5, 0.5, None
        for _ in range(steps):
            t += dt; x += dx / steps; y += dy / steps
            out = r.update(x, y, t, active=True)
            if out:
                res = out
        return res

    def test_swipe_directions(self):
        self.assertEqual(self._swipe(0.3, 0), "swipe_right")
        self.assertEqual(self._swipe(-0.3, 0), "swipe_left")
        self.assertEqual(self._swipe(0, -0.3), "swipe_up")
        self.assertEqual(self._swipe(0, 0.3), "swipe_down")

    def test_small_and_slow_rejected(self):
        self.assertIsNone(self._swipe(0.05, 0))           # слишком мелкое
        self.assertIsNone(self._swipe(0.3, 0, steps=20, dt=0.05))  # слишком медленное


class TestBimanual(unittest.TestCase):
    def _hand(self, cx):
        lm = np.zeros((21, 3), dtype=np.float32)
        lm[0] = [cx, 0.6, 0]; lm[9] = [cx, 0.4, 0]; lm[5] = [cx - 0.1, 0.5, 0]
        lm[8] = [cx, 0.3, 0]; lm[4] = [cx, 0.305, 0]   # щипок
        return HandResult(landmarks=lm, handedness="Right", score=0.9)

    def test_zoom(self):
        from aircontrol.gestures.bimanual import BimanualController
        bm = BimanualController(AppConfig().gestures)
        acts = []
        for d in np.linspace(0.1, 0.5, 12):
            acts += bm.process([self._hand(0.5 - d / 2), self._hand(0.5 + d / 2)])
        self.assertTrue(all(a == "zoom_in" for a in acts))
        self.assertGreater(len(acts), 0)
        # одна рука — нет зума
        self.assertEqual(bm.process([self._hand(0.5)]), [])


class TestStabilization(unittest.TestCase):
    def test_spurious_pose_smoothed(self):
        eng = GestureEngine(AppConfig().gestures)
        for _ in range(4):
            eng._stabilize("open_palm", 1.0)
        # один ложный кадр 'fist' не должен перебить 4 уверенных 'open_palm'
        pose, _ = eng._stabilize("fist", 1.0)
        self.assertEqual(pose, "open_palm")


class TestDiagnostics(unittest.TestCase):
    def test_doctor_without_camera_scan(self):
        from aircontrol.diagnostics import build_report
        report = build_report(scan_camera=False)
        self.assertIn("AirControl doctor", report)
        self.assertIn("System resources", report)
        self.assertIn("CPU cores", report)
        self.assertIn("SpeechRecognition FLAC converter", report)
        self.assertIn("Runtime config", report)
        self.assertIn("Configured dry-input", report)
        self.assertIn("input probe:", report)
        self.assertNotIn("input mouse move probe:", report)
        self.assertIn("Camera scan: skipped", report)

    def test_native_helper_report_reads_json(self):
        from aircontrol import diagnostics

        class Result:
            stdout = (
                '{"app":"AirControl","helper_version":"0.1.0","os":"linux",'
                '"arch":"amd64","display_server":"x11","tools":[]}'
            )

        with patch.object(diagnostics, "_native_helper_path", return_value="/tmp/aircontrol-helper"), \
             patch.object(diagnostics.subprocess, "run", return_value=Result()) as run:
            report = diagnostics.native_helper_report()

        self.assertEqual(report["app"], "AirControl")
        self.assertEqual(report["_helper_path"], "/tmp/aircontrol-helper")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["/tmp/aircontrol-helper", "doctor", "--json"])

    def test_doctor_summary_flags_input_probe_failure(self):
        from aircontrol.diagnostics import summarize_doctor_report
        summary = "\n".join(summarize_doctor_report(
            "=== AirControl doctor ===\n"
            "Frozen bundle: True\n"
            "OpenCV: OK\nMediaPipe: OK\nNumPy: OK\nPillow: OK\n"
            "Tkinter: OK\nHand model: OK (/tmp/model)\n"
            "Pynput: OK\ninput backend: OK (pynput)\n"
            "input probe: FAIL (backend=pynput; mouse_move=requested; mouse position did not change)\n"
            "input mouse move probe: FAIL (mouse position did not change)\n"
            "Camera scan: skipped\n"
        ))
        self.assertIn("OS input probe failed", summary)
        self.assertIn("проверьте разрешения", summary)

    def test_doctor_summary_flags_input_failure(self):
        from aircontrol.diagnostics import summarize_doctor_report
        summary = "\n".join(summarize_doctor_report(
            "=== AirControl doctor ===\n"
            "Frozen bundle: True\n"
            "OpenCV: OK\nMediaPipe: OK\nNumPy: OK\nPillow: OK\n"
            "Tkinter: OK\nHand model: OK (/tmp/model)\n"
            "Pynput: OK\ninput backend: FAIL (Wayland denied input)\n"
            "Camera scan: skipped\n"
        ))
        self.assertIn("needs attention", summary)
        self.assertIn("OS input backend is unavailable", summary)
        self.assertIn("Camera was not opened", summary)
        self.assertIn("Что сделать дальше", summary)
        self.assertIn("не будет нажимать мышь", summary)
        self.assertIn("Xorg", summary)
        self.assertIn("ydotoold", summary)

    def test_doctor_summary_flags_input_warning(self):
        from aircontrol.diagnostics import summarize_doctor_report
        summary = "\n".join(summarize_doctor_report(
            "=== AirControl doctor ===\n"
            "Frozen bundle: True\n"
            "OpenCV: OK\nMediaPipe: OK\nNumPy: OK\nPillow: OK\n"
            "Tkinter: OK\nHand model: OK (/tmp/model)\n"
            "Pynput: OK\n"
            "input backend: WARN (pynput; Wayland session with non-ydotool input backend)\n"
            "Camera scan: skipped\n"
        ))
        self.assertIn("needs attention", summary)
        self.assertIn("OS input backend needs attention", summary)
        self.assertIn("Xorg", summary)
        self.assertIn("ydotoold", summary)

    def test_doctor_summary_notes_missing_flac_converter(self):
        from aircontrol.diagnostics import summarize_doctor_report
        summary = "\n".join(summarize_doctor_report(
            "=== AirControl doctor ===\n"
            "Frozen bundle: True\n"
            "OpenCV: OK\nMediaPipe: OK\nNumPy: OK\nPillow: OK\n"
            "Tkinter: OK\nHand model: OK (/tmp/model)\n"
            "Pynput: OK\ninput backend: OK (pynput)\n"
            "SpeechRecognition FLAC converter: missing\n"
            "Camera scan: skipped\n"
        ))
        self.assertIn("ready for safe training", summary)
        self.assertIn("Online Google voice commands are disabled", summary)
        self.assertIn("Голосовые команды можно пропустить", summary)
        self.assertIn("Начать ассистивное управление", summary)

    def test_support_bundle(self):
        from aircontrol.diagnostics import save_support_bundle
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "support.zip")
            out = save_support_bundle(path, scan_camera=False,
                                      runtime_info={
                                          "fps": 12.5,
                                          "detect_ms": 85.0,
                                          "mode": "control",
                                          "start_mode": "control",
                                          "profile": "assistive",
                                          "safe_input": True,
                                          "dwell_enabled": True,
                                          "input_status": "DRY INPUT",
                                          "hand_detected": False,
                                          "last_action": "left_click",
                                          "seconds_since_action": 1.25,
                                          "auto_tuned": True,
                                          "low_perf_reason": "FPS 12.5, detect 85ms",
                                          "performance": {
                                              "detect_downscale": 0.4,
                                              "detect_max_fps": 16,
                                              "show_landmarks": False,
                                              "show_particles": False,
                                              "show_trail": False,
                                          },
                                      })
            self.assertEqual(out, path)
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
                readme = zf.read("README.txt").decode("utf-8")
                manifest = zf.read("support-manifest.json").decode("utf-8")
                runtime = zf.read("runtime.json").decode("utf-8")
                summary = zf.read("runtime-summary.txt").decode("utf-8")
                doctor_summary = zf.read("doctor-summary.txt").decode("utf-8")
            self.assertIn("README.txt", names)
            self.assertIn("support-manifest.json", names)
            self.assertIn("doctor.txt", names)
            self.assertIn("doctor-summary.txt", names)
            self.assertIn("config.json", names)
            self.assertIn("runtime.json", names)
            self.assertIn("runtime-summary.txt", names)
            self.assertIn("AirControl support bundle", readme)
            self.assertIn('"app": "AirControl"', manifest)
            self.assertIn('"runtime_included": true', manifest)
            self.assertIn('"path": "runtime-summary.txt"', manifest)
            self.assertIn('"fps": 12.5', runtime)
            self.assertIn('"profile": "assistive"', runtime)
            self.assertIn('"dwell_enabled": true', runtime)
            self.assertIn('"summary":', runtime)
            self.assertIn("Profile: assistive", summary)
            self.assertIn("Control path: OFF (Safe input)", summary)
            self.assertIn("Safe input: ON", summary)
            self.assertIn("Dwell-click: ON", summary)
            self.assertIn("Last action: left_click (1.25s ago)", summary)
            self.assertIn("Safe input is ON", summary)
            self.assertIn("Low FPS", summary)
            self.assertIn("Slow hand detection", summary)
            self.assertIn("Performance: downscale=0.4, max_detect_fps=16", summary)
            self.assertIn("Auto tune: ON", summary)
            self.assertIn("Auto tune reason: FPS 12.5, detect 85ms", summary)
            self.assertIn("AirControl readiness summary", doctor_summary)
            self.assertIn("Что сделать дальше", doctor_summary)
            self.assertIn("готовый AirControl-Setup.exe", doctor_summary)

    def test_support_bundle_includes_native_helper_when_available(self):
        from aircontrol import diagnostics

        native_report = {
            "app": "AirControl",
            "helper_version": "0.1.0",
            "os": "linux",
            "arch": "amd64",
            "display_server": "x11",
            "tools": [{"name": "xdotool", "found": True}],
            "_helper_path": "/tmp/aircontrol-helper",
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "support.zip")
            with patch.object(diagnostics, "native_helper_report", return_value=native_report):
                diagnostics.save_support_bundle(path, scan_camera=False)

            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
                manifest = zf.read("support-manifest.json").decode("utf-8")
                native_json = zf.read("native-helper.json").decode("utf-8")
                doctor = zf.read("doctor.txt").decode("utf-8")

        self.assertIn("native-helper.json", names)
        self.assertIn('"native_helper_included": true', manifest)
        self.assertIn('"path": "native-helper.json"', manifest)
        self.assertIn('"helper_version": "0.1.0"', native_json)
        self.assertNotIn("_helper_path", native_json)
        self.assertIn("Native helper: OK", doctor)

    def test_support_readme_without_runtime_does_not_reference_runtime_json(self):
        from aircontrol.diagnostics import build_support_manifest, build_support_readme
        manifest = build_support_manifest("=== AirControl doctor ===\nCamera scan: skipped\n")
        readme = build_support_readme(manifest)
        self.assertIn("runtime-summary.txt отсутствует", readme)
        self.assertIn("откройте doctor.txt", readme)
        self.assertNotIn("runtime.json", readme)

    def test_runtime_summary_without_issues(self):
        from aircontrol.diagnostics import summarize_runtime
        summary = "\n".join(summarize_runtime({
            "fps": 30.0,
            "detect_ms": 25.0,
            "mode": "control",
            "profile": "assistive",
            "safe_input": False,
            "dwell_enabled": True,
            "input_status": "INPUT pynput",
            "hand_detected": True,
        }))
        self.assertIn("No obvious runtime issue", summary)
        self.assertIn("Control path: ON", summary)

    def test_runtime_summary_explains_view_mode(self):
        from aircontrol.diagnostics import summarize_runtime
        summary = "\n".join(summarize_runtime({
            "fps": 30.0,
            "detect_ms": 25.0,
            "mode": "view",
            "profile": "assistive",
            "safe_input": False,
            "dwell_enabled": True,
            "input_status": "INPUT pynput",
            "hand_detected": True,
        }))
        self.assertIn("Control path: OFF (View mode: preview/training only)", summary)
        self.assertIn("View mode is active", summary)

    def test_runtime_summary_flags_input_risk(self):
        from aircontrol.diagnostics import summarize_runtime
        summary = "\n".join(summarize_runtime({
            "fps": 30.0,
            "detect_ms": 25.0,
            "mode": "control",
            "input_status": "INPUT RISK",
            "hand_detected": True,
        }))
        self.assertIn("OS input backend is risky", summary)

    def test_runtime_summary_flags_input_execution_error(self):
        from aircontrol.diagnostics import summarize_runtime
        summary = "\n".join(summarize_runtime({
            "fps": 30.0,
            "detect_ms": 25.0,
            "mode": "control",
            "input_status": "INPUT ERROR",
            "hand_detected": True,
            "last_input_error": "left_click: permission denied",
            "seconds_since_input_error": 0.5,
            "input_error_count": 2,
        }))
        self.assertIn("Control path: ERROR", summary)
        self.assertIn("Last input error: left_click: permission denied (0.5s ago)", summary)
        self.assertIn("Input error count: 2", summary)
        self.assertIn("OS input execution failed recently", summary)

    def test_runtime_summary_marks_expected_capped_fps(self):
        from aircontrol.diagnostics import summarize_runtime
        summary = "\n".join(summarize_runtime({
            "fps": 15.5,
            "detect_ms": 30.0,
            "mode": "control",
            "input_status": "INPUT pynput",
            "hand_detected": True,
            "auto_tuned": True,
            "low_perf_reason": "FPS 15.2",
            "performance": {
                "detect_downscale": 0.4,
                "detect_max_fps": 16,
                "show_landmarks": False,
                "show_particles": False,
                "show_trail": False,
            },
        }))
        self.assertIn("Light mode is active", summary)
        self.assertIn("Auto tune reason: FPS 15.2", summary)
        self.assertNotIn("Low FPS: use the assistive/low preset", summary)

    def test_crash_log_written(self):
        from aircontrol.crash import write_crash_log
        with tempfile.TemporaryDirectory() as d, patch("aircontrol.config.DATA_DIR", d):
            path = write_crash_log(RuntimeError("boom"))
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                text = f.read()
        self.assertIn("AirControl crash report", text)
        self.assertIn("boom", text)

    def test_crash_message_mentions_camera_and_support_bundle(self):
        from aircontrol.crash import build_crash_message
        msg = build_crash_message(
            "/tmp/aircontrol.log",
            RuntimeError("Не удалось открыть камеру (индекс 0)"),
            "/tmp/support.zip",
        )
        self.assertIn("камера не открылась", msg)
        self.assertIn("/tmp/aircontrol.log", msg)
        self.assertIn("/tmp/support.zip", msg)

    def test_startup_support_bundle_written(self):
        from aircontrol.crash import write_startup_support_bundle
        with tempfile.TemporaryDirectory() as d, patch("aircontrol.config.DATA_DIR", d):
            path = write_startup_support_bundle()
            self.assertIsNotNone(path)
            self.assertTrue(os.path.exists(path))
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
        self.assertIn("doctor.txt", names)
        self.assertIn("doctor-summary.txt", names)

    def test_linux_display_detection(self):
        from aircontrol.platform.linux import LinuxBackend
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True):
            self.assertEqual(LinuxBackend().display_server(), "wayland")
        with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True):
            self.assertEqual(LinuxBackend().display_server(), "x11")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(LinuxBackend().display_server(), "headless")


class TestReleaseVerifier(unittest.TestCase):
    def test_speech_flac_names_from_artifact_entries(self):
        from tools.verify_release_artifacts import _speech_flac_names
        names = _speech_flac_names([
            "AirControl/_internal/speech_recognition/flac-win32.exe",
            "AirControl/_internal/speech_recognition/flac-linux-x86_64",
            "AirControl/_internal/other/flac-mac",
            "AirControl/_internal/speech_recognition/README",
        ])
        self.assertEqual(names, {"flac-win32.exe", "flac-linux-x86_64"})

    def test_speech_flac_policy_rejects_macos_flac(self):
        from tools.verify_release_artifacts import _check_speech_flac_policy
        with self.assertRaises(RuntimeError):
            _check_speech_flac_policy([
                "AirControl/_internal/speech_recognition/flac-mac",
            ], "macOS", "AirControl-macOS.zip")

    def test_speech_flac_policy_accepts_windows_converter(self):
        from tools.verify_release_artifacts import _check_speech_flac_policy
        _check_speech_flac_policy([
            "AirControl/_internal/speech_recognition/flac-win32.exe",
        ], "Windows", "AirControl-Windows.zip")


if __name__ == "__main__":
    unittest.main(verbosity=2)
