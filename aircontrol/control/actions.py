"""Исполнитель действий: переводит имена действий в реальные события ОС.

Низкоуровневые примитивы (клики, клавиши, скролл, ввод текста) реализованы
через pynput; системные операции (скриншот, громкость, приложения) — через
платформенный бэкенд. Это единственное место, где жесты/голос превращаются
в фактическое управление компьютером.
"""

import os
import sys
import time
from typing import Callable, Dict, Optional

from ..platform import get_platform
from .input_backend import (
    Button,
    Key,
    NullKeyboardController,
    NullMouseController,
    create_keyboard_controller,
    create_mouse_controller,
    input_backend_error,
    input_backend_name,
    input_backend_warning,
)


def _safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        fallback = str(message).encode(encoding, errors="backslashreplace").decode(
            encoding, errors="replace"
        )
        print(fallback)


class ActionExecutor:
    def __init__(self, screenshot_dir: str,
                 on_toggle_record: Optional[Callable[[], None]] = None,
                 dry_run: bool = False):
        self.dry_run = dry_run
        self.mouse = NullMouseController() if dry_run else create_mouse_controller()
        self.keyboard = NullKeyboardController() if dry_run else create_keyboard_controller()
        self.platform = get_platform()
        self.mod = Key.cmd if sys.platform == "darwin" else Key.ctrl
        self.screenshot_dir = screenshot_dir
        os.makedirs(screenshot_dir, exist_ok=True)
        self.on_toggle_record = on_toggle_record
        self._left_down = False
        self.last_action = ""
        self.last_action_time = 0.0
        self.last_input_error = ""
        self.last_input_error_time = 0.0
        self.input_error_count = 0
        self._dispatching = False
        self._warn_startup_once()

    def _warn_startup_once(self) -> None:
        if self.dry_run:
            _safe_print("[input] dry-input: real mouse/keyboard events are disabled.")
        err = input_backend_error()
        if err and not self.dry_run:
            _safe_print(f"[input] low-level input is unavailable: {err}")
            _safe_print("[input] gestures can be detected, but cursor/click/key events may not run.")
        warning = input_backend_warning()
        if warning and not self.dry_run:
            _safe_print(f"[input] input warning: {warning}")
        for warning in getattr(self.platform, "startup_warnings", lambda: [])():
            _safe_print(f"[platform:{self.platform.name}] {warning}")

    def input_status(self) -> str:
        if self.dry_run:
            return "DRY INPUT"
        err = input_backend_error()
        if err is not None:
            return "INPUT OFF"
        if self._input_error_recent():
            return "INPUT ERROR"
        return "INPUT RISK" if input_backend_warning() else f"INPUT {input_backend_name()}"

    def set_dry_run(self, enabled: bool) -> None:
        if self.dry_run == enabled:
            return
        self.release_all()
        self.dry_run = enabled
        if enabled:
            self.mouse = NullMouseController()
            self.keyboard = NullKeyboardController()
            _safe_print("[input] dry-input: real mouse/keyboard events are disabled.")
        else:
            self.mouse = create_mouse_controller()
            self.keyboard = create_keyboard_controller()
            err = input_backend_error()
            if err:
                _safe_print(f"[input] low-level input is unavailable: {err}")
            else:
                warning = input_backend_warning()
                if warning:
                    _safe_print(f"[input] input warning: {warning}")

    def _remember(self, action: str) -> None:
        self.last_action = action
        self.last_action_time = time.time()

    def _record_input_error(self, action: str, exc: BaseException) -> None:
        self.last_input_error = f"{action}: {exc}"
        self.last_input_error_time = time.time()
        self.input_error_count += 1

    def _input_error_recent(self, max_age: float = 10.0) -> bool:
        return bool(
            self.last_input_error
            and self.last_input_error_time
            and time.time() - self.last_input_error_time <= max_age
        )

    def _skip_input(self, action: str) -> bool:
        if not self._dispatching:
            self._remember(action)
        return self.dry_run

    # ---- низкоуровневые примитивы -----------------------------------------

    def hotkey(self, *keys) -> None:
        if self._skip_input("hotkey"):
            return
        pressed = []
        try:
            for k in keys:
                self.keyboard.press(k)
                pressed.append(k)
        except Exception as exc:
            self._record_input_error("hotkey", exc)
        finally:
            for k in reversed(pressed):
                try:
                    self.keyboard.release(k)
                except Exception as exc:
                    self._record_input_error("hotkey.release", exc)

    def key_tap(self, key) -> None:
        if self._skip_input("key_tap"):
            return
        pressed = False
        try:
            self.keyboard.press(key)
            pressed = True
        except Exception as exc:
            self._record_input_error("key_tap.press", exc)
        finally:
            if pressed:
                try:
                    self.keyboard.release(key)
                except Exception as exc:
                    self._record_input_error("key_tap.release", exc)

    def type_text(self, text: str) -> None:
        if self._skip_input("type_text"):
            return
        try:
            self.keyboard.type(text)
        except Exception as exc:
            self._record_input_error("type_text", exc)

    def left_down(self) -> None:
        if self._skip_input("left_down"):
            self._left_down = True
            return
        if not self._left_down:
            try:
                self.mouse.press(Button.left)
                self._left_down = True
            except Exception as exc:
                self._record_input_error("left_down", exc)

    def left_up(self) -> None:
        if self._skip_input("left_up"):
            self._left_down = False
            return
        if self._left_down:
            try:
                self.mouse.release(Button.left)
            except Exception as exc:
                self._record_input_error("left_up", exc)
            self._left_down = False

    def left_click(self) -> None:
        if self._skip_input("left_click"):
            return
        try:
            self.mouse.click(Button.left)
        except Exception as exc:
            self._record_input_error("left_click", exc)

    def double_click(self) -> None:
        if self._skip_input("double_click"):
            return
        try:
            self.mouse.click(Button.left, 2)
        except Exception as exc:
            self._record_input_error("double_click", exc)

    def right_click(self) -> None:
        if self._skip_input("right_click"):
            return
        try:
            self.mouse.click(Button.right)
        except Exception as exc:
            self._record_input_error("right_click", exc)

    def middle_click(self) -> None:
        if self._skip_input("middle_click"):
            return
        try:
            self.mouse.click(Button.middle)
        except Exception as exc:
            self._record_input_error("middle_click", exc)

    def scroll(self, steps: int) -> None:
        if self._skip_input(f"scroll:{steps}"):
            return
        if steps:
            try:
                self.mouse.scroll(0, steps)
            except Exception as exc:
                self._record_input_error("scroll", exc)

    def zoom(self, steps: int) -> None:
        """Зум через модификатор + скролл (универсально для браузеров/карт/редакторов)."""
        if self._skip_input(f"zoom:{steps}"):
            return
        pressed = False
        try:
            self.keyboard.press(self.mod)
            pressed = True
            self.mouse.scroll(0, steps)
        except Exception as exc:
            self._record_input_error("zoom", exc)
        finally:
            if pressed:
                try:
                    self.keyboard.release(self.mod)
                except Exception as exc:
                    self._record_input_error("zoom.release", exc)

    def nav_back(self) -> None:
        if sys.platform == "darwin":
            self.hotkey(self.mod, "[")
        else:
            self.hotkey(Key.alt, Key.left)

    def nav_forward(self) -> None:
        if sys.platform == "darwin":
            self.hotkey(self.mod, "]")
        else:
            self.hotkey(Key.alt, Key.right)

    def screenshot(self) -> Optional[str]:
        if self.dry_run:
            self._remember("screenshot")
            return None
        path = os.path.join(self.screenshot_dir,
                            f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png")
        return path if self.platform.screenshot(path) else None

    # ---- диспетчер событий жестов -----------------------------------------

    def execute(self, action: str) -> None:
        self._remember(action)
        if self.dry_run:
            return
        handler = self._dispatch.get(action)
        if handler:
            self._dispatching = True
            try:
                handler(self)
            finally:
                self._dispatching = False

    _dispatch: Dict[str, Callable] = {
        "left_down":    lambda s: s.left_down(),
        "left_up":      lambda s: s.left_up(),
        "left_click":   lambda s: s.left_click(),
        "double_click": lambda s: s.double_click(),
        "right_click":  lambda s: s.right_click(),
        "middle_click": lambda s: s.middle_click(),
        "backspace":    lambda s: s.key_tap(Key.backspace),
        "enter":        lambda s: s.key_tap(Key.enter),
        "copy":         lambda s: s.hotkey(s.mod, "c"),
        "paste":        lambda s: s.hotkey(s.mod, "v"),
        "cut":          lambda s: s.hotkey(s.mod, "x"),
        "screenshot":   lambda s: s.screenshot(),
        "toggle_record": lambda s: (s.on_toggle_record and s.on_toggle_record()),
        "scroll_mode":  lambda s: None,  # фактический скролл идёт через scroll()
        # Динамические жесты (свайпы открытой ладонью).
        "swipe_left":   lambda s: s.nav_back(),
        "swipe_right":  lambda s: s.nav_forward(),
        "swipe_up":     lambda s: s.platform.change_volume(10),
        "swipe_down":   lambda s: s.platform.change_volume(-10),
        # Двуручный pinch-to-zoom.
        "zoom_in":      lambda s: s.zoom(1),
        "zoom_out":     lambda s: s.zoom(-1),
    }

    def release_all(self) -> None:
        """Подстраховка при выходе/смене режима."""
        self.left_up()
