"""Safe wrapper around OS input controllers.

On Linux, especially under Wayland or in a headless session, importing or
creating pynput controllers can fail before the app window is shown. This module
keeps that failure local: callers get no-op controllers and a diagnostic string
instead of a hard crash.

Backend order:
  * pynput for regular Windows/macOS/X11 use;
  * xdotool for Linux/X11 when pynput is unavailable;
  * ydotool for Linux uinput setups, including Wayland when ydotoold is ready;
  * no-op controllers with a clear diagnostic string.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Set


@dataclass(frozen=True)
class _FallbackKeys:
    cmd: str = "cmd"
    ctrl: str = "ctrl"
    alt: str = "alt"
    shift: str = "shift"
    left: str = "left"
    right: str = "right"
    tab: str = "tab"
    backspace: str = "backspace"
    enter: str = "enter"
    media_volume_up: str = "media_volume_up"
    media_volume_down: str = "media_volume_down"
    media_volume_mute: str = "media_volume_mute"


@dataclass(frozen=True)
class _FallbackButtons:
    left: str = "left"
    right: str = "right"
    middle: str = "middle"


class NullKeyboardController:
    """No-op keyboard used when no real backend is available."""

    def press(self, key) -> None:
        pass

    def release(self, key) -> None:
        pass

    def type(self, text: str) -> None:
        pass


class NullMouseController:
    """No-op mouse with a position property compatible with pynput."""

    def __init__(self) -> None:
        self._position = (0, 0)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value) -> None:
        self._position = value

    def press(self, button) -> None:
        pass

    def release(self, button) -> None:
        pass

    def click(self, button, count: int = 1) -> None:
        pass

    def scroll(self, dx: int, dy: int) -> None:
        pass


class XDoToolMouseController:
    """Mouse controller backed by xdotool for Linux/X11 sessions."""

    def __init__(self) -> None:
        self._position = (0, 0)

    @property
    def position(self):
        try:
            out = subprocess.run(["xdotool", "getmouselocation", "--shell"],
                                 capture_output=True, text=True, timeout=0.5)
            if out.returncode == 0:
                vals = {}
                for line in out.stdout.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        vals[k] = v
                self._position = (int(vals.get("X", self._position[0])),
                                  int(vals.get("Y", self._position[1])))
        except Exception:
            pass
        return self._position

    @position.setter
    def position(self, value) -> None:
        x, y = int(value[0]), int(value[1])
        self._position = (x, y)
        _run_xdotool(["mousemove", str(x), str(y)])

    def press(self, button) -> None:
        _run_xdotool(["mousedown", _button_number(button)])

    def release(self, button) -> None:
        _run_xdotool(["mouseup", _button_number(button)])

    def click(self, button, count: int = 1) -> None:
        args = ["click"]
        if count > 1:
            args += ["--repeat", str(count), "--delay", "80"]
        args.append(_button_number(button))
        _run_xdotool(args)

    def scroll(self, dx: int, dy: int) -> None:
        button = "4" if dy > 0 else "5"
        for _ in range(abs(int(dy))):
            _run_xdotool(["click", button])


class XDoToolKeyboardController:
    """Keyboard controller backed by xdotool for Linux/X11 sessions."""

    def press(self, key) -> None:
        _run_xdotool(["keydown", _key_name(key)])

    def release(self, key) -> None:
        _run_xdotool(["keyup", _key_name(key)])

    def type(self, text: str) -> None:
        _run_xdotool(["type", "--clearmodifiers", "--", text])


class YDoToolMouseController:
    """Mouse controller backed by ydotool/ydotoold for Linux uinput sessions."""

    def __init__(self) -> None:
        self._position = (0, 0)

    @property
    def position(self):
        # ydotool does not expose a reliable get-position command. CursorController
        # resyncs on first target, so an internal cache is enough here.
        return self._position

    @position.setter
    def position(self, value) -> None:
        x, y = int(value[0]), int(value[1])
        self._position = (x, y)
        _run_ydotool_mousemove_absolute(x, y)

    def press(self, button) -> None:
        _run_ydotool(["click", _ydotool_button(button, down=True)])

    def release(self, button) -> None:
        _run_ydotool(["click", _ydotool_button(button, up=True)])

    def click(self, button, count: int = 1) -> None:
        args = ["click"]
        if count > 1:
            args += ["--repeat", str(count), "--next-delay", "80"]
        args.append(_ydotool_button(button, click=True))
        _run_ydotool(args)

    def scroll(self, dx: int, dy: int) -> None:
        # Current ydotool exposes button clicks and pointer moves, but not a
        # portable wheel primitive. Keep scroll as a no-op instead of sending
        # surprising keyboard navigation under the user's cursor.
        return


class YDoToolKeyboardController:
    """Keyboard controller backed by ydotool/ydotoold for Linux uinput sessions."""

    def press(self, key) -> None:
        code = _linux_key_code(key)
        if code is not None:
            _run_ydotool(["key", f"{code}:1"])

    def release(self, key) -> None:
        code = _linux_key_code(key)
        if code is not None:
            _run_ydotool(["key", f"{code}:0"])

    def type(self, text: str) -> None:
        _run_ydotool(["type", text])


_KeyboardController = None
_MouseController = None
_IMPORT_ERROR: Optional[str] = None
_CREATE_ERROR: Optional[str] = None
_ACTIVE_BACKENDS: Set[str] = set()
_YDOTOOL_READY = False
_YDOTOOL_MOUSEMOVE_STYLE: Optional[str] = None

try:  # pragma: no cover - availability depends on host display server.
    from pynput.keyboard import Controller as _PynputKeyboardController
    from pynput.keyboard import Key as Key
    from pynput.mouse import Button as Button
    from pynput.mouse import Controller as _PynputMouseController

    _KeyboardController = _PynputKeyboardController
    _MouseController = _PynputMouseController
except Exception as exc:  # pragma: no cover - exercised on Linux/CI variants.
    Key = _FallbackKeys()
    Button = _FallbackButtons()
    _IMPORT_ERROR = str(exc)


def create_keyboard_controller():
    global _CREATE_ERROR
    if _prefer_ydotool() and _ydotool_available():
        _ACTIVE_BACKENDS.discard("none")
        _ACTIVE_BACKENDS.add("ydotool")
        return YDoToolKeyboardController()
    if _KeyboardController is not None:
        try:
            controller = _KeyboardController()
            _ACTIVE_BACKENDS.discard("none")
            _ACTIVE_BACKENDS.add("pynput")
            return controller
        except Exception as exc:  # pragma: no cover - display-server dependent.
            _CREATE_ERROR = str(exc)
    if _xdotool_available():
        _ACTIVE_BACKENDS.discard("none")
        _ACTIVE_BACKENDS.add("xdotool")
        return XDoToolKeyboardController()
    if _ydotool_available():
        _ACTIVE_BACKENDS.discard("none")
        _ACTIVE_BACKENDS.add("ydotool")
        return YDoToolKeyboardController()
    _ACTIVE_BACKENDS.add("none")
    return NullKeyboardController()


def create_mouse_controller():
    global _CREATE_ERROR
    if _prefer_ydotool() and _ydotool_available():
        _ACTIVE_BACKENDS.discard("none")
        _ACTIVE_BACKENDS.add("ydotool")
        return YDoToolMouseController()
    if _MouseController is not None:
        try:
            controller = _MouseController()
            _ACTIVE_BACKENDS.discard("none")
            _ACTIVE_BACKENDS.add("pynput")
            return controller
        except Exception as exc:  # pragma: no cover - display-server dependent.
            _CREATE_ERROR = str(exc)
    if _xdotool_available():
        _ACTIVE_BACKENDS.discard("none")
        _ACTIVE_BACKENDS.add("xdotool")
        return XDoToolMouseController()
    if _ydotool_available():
        _ACTIVE_BACKENDS.discard("none")
        _ACTIVE_BACKENDS.add("ydotool")
        return YDoToolMouseController()
    _ACTIVE_BACKENDS.add("none")
    return NullMouseController()


def input_backend_error() -> Optional[str]:
    if _ACTIVE_BACKENDS and "none" not in _ACTIVE_BACKENDS:
        return None
    return _CREATE_ERROR or _IMPORT_ERROR or _fallback_hint()


def input_backend_available() -> bool:
    return input_backend_error() is None


def probe_input_backend(move_mouse: bool = False) -> dict:
    """Return a support-facing low-level input readiness probe.

    The default path is non-invasive: it initializes the backend and reports
    availability/warnings. When ``move_mouse`` is true, the probe attempts a
    one-pixel mouse move and restores the original position. It never clicks or
    sends keyboard events.
    """
    mouse = create_mouse_controller()
    create_keyboard_controller()
    backend = input_backend_name()
    error = input_backend_error()
    warning = input_backend_warning()
    status = "FAIL" if error else ("WARN" if warning else "OK")
    detail = error or warning or "backend initialized"
    mouse_move = None
    mouse_detail = "not requested"

    if error is None and move_mouse:
        mouse_move, mouse_detail = _probe_mouse_move(mouse, backend)
        if mouse_move is False:
            status = "FAIL"
            detail = mouse_detail
        elif mouse_move is None and status == "OK":
            status = "WARN"
            detail = mouse_detail

    return {
        "status": status,
        "backend": backend,
        "available": error is None,
        "warning": warning,
        "error": error,
        "mouse_move_requested": bool(move_mouse),
        "mouse_move": mouse_move,
        "mouse_detail": mouse_detail,
        "detail": detail,
    }


def input_backend_warning() -> Optional[str]:
    """Return a non-fatal warning for backends that may be blocked by the OS."""
    if input_backend_error() is not None:
        return None
    if not sys.platform.startswith("linux"):
        return None

    display = _linux_display_server()
    backend = input_backend_name()
    if display == "wayland" and backend != "ydotool":
        if shutil.which("ydotool"):
            return (
                "Wayland session with non-ydotool input backend; ydotool is "
                "installed, but ydotoold/uinput was not usable."
            )
        return (
            "Wayland session with non-ydotool input backend; global mouse and "
            "keyboard control may be blocked. Use Xorg or configure ydotool/ydotoold."
        )
    if display == "headless":
        return "No graphical display session was detected."
    return None


def input_backend_name() -> str:
    if not _ACTIVE_BACKENDS:
        return "not initialized"
    if "none" in _ACTIVE_BACKENDS:
        return "none"
    if "xdotool" in _ACTIVE_BACKENDS:
        return "xdotool"
    if "ydotool" in _ACTIVE_BACKENDS:
        return "ydotool"
    if "pynput" in _ACTIVE_BACKENDS:
        return "pynput"
    return ",".join(sorted(_ACTIVE_BACKENDS))


def _probe_mouse_move(mouse, backend: str) -> tuple[Optional[bool], str]:
    if backend == "ydotool":
        return None, (
            "ydotoold/uinput is reachable; visible mouse movement was not "
            "attempted because ydotool cannot safely read the current cursor position."
        )
    if isinstance(mouse, NullMouseController):
        return False, "no usable mouse backend"

    try:
        original = tuple(mouse.position)
        if len(original) < 2:
            return False, f"mouse position is not a 2D point: {original!r}"
        x, y = int(original[0]), int(original[1])
        target = (x + 1, y)
        mouse.position = target
        time.sleep(0.03)
        moved = tuple(mouse.position)
        mouse.position = (x, y)
        time.sleep(0.03)
        restored = tuple(mouse.position)
        mx, my = int(moved[0]), int(moved[1])
        rx, ry = int(restored[0]), int(restored[1])
        moved_from_original = abs(mx - x) + abs(my - y) > 0
        restored_near_original = abs(rx - x) <= 10 and abs(ry - y) <= 10
        if moved_from_original and restored_near_original:
            return True, "mouse moved and was restored near the original position"
        return False, (
            f"mouse position did not change/restore as expected: "
            f"before={(x, y)}, after={moved}, restored={restored}"
        )
    except Exception as exc:
        try:
            if "x" in locals() and "y" in locals():
                mouse.position = (x, y)
        except Exception:
            pass
        return False, str(exc)


def _xdotool_available() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if not shutil.which("xdotool"):
        return False
    if not os.environ.get("DISPLAY"):
        return False
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        return False
    return True


def _prefer_ydotool() -> bool:
    return (
        sys.platform.startswith("linux")
        and os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )


def _linux_display_server() -> str:
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session in ("wayland", "x11"):
        return session
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "headless"


def _ydotool_available() -> bool:
    global _YDOTOOL_READY
    if _YDOTOOL_READY:
        return True
    if not sys.platform.startswith("linux"):
        return False
    if not shutil.which("ydotool"):
        return False
    try:
        # 0x00 is documented as a no-op button value. This checks that ydotool
        # can reach ydotoold without moving/clicking anything visible.
        out = subprocess.run(["ydotool", "click", "0x00"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             timeout=0.7)
        _YDOTOOL_READY = out.returncode == 0
        return _YDOTOOL_READY
    except Exception:
        return False


def _run_xdotool(args) -> None:
    try:
        out = subprocess.run(["xdotool", *args], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, timeout=1.0)
    except Exception as exc:
        raise RuntimeError(f"xdotool {' '.join(args)} failed: {exc}") from exc
    if out.returncode != 0:
        raise RuntimeError(f"xdotool {' '.join(args)} failed with exit code {out.returncode}")


def _run_ydotool(args, *, raise_on_error: bool = True) -> bool:
    try:
        out = subprocess.run(["ydotool", *args], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, timeout=1.0)
    except Exception as exc:
        if raise_on_error:
            raise RuntimeError(f"ydotool {' '.join(args)} failed: {exc}") from exc
        return False
    ok = out.returncode == 0
    if not ok and raise_on_error:
        raise RuntimeError(f"ydotool {' '.join(args)} failed with exit code {out.returncode}")
    return ok


def _run_ydotool_mousemove_absolute(x: int, y: int) -> None:
    global _YDOTOOL_MOUSEMOVE_STYLE
    styles = [_YDOTOOL_MOUSEMOVE_STYLE] if _YDOTOOL_MOUSEMOVE_STYLE else ["plain", "xy_flags"]
    for style in styles:
        if style == "plain":
            args = ["mousemove", "--absolute", str(x), str(y)]
        else:
            args = ["mousemove", "--absolute", "-x", str(x), "-y", str(y)]
        if _run_ydotool(args, raise_on_error=False):
            _YDOTOOL_MOUSEMOVE_STYLE = style
            return
    raise RuntimeError("ydotool mousemove failed with all supported argument styles")


def _button_number(button) -> str:
    name = str(button).lower()
    if "middle" in name:
        return "2"
    if "right" in name:
        return "3"
    return "1"


def _ydotool_button(button, *, down: bool = False, up: bool = False,
                   click: bool = False) -> str:
    name = str(button).lower()
    base = 0x00
    if "middle" in name:
        base = 0x02
    elif "right" in name:
        base = 0x01
    mask = 0
    if down:
        mask |= 0x40
    if up:
        mask |= 0x80
    if click:
        mask |= 0xC0
    return f"0x{base | mask:02X}"


def _key_name(key) -> str:
    raw = str(key)
    name = raw.split(".")[-1].lower()
    mapping = {
        "cmd": "Super_L",
        "ctrl": "ctrl",
        "ctrl_l": "ctrl",
        "alt": "alt",
        "alt_l": "alt",
        "shift": "shift",
        "shift_l": "shift",
        "left": "Left",
        "right": "Right",
        "tab": "Tab",
        "backspace": "BackSpace",
        "enter": "Return",
        "media_volume_up": "XF86AudioRaiseVolume",
        "media_volume_down": "XF86AudioLowerVolume",
        "media_volume_mute": "XF86AudioMute",
    }
    if len(raw) == 1:
        return raw
    return mapping.get(name, raw.strip("'"))


def _linux_key_code(key) -> Optional[int]:
    raw = str(key)
    name = raw.split(".")[-1].strip("'").lower()
    if len(name) == 1 and "a" <= name <= "z":
        return {
            "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33,
            "g": 34, "h": 35, "i": 23, "j": 36, "k": 37, "l": 38,
            "m": 50, "n": 49, "o": 24, "p": 25, "q": 16, "r": 19,
            "s": 31, "t": 20, "u": 22, "v": 47, "w": 17, "x": 45,
            "y": 21, "z": 44,
        }[name]
    if len(name) == 1 and "0" <= name <= "9":
        return {"1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
                "6": 7, "7": 8, "8": 9, "9": 10, "0": 11}[name]
    mapping = {
        "cmd": 125,
        "ctrl": 29,
        "ctrl_l": 29,
        "alt": 56,
        "alt_l": 56,
        "shift": 42,
        "shift_l": 42,
        "left": 105,
        "right": 106,
        "tab": 15,
        "backspace": 14,
        "enter": 28,
        "media_volume_up": 115,
        "media_volume_down": 114,
        "media_volume_mute": 113,
    }
    return mapping.get(name)


def _fallback_hint() -> str:
    if sys.platform.startswith("linux"):
        if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            if shutil.which("ydotool"):
                return "ydotool is installed, but ydotoold/uinput is not ready."
            return "Wayland blocks global input; use Xorg or configure ydotool/ydotoold."
        if not shutil.which("xdotool") and not shutil.which("ydotool"):
            return "pynput unavailable and xdotool/ydotool are not installed."
    return "No usable input backend."
