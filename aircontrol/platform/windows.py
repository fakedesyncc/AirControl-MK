"""Платформенный бэкенд Windows.

Запуск приложений — через `start`, громкость — через pycaw (если установлен)
или нажатия мультимедийных клавиш как фолбэк. Скриншот наследуется из base (mss).
"""

import subprocess
from typing import Optional

from .base import PlatformBackend


class WindowsBackend(PlatformBackend):
    name = "windows"

    def __init__(self) -> None:
        super().__init__()
        self.app_aliases = {
            "хром": "chrome", "chrome": "chrome",
            "браузер": "msedge", "edge": "msedge",
            "телеграм": "telegram", "telegram": "telegram",
            "spotify": "spotify", "спотифай": "spotify",
            "код": "code", "vscode": "code",
            "блокнот": "notepad", "заметки": "notepad",
            "калькулятор": "calc", "проводник": "explorer",
            "терминал": "wt", "настройки": "ms-settings:",
        }
        self._volume = self._init_pycaw()

    def _init_pycaw(self):
        try:
            from ctypes import POINTER, cast  # noqa: F401
            from comtypes import CLSCTX_ALL  # type: ignore
            from pycaw.pycaw import (AudioUtilities, IAudioEndpointVolume)  # type: ignore

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(interface, POINTER(IAudioEndpointVolume))
        except Exception:
            return None

    def open_app(self, name: str) -> bool:
        app = self.resolve_app(name)
        try:
            subprocess.Popen(["cmd", "/c", "start", "", app], shell=False)
            return True
        except Exception:
            return False

    def close_app(self, name: str) -> bool:
        app = self.resolve_app(name)
        exe = app if app.lower().endswith(".exe") else f"{app}.exe"
        try:
            subprocess.run(["taskkill", "/IM", exe, "/F"], capture_output=True)
            return True
        except Exception:
            return False

    def minimize_app(self, name: str) -> bool:
        # Универсальная минимизация активного окна (Win+Down доступна не всегда).
        try:
            from ..control.input_backend import Key, create_keyboard_controller

            kb = create_keyboard_controller()
            kb.press(Key.cmd); kb.press("d"); kb.release("d"); kb.release(Key.cmd)
            return True
        except Exception:
            return False

    def change_volume(self, delta_percent: int) -> Optional[int]:
        if self._volume is not None:
            try:
                cur = self._volume.GetMasterVolumeLevelScalar()
                new = max(0.0, min(1.0, cur + delta_percent / 100.0))
                self._volume.SetMasterVolumeLevelScalar(new, None)
                return int(round(new * 100))
            except Exception:
                pass
        # Фолбэк — мультимедийные клавиши.
        try:
            from ..control.input_backend import Key, create_keyboard_controller

            kb = create_keyboard_controller()
            key = Key.media_volume_up if delta_percent > 0 else Key.media_volume_down
            for _ in range(max(1, abs(delta_percent) // 2)):
                kb.press(key); kb.release(key)
            return None
        except Exception:
            return None

    def set_muted(self, muted: bool) -> bool:
        if self._volume is not None:
            try:
                self._volume.SetMute(1 if muted else 0, None)
                return True
            except Exception:
                pass
        try:
            from ..control.input_backend import Key, create_keyboard_controller

            kb = create_keyboard_controller()
            kb.press(Key.media_volume_mute); kb.release(Key.media_volume_mute)
            return True
        except Exception:
            return False
