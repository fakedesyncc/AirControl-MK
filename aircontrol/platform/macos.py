"""Платформенный бэкенд macOS (AppleScript / osascript / screencapture)."""

import subprocess
import time
from typing import Optional

from .base import PlatformBackend


class MacOSBackend(PlatformBackend):
    name = "macos"

    def __init__(self) -> None:
        super().__init__()
        self.app_aliases = {
            "сафари": "Safari", "хром": "Google Chrome", "chrome": "Google Chrome",
            "телеграм": "Telegram", "телеграмм": "Telegram",
            "spotify": "Spotify", "спотифай": "Spotify",
            "музыку": "Music", "музыка": "Music",
            "код": "Visual Studio Code", "vscode": "Visual Studio Code",
            "заметки": "Notes", "календарь": "Calendar", "почту": "Mail",
            "finder": "Finder", "финдер": "Finder",
            "терминал": "Terminal", "настройки": "System Settings",
        }

    def _osa(self, script: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["osascript", "-e", script],
                              capture_output=True, text=True, check=check)

    def open_app(self, name: str) -> bool:
        app = self.resolve_app(name)
        try:
            self._osa(f'tell application "{app}"\n activate\n reopen\nend tell')
            self._osa(f'tell application "System Events" to set frontmost of '
                      f'process "{app}" to true', check=False)
            return True
        except subprocess.CalledProcessError:
            try:
                subprocess.run(["open", "-a", app], check=True)
                time.sleep(0.3)
                return True
            except Exception:
                return False

    def close_app(self, name: str) -> bool:
        app = self.resolve_app(name)
        try:
            self._osa(f'tell application "{app}" to quit')
            return True
        except Exception:
            return False

    def minimize_app(self, name: str) -> bool:
        app = self.resolve_app(name)
        try:
            self._osa(f'tell application "{app}" to set miniaturized of every window to true')
            return True
        except Exception:
            try:
                self._osa(f'tell application "System Events" to tell process '
                          f'"{app}" to set visible to false', check=False)
                return True
            except Exception:
                return False

    def change_volume(self, delta_percent: int) -> Optional[int]:
        try:
            r = self._osa("output volume of (get volume settings)")
            vol = int(r.stdout.strip())
            vol = max(0, min(100, vol + delta_percent))
            self._osa(f"set volume output volume {vol}")
            return vol
        except Exception:
            return None

    def set_muted(self, muted: bool) -> bool:
        try:
            self._osa(f"set volume output muted {'true' if muted else 'false'}")
            return True
        except Exception:
            return False

    def screenshot(self, path: str) -> bool:
        # Нативный screencapture: корректный HiDPI и без задержек прав на запись.
        try:
            subprocess.run(["screencapture", "-x", path], check=True)
            return True
        except Exception:
            return super().screenshot(path)
