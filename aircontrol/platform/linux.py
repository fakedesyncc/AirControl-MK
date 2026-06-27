"""Платформенный бэкенд Linux."""

import os
import shutil
import subprocess
from typing import List, Optional

from .base import PlatformBackend


class LinuxBackend(PlatformBackend):
    name = "linux"

    def __init__(self) -> None:
        super().__init__()
        self.app_aliases = {
            "хром": "google-chrome", "chrome": "google-chrome",
            "браузер": "firefox", "firefox": "firefox",
            "телеграм": "telegram-desktop", "telegram": "telegram-desktop",
            "spotify": "spotify", "код": "code", "vscode": "code",
            "терминал": "gnome-terminal", "файлы": "nautilus",
            "калькулятор": "gnome-calculator",
        }
        self.desktop_ids = {
            "google-chrome": "google-chrome.desktop",
            "firefox": "firefox.desktop",
            "telegram-desktop": "org.telegram.desktop.desktop",
            "spotify": "spotify.desktop",
            "code": "code.desktop",
            "gnome-terminal": "org.gnome.Terminal.desktop",
            "nautilus": "org.gnome.Nautilus.desktop",
            "gnome-calculator": "org.gnome.Calculator.desktop",
        }
        self._has_wpctl = shutil.which("wpctl") is not None
        self._has_pactl = shutil.which("pactl") is not None
        self._has_amixer = shutil.which("amixer") is not None

    def display_server(self) -> str:
        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session in ("wayland", "x11"):
            return session
        if os.environ.get("WAYLAND_DISPLAY"):
            return "wayland"
        if os.environ.get("DISPLAY"):
            return "x11"
        return "headless"

    def startup_warnings(self) -> List[str]:
        warnings: List[str] = []
        display = self.display_server()
        if display == "wayland":
            if shutil.which("ydotool"):
                warnings.append(
                    "Wayland блокирует многие X11-бэкенды. Если курсор/клики не работают, "
                    "проверьте, что ydotoold запущен и имеет доступ к /dev/uinput."
                )
            else:
                warnings.append(
                    "Wayland часто блокирует глобальное управление курсором/клавиатурой через pynput. "
                    "Для теста управления используйте Xorg-сессию или настройте ydotool/ydotoold."
                )
        elif display == "headless":
            warnings.append("Не найден DISPLAY/WAYLAND_DISPLAY: GUI и управление вводом недоступны.")
        if not (self._has_wpctl or self._has_pactl or self._has_amixer):
            warnings.append("Не найден wpctl/pactl/amixer: жесты громкости будут недоступны.")
        return warnings

    def open_app(self, name: str) -> bool:
        app = self.resolve_app(name)
        if shutil.which(app):
            try:
                subprocess.Popen([app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass

        desktop_id = self.desktop_ids.get(app)
        if desktop_id and shutil.which("gtk-launch"):
            try:
                subprocess.Popen(["gtk-launch", desktop_id],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass

        if os.path.exists(app) or "://" in app:
            try:
                subprocess.Popen(["xdg-open", app],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass

        try:
            subprocess.Popen([app], shell=False,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def close_app(self, name: str) -> bool:
        app = self.resolve_app(name)
        try:
            subprocess.run(["pkill", "-f", app], capture_output=True)
            return True
        except Exception:
            return False

    def minimize_app(self, name: str) -> bool:
        app = self.resolve_app(name)
        if shutil.which("wmctrl") and shutil.which("xdotool"):
            try:
                subprocess.run(["wmctrl", "-a", app], capture_output=True)
                subprocess.run(["xdotool", "getactivewindow", "windowminimize"],
                               capture_output=True)
                return True
            except Exception:
                return False
        return False

    def change_volume(self, delta_percent: int) -> Optional[int]:
        sign = "+" if delta_percent >= 0 else "-"
        amount = abs(delta_percent)
        try:
            if self._has_wpctl:
                suffix = "+" if delta_percent >= 0 else "-"
                subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@",
                                f"{amount}%{suffix}"], check=True, capture_output=True)
            elif self._has_pactl:
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@",
                                f"{sign}{amount}%"], check=True, capture_output=True)
            elif self._has_amixer:
                subprocess.run(["amixer", "set", "Master", f"{amount}%{sign}"],
                               check=True, capture_output=True)
            else:
                return None
            return None
        except Exception:
            return None

    def set_muted(self, muted: bool) -> bool:
        try:
            if self._has_wpctl:
                subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@",
                                "1" if muted else "0"], check=True, capture_output=True)
            elif self._has_pactl:
                subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@",
                                "1" if muted else "0"], check=True, capture_output=True)
            elif self._has_amixer:
                subprocess.run(["amixer", "set", "Master",
                                "mute" if muted else "unmute"], check=True, capture_output=True)
            else:
                return False
            return True
        except Exception:
            return False

    def screenshot(self, path: str) -> bool:
        commands = [
            ("grim", ["grim", path]),
            ("gnome-screenshot", ["gnome-screenshot", "-f", path]),
            ("spectacle", ["spectacle", "-b", "-o", path]),
            ("scrot", ["scrot", path]),
            ("import", ["import", "-window", "root", path]),
        ]
        for tool, cmd in commands:
            if shutil.which(tool):
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    return True
                except Exception:
                    pass
        return super().screenshot(path)
