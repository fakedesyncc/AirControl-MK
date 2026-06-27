"""Crash logging for no-console packaged builds."""

from __future__ import annotations

import os
import time
import traceback
from typing import Optional


def write_crash_log(exc: BaseException) -> str:
    try:
        from .config import DATA_DIR
        base = DATA_DIR
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".aircontrol", "data")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"crash_{time.strftime('%Y%m%d_%H%M%S')}.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write("AirControl crash report\n")
        f.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"error: {exc!r}\n\n")
        tb = exc.__traceback__
        if tb is not None:
            f.write("".join(traceback.format_exception(type(exc), exc, tb)))
        else:
            f.write(f"{type(exc).__name__}: {exc}\n")
    return path


def write_startup_support_bundle() -> Optional[str]:
    """Create a no-camera support bundle after startup failure, best-effort."""
    try:
        from .config import DATA_DIR
        from .diagnostics import save_support_bundle

        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f"startup-support-{time.strftime('%Y%m%d_%H%M%S')}.zip")
        return save_support_bundle(path, scan_camera=False)
    except Exception:
        return None


def build_crash_message(path: str, exc: BaseException,
                        support_path: Optional[str] = None) -> str:
    reason = _friendly_reason(exc)
    parts = [
        "AirControl не смог запуститься.",
        "",
        reason,
        "",
        f"Ошибка: {exc}",
        "",
        f"Лог сохранён:\n{path}",
    ]
    if support_path:
        parts.extend(["", f"Отчёт поддержки сохранён:\n{support_path}"])
    return "\n".join(parts)


def _friendly_reason(exc: BaseException) -> str:
    text = str(exc).lower()
    if "камер" in text or "camera" in text:
        return (
            "Похоже, камера не открылась. Закройте другие приложения с камерой, "
            "проверьте разрешение на камеру и попробуйте безопасную тренировку снова."
        )
    if "hand_landmarker" in text or "model" in text or "модель" in text:
        return "Не найдена или не загрузилась модель руки. Переустановите приложение или сохраните отчёт поддержки."
    if "tk" in text or "display" in text:
        return "Не удалось открыть графическое окно. Проверьте, что запущена обычная desktop-сессия."
    return "Сохраните отчёт поддержки и отправьте его разработчику."


def show_crash_message(path: str, exc: BaseException,
                       support_path: Optional[str] = None) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "AirControl",
            build_crash_message(path, exc, support_path),
        )
        root.destroy()
    except Exception:
        pass
