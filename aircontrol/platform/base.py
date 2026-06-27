"""Абстракция платформенно-зависимых операций ОС.

Назначение: вынести всё, что отличается между macOS / Windows / Linux
(запуск приложений, громкость, скриншот), за единый интерфейс. Остальной код
работает только с этим интерфейсом и не знает, на какой ОС он запущен.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class PlatformBackend(ABC):
    """Базовый интерфейс платформенного бэкенда."""

    name = "base"

    def __init__(self) -> None:
        # Псевдонимы приложений: распознанное голосом имя -> системное имя.
        # Базовый набор кросс-платформенный; конкретные бэкенды дополняют его.
        self.app_aliases: Dict[str, str] = {}

    def startup_warnings(self) -> List[str]:
        """Human-readable warnings for fragile platform/runtime combinations."""
        return []

    # ---- Приложения --------------------------------------------------------

    def resolve_app(self, spoken_name: str) -> str:
        """Нечёткое сопоставление произнесённого имени с реальным приложением."""
        key = spoken_name.lower().strip()
        if key in self.app_aliases:
            return self.app_aliases[key]
        for alias, real in self.app_aliases.items():
            if alias in key or key in alias:
                return real
        return spoken_name

    @abstractmethod
    def open_app(self, name: str) -> bool:
        ...

    @abstractmethod
    def close_app(self, name: str) -> bool:
        ...

    @abstractmethod
    def minimize_app(self, name: str) -> bool:
        ...

    # ---- Громкость ---------------------------------------------------------

    @abstractmethod
    def change_volume(self, delta_percent: int) -> Optional[int]:
        """Изменить громкость на delta_percent. Вернуть новый уровень или None."""
        ...

    @abstractmethod
    def set_muted(self, muted: bool) -> bool:
        ...

    # ---- Скриншот ----------------------------------------------------------

    def screenshot(self, path: str) -> bool:
        """Кросс-платформенный скриншот через mss (по умолчанию).

        Платформенные бэкенды могут переопределить (например, macOS использует
        нативный screencapture для лучшего качества/HiDPI)."""
        try:
            import mss  # type: ignore
            import mss.tools  # type: ignore

            with mss.mss() as sct:
                monitor = sct.monitors[0]  # все мониторы
                shot = sct.grab(monitor)
                mss.tools.to_png(shot.rgb, shot.size, output=path)
            return True
        except Exception:
            # Фолбэк на PIL.ImageGrab (работает на Windows/macOS).
            try:
                from PIL import ImageGrab

                img = ImageGrab.grab()
                img.save(path)
                return True
            except Exception as exc:  # pragma: no cover
                print(f"[platform] Не удалось сделать скриншот: {exc}")
                return False

    # ---- Открытие URL ------------------------------------------------------

    def open_url(self, url: str) -> bool:
        import webbrowser

        try:
            return webbrowser.open(url)
        except Exception:
            return False
