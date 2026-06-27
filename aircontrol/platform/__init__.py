"""Фабрика платформенного бэкенда — выбирает реализацию по текущей ОС."""

import sys

from .base import PlatformBackend

_backend: PlatformBackend | None = None


def get_platform() -> PlatformBackend:
    """Синглтон платформенного бэкенда для текущей ОС."""
    global _backend
    if _backend is not None:
        return _backend

    if sys.platform == "darwin":
        from .macos import MacOSBackend
        _backend = MacOSBackend()
    elif sys.platform.startswith("win"):
        from .windows import WindowsBackend
        _backend = WindowsBackend()
    else:
        from .linux import LinuxBackend
        _backend = LinuxBackend()
    return _backend


__all__ = ["PlatformBackend", "get_platform"]
