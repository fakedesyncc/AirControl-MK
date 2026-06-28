"""Захват видео с камеры с автоопределением и восстановлением после сбоев."""

import sys
import time

import cv2

from ..config import CameraConfig


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        index = cfg.index if cfg.index is not None else self._auto_index()
        self.cap = self._open_capture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру (индекс {index})")
        self.index = index
        self._last_reopen = 0.0

    def _backend_api(self) -> int:
        backend = (getattr(self.cfg, "backend", "auto") or "auto").lower()
        if backend == "any":
            return cv2.CAP_ANY
        if backend == "v4l2":
            return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        if backend == "auto" and sys.platform.startswith("linux"):
            return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        return cv2.CAP_ANY

    def _open_capture(self, index: int):
        api = self._backend_api()
        cap = cv2.VideoCapture(index, api) if api != cv2.CAP_ANY else cv2.VideoCapture(index)
        self._configure_capture(cap)
        return cap

    def _configure_capture(self, cap) -> None:
        if cap is None:
            return
        fourcc = getattr(self.cfg, "fourcc", "")
        if fourcc and len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.target_fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(getattr(self.cfg, "buffer_size", 1))))

    def _auto_index(self) -> int:
        """Поиск встроенной камеры. На macOS встроенная обычно имеет индекс 1
        при подключённой внешней (Continuity Camera/телефон)."""
        limit = max(1, int(getattr(self.cfg, "scan_indices", 4)))
        order = list(range(limit))
        if self.cfg.prefer_builtin and sys.platform == "darwin" and 1 in order:
            order = [1, 0] + [i for i in order if i not in (0, 1)]
        for idx in order:
            cap = self._open_capture(idx)
            if cap.isOpened():
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None and frame.size > 0:
                    print(f"[camera] Используется камера с индексом {idx}")
                    return idx
            cap.release()
        print("[camera] Камера не найдена — пробуем индекс 0")
        return 0

    def read(self):
        """Возвращает (ok, frame). Кадр зеркалится по горизонтали при необходимости."""
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            ok, frame = self._try_reopen_and_read()
        if ok and self.cfg.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return ok, frame

    def _try_reopen_and_read(self):
        now = time.monotonic()
        if now - self._last_reopen < getattr(self.cfg, "reopen_delay", 0.7):
            return False, None
        self._last_reopen = now
        print(f"[camera] Пустой кадр — переоткрываю камеру {self.index}")
        try:
            self.release()
            self.cap = self._open_capture(self.index)
            if self.cap.isOpened():
                return self.cap.read()
        except Exception as exc:
            print(f"[camera] Не удалось переоткрыть камеру: {exc}")
        return False, None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
