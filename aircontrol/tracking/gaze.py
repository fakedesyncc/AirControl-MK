"""Грубая оценка взгляда по веб-камере (MediaPipe Face Landmarker, iris).

Идея: модель лица MediaPipe с refine-режимом отдаёт лендмарки радужки (iris).
Положение центра радужки внутри прямоугольника глаза даёт «сырой» вектор взгляда
по каждой оси (горизонталь/вертикаль), который грубо коррелирует с тем, куда
смотрит пользователь. Этот вектор отображается в координаты экрана [0..1]
лёгкой аффинной калибровкой и сглаживается EMA.

ЧЕСТНО О ТОЧНОСТИ. Взгляд по обычной веб-камере без аппаратного айтрекера —
ГРУБЫЙ сигнал. Даже после калибровки ошибка легко достигает десятков процентов
ширины экрана: на неё влияют поворот головы, освещение, очки, расстояние до
камеры и дрожание модели радужки. Поэтому здесь это ВСПОМОГАТЕЛЬНОЕ грубое
наведение (в режиме 'assist' его уточняет рука), а не самостоятельный точный
указатель. Модуль опционален: без модели лица оценщик молча отдаёт valid=False.

Вся математика отображения/калибровки/сглаживания живёт в gaze_math.py БЕЗ
импорта MediaPipe — её можно тестировать отдельно. Тяжёлый MediaPipe здесь
импортируется ЛЕНИВО (только при создании GazeEstimator / в estimate()).
"""

import os
import time
from typing import Optional, Sequence, Tuple

from ..config import GazeConfig
from ..fusion.coordinator import GazeResult
from .gaze_math import EMA2D, GazeCalibration, clamp01, raw_gaze_from_landmarks

__all__ = ["GazeEstimator", "GazeCalibration", "calibration_from_config",
           "write_calibration_to_config"]


def calibration_from_config(cfg: GazeConfig) -> GazeCalibration:
    """Собрать аффинную калибровку из сохранённых в конфиге коэффициентов."""
    return GazeCalibration(ax=cfg.cal_ax, bx=cfg.cal_bx, ay=cfg.cal_ay, by=cfg.cal_by)


def write_calibration_to_config(cal: GazeCalibration, cfg: GazeConfig) -> None:
    """Сохранить коэффициенты калибровки обратно в конфиг (для персистентности)."""
    cfg.cal_ax, cfg.cal_bx = cal.ax, cal.bx
    cfg.cal_ay, cfg.cal_by = cal.ay, cal.by


class GazeEstimator:
    """Оценщик взгляда поверх MediaPipe Face Landmarker.

    Создавать ТОЛЬКО когда cfg.fusion.gaze_enabled=True. Если модель лица
    отсутствует или MediaPipe/модель не загрузились — estimate() молча отдаёт
    GazeResult(valid=False), а ready=False; приложение продолжает работать как
    раньше (только рука). Это держит фичу полностью опциональной."""

    def __init__(self, cfg: GazeConfig):
        self.cfg = cfg
        self.ready = False
        self._landmarker = None
        self._video = False
        self._last_ts_ms = 0
        self._ema = EMA2D(cfg.smoothing_alpha)
        self.calibration = calibration_from_config(cfg)
        self._last_raw: Optional[Tuple[float, float]] = None
        self._init_error = ""
        self._init_landmarker()

    # ---- инициализация -----------------------------------------------------

    def _init_landmarker(self) -> None:
        if not self.cfg.model_path or not os.path.exists(self.cfg.model_path):
            self._init_error = f"face model not found: {self.cfg.model_path}"
            return
        try:
            import cv2  # noqa: F401  (нужен в estimate; проверяем доступность)
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )
            from mediapipe.tasks.python.vision.face_landmarker import (
                FaceLandmarker, FaceLandmarkerOptions,
            )
        except Exception as exc:                       # MediaPipe/cv2 недоступны
            self._init_error = f"mediapipe import failed: {exc}"
            return

        delegate = (BaseOptions.Delegate.GPU if self.cfg.delegate.upper() == "GPU"
                    else BaseOptions.Delegate.CPU)
        base_options = BaseOptions(model_asset_path=self.cfg.model_path, delegate=delegate)
        self._video = (self.cfg.running_mode or "video").lower() == "video"
        running_mode = (VisionTaskRunningMode.VIDEO if self._video
                        else VisionTaskRunningMode.IMAGE)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=self.cfg.min_face_confidence,
            min_face_presence_confidence=self.cfg.min_face_confidence,
            output_face_blendshapes=False,             # не нужно → экономим CPU
        )
        try:
            self._landmarker = FaceLandmarker.create_from_options(options)
        except Exception:
            # Фолбэк на IMAGE-режим (как в hand_tracker), если VIDEO не создаётся.
            try:
                self._video = False
                options.running_mode = VisionTaskRunningMode.IMAGE
                self._landmarker = FaceLandmarker.create_from_options(options)
            except Exception as exc:
                self._init_error = f"face landmarker create failed: {exc}"
                return
        self.ready = True

    @property
    def init_error(self) -> str:
        return self._init_error

    # ---- основной вызов ----------------------------------------------------

    def estimate(self, frame_bgr, timestamp: Optional[float] = None) -> GazeResult:
        """Оценить точку взгляда на BGR-кадре OpenCV. Никогда не бросает —
        при любой проблеме возвращает GazeResult(valid=False)."""
        now = time.time() if timestamp is None else timestamp
        if not self.ready or self._landmarker is None or frame_bgr is None:
            return GazeResult(valid=False, timestamp=now)
        try:
            import cv2
            import numpy as np
            from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # уже C-contiguous
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            if self._video:
                ts = max(self._last_ts_ms + 1, int(time.monotonic() * 1000))
                self._last_ts_ms = ts
                result = self._landmarker.detect_for_video(mp_image, ts)
            else:
                result = self._landmarker.detect(mp_image)
        except Exception:
            return GazeResult(valid=False, timestamp=now)

        if not getattr(result, "face_landmarks", None):
            self._ema.reset()
            self._last_raw = None
            return GazeResult(valid=False, timestamp=now)

        raw = raw_gaze_from_landmarks(result.face_landmarks[0])
        if raw is None:
            return GazeResult(valid=False, timestamp=now)

        raw_x, raw_y, quality = raw
        self._last_raw = (raw_x, raw_y)
        sx, sy = self.calibration.apply(raw_x, raw_y)
        sx, sy = self._ema(sx, sy)
        return GazeResult(x=clamp01(sx), y=clamp01(sy),
                          confidence=quality, valid=True, timestamp=now)

    @property
    def last_raw(self) -> Optional[Tuple[float, float]]:
        """Последний сырой вектор взгляда — нужен мастеру калибровки."""
        return self._last_raw

    def calibrate(self, samples: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
        """Подогнать аффинную калибровку по парам (raw, target_screen[0..1]).

        Возвращает True при успешном обновлении. Параметры можно сохранить в
        конфиг через write_calibration_to_config(estimator.calibration, cfg)."""
        ok = self.calibration.fit(samples)
        if ok:
            self._ema.reset()
        return ok

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        self.ready = False
