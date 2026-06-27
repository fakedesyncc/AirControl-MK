"""Обёртка над MediaPipe Hand Landmarker.

Инкапсулирует детали Tasks API и возвращает упрощённый результат: список рук,
каждая — массив из 21 лендмарка (x, y, z в нормализованных координатах [0..1]),
плюс признак «правая/левая». Остальной код не зависит от MediaPipe напрямую.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions

from ..config import TrackingConfig

# Индексы ключевых точек руки в модели MediaPipe (для читаемости в коде жестов).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


@dataclass
class HandResult:
    """Результат детекции одной руки."""
    landmarks: np.ndarray          # shape (21, 3), нормализованные координаты
    handedness: str                # "Left" | "Right"
    score: float


class HandTracker:
    def __init__(self, cfg: TrackingConfig):
        delegate = (BaseOptions.Delegate.GPU if cfg.delegate.upper() == "GPU"
                    else BaseOptions.Delegate.CPU)
        base_options = BaseOptions(model_asset_path=cfg.model_path, delegate=delegate)

        # VIDEO-режим использует трекинг руки между кадрами вместо повторной
        # детекции — это в разы быстрее и даёт более плавный курсор.
        self._video = getattr(cfg, "running_mode", "video").lower() == "video"
        running_mode = (VisionTaskRunningMode.VIDEO if self._video
                        else VisionTaskRunningMode.IMAGE)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        try:
            self.landmarker = HandLandmarker.create_from_options(options)
        except Exception:
            # Фолбэк на IMAGE-режим, если VIDEO недоступен.
            self._video = False
            options.running_mode = VisionTaskRunningMode.IMAGE
            self.landmarker = HandLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def detect(self, frame_bgr) -> List[HandResult]:
        """Детектирует руки на BGR-кадре OpenCV."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        if self._video:
            # Таймстемп должен строго возрастать (мс).
            ts = max(self._last_ts_ms + 1, int(time.monotonic() * 1000))
            self._last_ts_ms = ts
            result = self.landmarker.detect_for_video(mp_image, ts)
        else:
            result = self.landmarker.detect(mp_image)

        hands: List[HandResult] = []
        if result.hand_landmarks:
            for i, lm_list in enumerate(result.hand_landmarks):
                arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
                label, score = "Unknown", 0.0
                if result.handedness and i < len(result.handedness):
                    cat = result.handedness[i][0]
                    label, score = cat.category_name, cat.score
                hands.append(HandResult(landmarks=arr, handedness=label, score=score))
        return hands

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass
