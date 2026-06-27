"""Извлечение признаков из лендмарков руки.

Два уровня:
  * геометрические помощники (расстояния, размер ладони, «палец вытянут/согнут») —
    используются эвристическим распознавателем;
  * инвариантный вектор признаков для машинного обучения.

Вектор признаков для ML делается инвариантным к:
  * сдвигу     — координаты берутся относительно запястья;
  * масштабу   — делятся на размер ладони (запястье→основание среднего пальца);
  * повороту   — система координат поворачивается так, чтобы ось ладони
                 (запястье→основание среднего) смотрела вверх.
Это позволяет классификатору обобщаться на разные позиции и наклоны руки.
"""

import numpy as np

from ..tracking.hand_tracker import (INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP,
                                      PINKY_TIP, RING_TIP, THUMB_TIP, WRIST)


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Евклидово расстояние в плоскости XY."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def palm_size(landmarks: np.ndarray) -> float:
    """Опорный масштаб руки: запястье → основание указательного пальца."""
    size = distance(landmarks[WRIST], landmarks[INDEX_MCP])
    return max(size, 1e-3)


def palm_center(landmarks: np.ndarray) -> tuple:
    """Центр ладони для позиционирования курсора (среднее запястья и основания
    среднего пальца) — устойчивее, чем кончик одного пальца."""
    cx = (landmarks[WRIST][0] + landmarks[MIDDLE_MCP][0]) / 2.0
    cy = (landmarks[WRIST][1] + landmarks[MIDDLE_MCP][1]) / 2.0
    return cx, cy


def finger_extended(landmarks: np.ndarray, tip_idx: int, ratio: float = 1.6) -> bool:
    """Палец вытянут, если кончик далеко от запястья (в долях размера ладони)."""
    return distance(landmarks[tip_idx], landmarks[WRIST]) > palm_size(landmarks) * ratio


def finger_folded(landmarks: np.ndarray, tip_idx: int, ratio: float = 1.3) -> bool:
    return distance(landmarks[tip_idx], landmarks[WRIST]) < palm_size(landmarks) * ratio


def pinch_ratio(landmarks: np.ndarray, tip_idx: int) -> float:
    """Нормализованное расстояние большой↔палец (для детекции щипка)."""
    return distance(landmarks[THUMB_TIP], landmarks[tip_idx]) / palm_size(landmarks)


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """Инвариантный 42-мерный вектор признаков для ML-классификатора поз."""
    pts = landmarks[:, :2].astype(np.float64).copy()

    # 1. Сдвиг к запястью.
    origin = pts[WRIST].copy()
    pts -= origin

    # 2. Масштаб по размеру ладони.
    scale = palm_size(landmarks)
    pts /= scale

    # 3. Поворот: ось запястье→основание среднего пальца направляем вверх (-Y).
    axis = pts[MIDDLE_MCP]
    angle = np.arctan2(axis[1], axis[0]) + np.pi / 2.0  # довернуть до вертикали
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rot.T

    return pts.flatten().astype(np.float32)  # 21*2 = 42


# Метки поз, которые умеет распознавать система.
POSE_LABELS = ["none", "fist", "open_palm", "peace", "point"]
