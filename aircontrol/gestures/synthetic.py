"""Генератор синтетических лендмарков поз руки.

Параметрическая модель кисти из 21 точки (раскладка MediaPipe): пальцы строятся
как цепочки суставов с управляемым «загибом» (curl ∈ [0..1]). Для каждой позы
задаётся набор загибов и положение большого пальца, после чего применяется
аугментация: случайные поворот, масштаб, сдвиг и гауссов шум по суставам.

Назначение:
  * мгновенно получить обучаемый размеченный датасет для отладки ML-конвейера
    и демонстрации метрик (accuracy, confusion matrix);
  * аугментация/расширение реального датасета, собранного с камеры (`collect`).

Это НЕ замена реальных данных для итоговой оценки в работе — для эксперимента
«эвристика vs ML» используйте данные реальных пользователей. Синтетика помечается
как таковая и хранится отдельно.
"""

import math
from typing import Dict, List

import numpy as np

from .features import POSE_LABELS
from .ml import GestureDataset

# Локальная раскладка оснований пальцев (frame: y вверх, запястье в (0,0)).
# Индексы соответствуют MediaPipe.
_MCP = {
    5:  (-0.35, 1.00),   # index_mcp
    9:  (-0.08, 1.05),   # middle_mcp
    13: (0.18, 1.00),    # ring_mcp
    17: (0.42, 0.88),    # pinky_mcp
}
# Длины фаланг (pip, dip, tip) для каждого пальца.
_SEG = {5: (0.34, 0.28, 0.22), 9: (0.38, 0.30, 0.24),
        13: (0.34, 0.28, 0.22), 17: (0.28, 0.22, 0.18)}
# Порядок суставов пальца: (pip, dip, tip) индексы для каждого MCP.
_JOINTS = {5: (6, 7, 8), 9: (10, 11, 12), 13: (14, 15, 16), 17: (18, 19, 20)}

# Загибы пальцев [index, middle, ring, pinky] и «приведённость» большого пальца
# (0 — прижат к ладони, 1 — отведён в сторону) для каждой позы.
POSE_PARAMS: Dict[str, dict] = {
    "open_palm": {"curls": [0.05, 0.05, 0.05, 0.05], "thumb": 1.0},
    "fist":      {"curls": [0.95, 0.95, 0.95, 0.95], "thumb": 0.1},
    "peace":     {"curls": [0.05, 0.05, 0.95, 0.95], "thumb": 0.2},
    "point":     {"curls": [0.05, 0.95, 0.95, 0.95], "thumb": 0.15},
    "none":      {"curls": None, "thumb": None},  # генерится случайно
}


def _rotate(vec, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c)


def _build_finger(mcp, segs, curl):
    """Строит точки (pip, dip, tip) пальца, загибая его вперёд по curl."""
    bend = curl * (math.pi / 2.2)        # угол на сустав
    pos = list(mcp)
    direction = (0.0, 1.0)               # вверх вдоль пальца
    pts = []
    for seg_len in segs:
        direction = _rotate(direction, -bend)   # загиб «к ладони»
        pos = [pos[0] + direction[0] * seg_len, pos[1] + direction[1] * seg_len]
        pts.append(tuple(pos))
    return pts


def _build_thumb(spread, rng):
    """Большой палец: spread=1 — отведён вбок, spread=0 — прижат к ладони."""
    # cmc, mcp, ip, tip (индексы 1..4)
    base = (-0.30, 0.30)
    out_dir = (-1.0, 0.4)   # в сторону при отведении
    in_dir = (0.35, 0.9)    # поперёк ладони при прижатии
    dx = out_dir[0] * spread + in_dir[0] * (1 - spread)
    dy = out_dir[1] * spread + in_dir[1] * (1 - spread)
    n = math.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    pts = [base]
    pos = list(base)
    for seg in (0.22, 0.20, 0.18):
        pos = [pos[0] + dx * seg, pos[1] + dy * seg]
        pts.append(tuple(pos))
    return pts  # 4 точки: cmc, mcp, ip, tip


def build_hand(curls, thumb_spread, rng, noise=0.012,
               rot_range=0.5, scale_range=(0.75, 1.25)):
    """Собирает (21,3) лендмарков позы с аугментацией."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (0.0, 0.0, 0.0)              # wrist

    thumb_pts = _build_thumb(thumb_spread, rng)
    for i, p in enumerate(thumb_pts):    # 1..4
        lm[i + 1, :2] = p

    for fi, mcp_idx in enumerate((5, 9, 13, 17)):
        lm[mcp_idx, :2] = _MCP[mcp_idx]
        finger_pts = _build_finger(_MCP[mcp_idx], _SEG[mcp_idx], curls[fi])
        for j, p in zip(_JOINTS[mcp_idx], finger_pts):
            lm[j, :2] = p

    # --- Аугментация: поворот, масштаб, сдвиг, шум ---
    angle = rng.uniform(-rot_range, rot_range)
    scale = rng.uniform(*scale_range)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = lm[:, :2] @ rot.T * scale
    xy += rng.normal(0, noise, xy.shape).astype(np.float32)

    # Перевод в «экранные» координаты: y вниз, центрируем в кадре.
    cx = rng.uniform(0.35, 0.65)
    cy = rng.uniform(0.55, 0.75)
    lm[:, 0] = cx + xy[:, 0] * 0.28
    lm[:, 1] = cy - xy[:, 1] * 0.28
    return lm


def generate_synthetic_dataset(per_pose: int = 300, seed: int = 42,
                               poses: List[str] = None) -> GestureDataset:
    rng = np.random.default_rng(seed)
    poses = poses or [p for p in POSE_LABELS]
    ds = GestureDataset()
    for pose in poses:
        params = POSE_PARAMS[pose]
        for _ in range(per_pose):
            if pose == "none":
                # Неоднозначная/расслабленная поза — негативный класс.
                curls = [float(rng.uniform(0.3, 0.7)) for _ in range(4)]
                thumb = float(rng.uniform(0.3, 0.7))
            else:
                curls = [min(1.0, max(0.0, cu + rng.normal(0, 0.06)))
                         for cu in params["curls"]]
                thumb = min(1.0, max(0.0, params["thumb"] + rng.normal(0, 0.08)))
            ds.add(build_hand(curls, thumb, rng), pose)
    return ds
