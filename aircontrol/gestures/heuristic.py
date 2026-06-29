"""Эвристический распознаватель поз руки.

Пороговые правила по геометрии руки (без обучения). Быстрый и предсказуемый
baseline, с которым в работе сравнивается ML-подход (точность, устойчивость
к разным пользователям и освещению)."""

import numpy as np

from . import features as F
from ..tracking.hand_tracker import (INDEX_MCP, INDEX_TIP, MIDDLE_TIP, PINKY_TIP,
                                      RING_TIP, THUMB_TIP)


class HeuristicPoseClassifier:
    """Возвращает (label, confidence). confidence у эвристики условный (1.0/0.0)."""

    name = "heuristic"

    def predict(self, landmarks: np.ndarray):
        # palm_size считаем один раз и прокидываем во все помощники (вместо ~13
        # повторных np.hypot одного и того же масштаба руки за кадр).
        palm = F.palm_size(landmarks)
        index_up = F.finger_extended(landmarks, INDEX_TIP, 1.5, palm)
        middle_up = F.finger_extended(landmarks, MIDDLE_TIP, 1.5, palm)
        ring_up = F.finger_extended(landmarks, RING_TIP, 1.5, palm)
        pinky_up = F.finger_extended(landmarks, PINKY_TIP, 1.5, palm)

        index_fold = F.finger_folded(landmarks, INDEX_TIP, 1.3, palm)
        middle_fold = F.finger_folded(landmarks, MIDDLE_TIP, 1.3, palm)
        ring_fold = F.finger_folded(landmarks, RING_TIP, 1.3, palm)
        pinky_fold = F.finger_folded(landmarks, PINKY_TIP, 1.3, palm)

        folded = sum([index_fold, middle_fold, ring_fold, pinky_fold])

        # Кулак: все четыре пальца согнуты.
        if folded == 4:
            return "fist", 1.0

        # Открытая ладонь («стоп») — намеренный жест: ВСЕ четыре пальца уверенно
        # вытянуты (порог выше) И отставлен большой палец. Так обычная рука при
        # движении курсора не вызывает случайную заморозку.
        thumb_out = F.distance(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) > palm * 0.45
        strict_open = sum([F.finger_extended(landmarks, t, 1.45, palm)
                           for t in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)])
        if strict_open == 4 and thumb_out:
            return "open_palm", 1.0

        # «Мир»: указательный и средний вытянуты, безымянный и мизинец согнуты.
        if index_up and middle_up and ring_fold and pinky_fold:
            return "peace", 1.0

        # Указатель: только указательный вытянут.
        if index_up and middle_fold and ring_fold and pinky_fold:
            return "point", 1.0

        return "none", 1.0
