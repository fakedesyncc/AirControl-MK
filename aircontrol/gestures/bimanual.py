"""Двуручные (бимануальные) жесты.

Реализован pinch-to-zoom: обе руки делают щипок (большой+указательный) и
разводятся/сводятся — расстояние между руками управляет масштабом. Это
естественное расширение интерфейса: одна рука обычно ведёт курсор, а согласованное
действие двумя руками задаёт отдельную команду без конфликта с одноручными жестами.
"""

import math
from typing import List, Optional, Tuple

from . import features as F
from ..tracking.hand_tracker import HandResult, INDEX_TIP


class BimanualController:
    def __init__(self, cfg):
        self.trigger = cfg.pinch_trigger_ratio
        self.zoom_step = cfg.zoom_step_dist
        self._last_dist: Optional[float] = None
        self._accum = 0.0
        self.engaged = False
        self.points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    def reset(self) -> None:
        self._last_dist = None
        self._accum = 0.0
        self.engaged = False
        self.points = None

    def process(self, hands: List[HandResult]) -> List[str]:
        """Возвращает список действий зума ('zoom_in'/'zoom_out')."""
        if len(hands) < 2:
            self.reset()
            return []

        h1, h2 = hands[0], hands[1]
        p1 = F.pinch_ratio(h1.landmarks, INDEX_TIP)
        p2 = F.pinch_ratio(h2.landmarks, INDEX_TIP)
        self.engaged = p1 < self.trigger and p2 < self.trigger
        if not self.engaged:
            self._last_dist = None
            self.points = None
            return []

        c1 = F.palm_center(h1.landmarks)
        c2 = F.palm_center(h2.landmarks)
        self.points = (c1, c2)
        dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

        actions: List[str] = []
        if self._last_dist is not None:
            self._accum += dist - self._last_dist
            while abs(self._accum) >= self.zoom_step:
                if self._accum > 0:
                    actions.append("zoom_in")
                    self._accum -= self.zoom_step
                else:
                    actions.append("zoom_out")
                    self._accum += self.zoom_step
        self._last_dist = dist
        return actions
