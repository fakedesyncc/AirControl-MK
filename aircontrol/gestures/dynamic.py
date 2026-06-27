"""Распознавание динамических (траекторных) жестов: свайпы и круг.

В отличие от статических поз, динамические жесты определяются формой ТРАЕКТОРИИ
движения руки во времени. Реализован буфер последних позиций с таймстемпами и
детектор быстрых направленных взмахов (свайпов) и кругового движения.

Чтобы не конфликтовать с позиционированием курсора, свайпы распознаются только
когда рука в позе «открытая ладонь» (в этой позе курсор и так заморожен), —
получается естественный жест «навигация взмахом».
"""

import math
from collections import deque
from typing import Optional, Tuple


class DynamicGestureRecognizer:
    def __init__(self, min_dist: float = 0.18, max_time: float = 0.5,
                 cooldown: float = 0.8, straightness: float = 0.7):
        self.min_dist = min_dist          # мин. смещение (норм. координаты)
        self.max_time = max_time          # макс. длительность взмаха, с
        self.cooldown = cooldown          # пауза после срабатывания, с
        self.straightness = straightness  # доля «прямизны» пути (net/path)
        self._buf: deque = deque()        # (t, x, y)
        self._last_fire = float("-inf")   # не блокировать первый свайп

    def reset(self) -> None:
        self._buf.clear()

    def update(self, x: float, y: float, t: float,
               active: bool) -> Optional[str]:
        """Подаёт точку. Возвращает имя свайпа или None.

        active=True — рука в распознающей позе (открытая ладонь)."""
        if not active:
            self._buf.clear()
            return None
        if t - self._last_fire < self.cooldown:
            return None

        self._buf.append((t, x, y))
        # Держим только окно max_time.
        while self._buf and t - self._buf[0][0] > self.max_time:
            self._buf.popleft()
        if len(self._buf) < 4:
            return None

        x0, y0 = self._buf[0][1], self._buf[0][2]
        dx, dy = x - x0, y - y0
        net = math.hypot(dx, dy)
        if net < self.min_dist:
            return None

        # Прямизна: отношение прямого смещения к длине пути.
        path = 0.0
        for i in range(1, len(self._buf)):
            path += math.hypot(self._buf[i][1] - self._buf[i - 1][1],
                               self._buf[i][2] - self._buf[i - 1][2])
        if path > 0 and net / path < self.straightness:
            return None

        self._last_fire = t
        self._buf.clear()
        if abs(dx) > abs(dy):
            return "swipe_right" if dx > 0 else "swipe_left"
        return "swipe_down" if dy > 0 else "swipe_up"
