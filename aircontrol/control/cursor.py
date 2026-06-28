"""Контроллер курсора: нормализованная позиция ладони → координаты экрана.

Этапы:
  1. Активная зона (active_region) — растягиваем центральную часть кадра на весь
     экран, чтобы дотягиваться до краёв без выхода руки из поля зрения камеры.
  2. Чувствительность — усиление амплитуды относительно центра.
  3. Фильтр стабилизации (см. tracking.filters) — в нормализованном пространстве.
  4. Перевод в пиксели экрана и установка абсолютной позиции курсора.

Дополнительно реализован dwell-click — клик по удержанию курсора на месте,
ключевая ассистивная функция для пользователей, которым трудно делать щипок.
"""

import time
from typing import Optional, Tuple

from ..config import CursorConfig, FilterConfig
from ..gestures.engine import FrameGestures
from ..tracking.filters import create_filter


class CursorController:
    def __init__(self, cursor_cfg: CursorConfig, filter_cfg: FilterConfig,
                 mouse, screen_size: Tuple[int, int]):
        self.cfg = cursor_cfg
        self.filter = create_filter(filter_cfg)
        self.mouse = mouse
        self.screen_w, self.screen_h = screen_size

        self._dwell_anchor: Optional[Tuple[int, int]] = None
        self._dwell_start = 0.0
        self._dwell_fired = False
        self._dwell_cooldown_until = 0.0
        self.last_screen_pos: Tuple[int, int] = (0, 0)
        self.dwell_progress = 0.0   # [0..1] для отрисовки прогресса в UI

        # Разделение «цель» (ставит поток детекции) и «движение» (воркер мыши).
        try:
            pos = self.mouse.position
            self._current = [float(pos[0]), float(pos[1])]
        except Exception:
            self._current = [self.screen_w / 2.0, self.screen_h / 2.0]
        self.target: Optional[Tuple[int, int]] = None
        self._resync = False        # при повторном захвате руки — прыжок к цели
        self.easing = getattr(cursor_cfg, "worker_easing", 0.4)

    def set_filter(self, filter_cfg: FilterConfig) -> None:
        """Смена фильтра в рантайме (используется при сравнении в работе)."""
        self.filter = create_filter(filter_cfg)

    def set_screen_size(self, w: int, h: int) -> None:
        self.screen_w, self.screen_h = w, h

    def set_mouse(self, mouse) -> None:
        """Swap the low-level mouse backend after Safe ON/OFF changes."""
        self.mouse = mouse
        try:
            pos = self.mouse.position
            self._current = [float(pos[0]), float(pos[1])]
        except Exception:
            self._current = [float(self.last_screen_pos[0]), float(self.last_screen_pos[1])]
        self._resync = True

    # ---- отображение нормализованных координат в экранные ------------------

    def map_to_screen(self, nx: float, ny: float) -> Tuple[int, int]:
        r = max(0.1, min(1.0, self.cfg.active_region))
        lo = 0.5 - r / 2.0
        mx = (nx - lo) / r
        my = (ny - lo) / r

        mx = 0.5 + (mx - 0.5) * self.cfg.sensitivity
        my = 0.5 + (my - 0.5) * self.cfg.sensitivity

        if self.cfg.invert_x:
            mx = 1.0 - mx
        if self.cfg.invert_y:
            my = 1.0 - my

        mx = max(0.0, min(1.0, mx))
        my = max(0.0, min(1.0, my))

        m = self.cfg.edge_margin
        sx = int(round(mx * (self.screen_w - 1)))
        sy = int(round(my * (self.screen_h - 1)))
        sx = max(m, min(self.screen_w - 1 - m, sx))
        sy = max(m, min(self.screen_h - 1 - m, sy))
        return sx, sy

    # ---- основной апдейт ---------------------------------------------------

    def update(self, fg: FrameGestures, timestamp: float) -> Optional[str]:
        """Вычисляет ЦЕЛЬ курсора (фильтр + dwell). Само движение мыши делает
        высокочастотный воркер через step() — это разъединяет плавность курсора
        и частоту детекции. Возвращает 'left_click' при срабатывании dwell."""
        if not fg.hand_detected or fg.cursor_norm is None or fg.frozen:
            self.target = None          # воркер не двигает мышь
            self._reset_dwell()
            return None

        nx, ny = fg.cursor_norm
        fx, fy = self.filter.filter(nx, ny, timestamp)
        sx, sy = self.map_to_screen(fx, fy)
        if self.target is None:
            self._resync = True         # рука только что вернулась — прыжок к цели
        self.target = (sx, sy)
        self.last_screen_pos = (sx, sy)
        return self._update_dwell(sx, sy, timestamp, fg)

    def step(self) -> None:
        """Вызывается высокочастотным воркером: плавно ведёт курсор к цели."""
        target = self.target
        if target is None:
            return
        try:
            if self._resync:
                self._current = [float(target[0]), float(target[1])]
                self._resync = False
            else:
                self._current[0] += (target[0] - self._current[0]) * self.easing
                self._current[1] += (target[1] - self._current[1]) * self.easing
            self.mouse.position = (int(round(self._current[0])), int(round(self._current[1])))
        except Exception:
            pass

    # ---- dwell-click -------------------------------------------------------

    def _update_dwell(self, sx: int, sy: int, now: float,
                      fg: FrameGestures) -> Optional[str]:
        # Dwell не активен при перетаскивании или в режиме скролла.
        if not self.cfg.dwell_enabled or fg.is_dragging or fg.scroll_delta:
            self._reset_dwell()
            return None

        if now < self._dwell_cooldown_until:
            self._reset_dwell()
            return None

        if self._dwell_anchor is None:
            self._dwell_anchor = (sx, sy)
            self._dwell_start = now
            self._dwell_fired = False
            self.dwell_progress = 0.0
            return None

        ax, ay = self._dwell_anchor
        moved = ((sx - ax) ** 2 + (sy - ay) ** 2) ** 0.5
        if moved > self.cfg.dwell_radius:
            # Курсор «уехал» — перезапускаем зону удержания.
            self._dwell_anchor = (sx, sy)
            self._dwell_start = now
            self._dwell_fired = False
            self.dwell_progress = 0.0
            return None

        elapsed = now - self._dwell_start
        self.dwell_progress = min(1.0, elapsed / self.cfg.dwell_time)
        if elapsed >= self.cfg.dwell_time and not self._dwell_fired:
            self._dwell_fired = True
            self.dwell_progress = 1.0
            cooldown = max(0.0, getattr(self.cfg, "dwell_cooldown", 0.0))
            self._dwell_cooldown_until = now + cooldown
            return "left_click"
        return None

    def _reset_dwell(self) -> None:
        self._dwell_anchor = None
        self._dwell_fired = False
        self.dwell_progress = 0.0
