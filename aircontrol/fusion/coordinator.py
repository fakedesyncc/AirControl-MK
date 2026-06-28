"""Координатор слияния модальностей.

Объединяет три потока ввода в одно согласованное поведение:
  * движение руки → позиционирование курсора (через CursorController);
  * жесты         → дискретные действия (через ActionExecutor);
  * голос         → команды/ввод текста (через VoiceRecognizer), активируется
                    жестом «кулак».

Здесь же реализуется разрешение конфликтов между модальностями:
  * пока система слушает голос, курсор не двигается (suppress_cursor_while_listening),
    чтобы случайное дрожание руки не сбивало наведение во время речи;
  * dwell-click и щипковый клик не срабатывают одновременно;
  * накопление дробных шагов скролла между кадрами.
"""

import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

from ..config import FusionConfig
from ..gestures.engine import FrameGestures


@dataclass
class GazeResult:
    """Нормализованная точка взгляда от внешнего eye-tracker backend."""

    point_norm: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    timestamp: Optional[float] = None
    source: str = "unknown"
    valid: bool = True


@dataclass
class FusionStatus:
    pose: str = "none"
    listening: bool = False
    dragging: bool = False
    frozen: bool = False
    gaze_active: bool = False
    gaze_source: str = ""
    cursor_source: str = "hand"
    voice_status: str = ""
    fired_actions: List[str] = field(default_factory=list)


class MultimodalCoordinator:
    def __init__(self, cfg: FusionConfig, action_executor, cursor_controller,
                 voice_recognizer):
        self.cfg = cfg
        self.act = action_executor
        self.cursor = cursor_controller
        self.voice = voice_recognizer
        self._scroll_accum = 0.0

    def process(self, fg: FrameGestures, timestamp: Optional[float] = None,
                gaze: Optional[GazeResult] = None) -> FusionStatus:
        if timestamp is None:
            timestamp = time.time()
        fg, cursor_source, gaze_active = self._apply_gaze(fg, gaze, timestamp)
        status = FusionStatus(
            pose=fg.pose,
            frozen=fg.frozen,
            dragging=fg.is_dragging,
            gaze_active=gaze_active,
            gaze_source=gaze.source if gaze_active and gaze else "",
            cursor_source=cursor_source,
        )

        listening = self.voice.is_listening if self.voice else False

        # 1. Голос: жест «кулак» запускает прослушивание.
        if fg.listening_requested and self.voice and not listening:
            self.voice.start_listening()
            listening = True
        status.listening = listening
        if self.voice:
            status.voice_status = self.voice.last_status

        # 2. Курсор (подавляется во время прослушивания, если включено).
        suppress = self.cfg.suppress_cursor_while_listening and listening
        if not suppress:
            dwell_action = self.cursor.update(fg, timestamp)
            if dwell_action:
                self.act.execute(dwell_action)
                status.fired_actions.append(f"dwell:{dwell_action}")

        # 3. Дискретные события жестов.
        for ev in fg.events:
            if ev.action in ("scroll_mode",):
                continue
            self.act.execute(ev.action)
            status.fired_actions.append(ev.action)

        # 4. Скролл с накоплением дробных шагов.
        if fg.scroll_delta:
            self._scroll_accum += fg.scroll_delta
            if abs(self._scroll_accum) >= 1.0:
                steps = int(self._scroll_accum)
                self.act.scroll(steps)
                self._scroll_accum -= steps
                status.fired_actions.append(f"scroll:{steps}")
        else:
            self._scroll_accum = 0.0

        return status

    def shutdown(self) -> None:
        """Отпускаем зажатые кнопки при выходе."""
        self.act.release_all()

    # --------------------------------------------------------------- gaze

    def _apply_gaze(self, fg: FrameGestures, gaze: Optional[GazeResult],
                    timestamp: float) -> tuple[FrameGestures, str, bool]:
        if not self._usable_gaze(gaze, timestamp):
            return fg, "hand", False

        gaze_point = _clamp_point(gaze.point_norm)
        if self.cfg.gaze_mode == "cursor" or fg.cursor_norm is None:
            return replace(fg, hand_detected=True, cursor_norm=gaze_point), "gaze", True

        hand_x, hand_y = fg.cursor_norm
        weight = max(0.0, min(1.0, self.cfg.gaze_weight))
        mixed = (
            hand_x * (1.0 - weight) + gaze_point[0] * weight,
            hand_y * (1.0 - weight) + gaze_point[1] * weight,
        )
        return replace(fg, cursor_norm=mixed), "hand+gaze", True

    def _usable_gaze(self, gaze: Optional[GazeResult], timestamp: float) -> bool:
        if not self.cfg.gaze_enabled or gaze is None or not gaze.valid:
            return False
        if gaze.point_norm is None or gaze.confidence < self.cfg.gaze_min_confidence:
            return False
        if gaze.timestamp is not None and timestamp - gaze.timestamp > self.cfg.gaze_max_age:
            return False
        return True


def _clamp_point(point: Tuple[float, float]) -> Tuple[float, float]:
    return (
        max(0.0, min(1.0, float(point[0]))),
        max(0.0, min(1.0, float(point[1]))),
    )
