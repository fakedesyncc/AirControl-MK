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
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import FusionConfig
from ..gestures.engine import FrameGestures


@dataclass
class FusionStatus:
    pose: str = "none"
    listening: bool = False
    dragging: bool = False
    frozen: bool = False
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

    def process(self, fg: FrameGestures, timestamp: Optional[float] = None) -> FusionStatus:
        if timestamp is None:
            timestamp = time.time()
        status = FusionStatus(pose=fg.pose, frozen=fg.frozen, dragging=fg.is_dragging)

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
