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
    """Грубая оценка точки внимания на экране по направлению взгляда.

    Координаты нормализованы [0..1] в системе экрана (как cursor_norm у руки),
    чтобы слияние шло в одном пространстве. valid=False означает «нет лица /
    низкая уверенность» — такая оценка не влияет на курсор.

    ВАЖНО: взгляд по веб-камере без калибровки — грубый сигнал (ошибка легко
    в десятки процентов ширины экрана), поэтому он используется лишь как
    вспомогательное наведение, а не как точный указатель."""

    x: float = 0.5
    y: float = 0.5
    confidence: float = 0.0
    valid: bool = False
    timestamp: float = 0.0

    @property
    def point(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def is_fresh(self, now: float, max_age: float) -> bool:
        """Свежесть оценки: устаревший кадр взгляда не применяем (рывки/лаг)."""
        if not self.valid:
            return False
        if self.timestamp <= 0.0:
            return True
        return (now - self.timestamp) <= max_age


@dataclass
class FusionStatus:
    pose: str = "none"
    listening: bool = False
    dragging: bool = False
    frozen: bool = False
    voice_status: str = ""
    gaze_active: bool = False
    fired_actions: List[str] = field(default_factory=list)


def fuse_cursor_point(
    hand_norm: Optional[Tuple[float, float]],
    gaze_norm: Optional[Tuple[float, float]],
    *,
    mode: str,
    weight: float,
) -> Optional[Tuple[float, float]]:
    """Чистая математика слияния точки наведения (без зависимостей от рантайма).

    mode == "assist": рука ведёт, взгляд лишь грубо смещает к зоне внимания
        (выпуклая смесь с весом weight в пользу руки). Если руки нет — отдаём
        взгляд как запасной указатель.
    mode == "cursor": взгляд сам ведёт курсор, но только когда руки нет в кадре
        (рука всегда точнее и перебивает взгляд).

    Вынесено отдельной функцией, чтобы тестировать без MediaPipe/камеры."""
    if hand_norm is None and gaze_norm is None:
        return None
    if gaze_norm is None:
        return hand_norm
    if hand_norm is None:
        return gaze_norm

    if mode == "cursor":
        # Рука есть → она и ведёт (взгляд игнорируется).
        return hand_norm

    # assist: выпуклая смесь, рука доминирует.
    w = max(0.0, min(1.0, weight))
    fx = (1.0 - w) * hand_norm[0] + w * gaze_norm[0]
    fy = (1.0 - w) * hand_norm[1] + w * gaze_norm[1]
    return (fx, fy)


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
            cursor_fg = self._apply_gaze(fg, gaze, timestamp, status)
            dwell_action = self.cursor.update(cursor_fg, timestamp)
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

    def _apply_gaze(self, fg: FrameGestures, gaze: Optional[GazeResult],
                    now: float, status: FusionStatus) -> FrameGestures:
        """Подмешивает взгляд в точку наведения курсора.

        Возвращает ИСХОДНЫЙ fg без изменений, когда взгляд выключен/отсутствует/
        неуверен/устарел или заморожен жестом — поведение «только рука» при этом
        бит-в-бит совпадает с прежним. Иначе возвращает копию с обновлённым
        cursor_norm (остальные поля fg, нужные для отрисовки/действий, не трогаем)."""
        if not getattr(self.cfg, "gaze_enabled", False) or gaze is None:
            return fg
        if fg.frozen:
            return fg          # «стоп» ладонью важнее любой модальности
        if gaze.confidence < self.cfg.gaze_min_confidence:
            return fg
        if not gaze.is_fresh(now, self.cfg.gaze_max_age):
            return fg

        fused = fuse_cursor_point(
            fg.cursor_norm, gaze.point,
            mode=self.cfg.gaze_mode, weight=self.cfg.gaze_weight,
        )
        if fused is None or fused == fg.cursor_norm:
            return fg

        status.gaze_active = True
        # В режиме "cursor" взгляд может вести курсор и без руки — отмечаем,
        # что точка наведения теперь валидна, чтобы dwell/движение заработали.
        return replace(fg, cursor_norm=fused, hand_detected=True)

    def shutdown(self) -> None:
        """Отпускаем зажатые кнопки при выходе."""
        self.act.release_all()
