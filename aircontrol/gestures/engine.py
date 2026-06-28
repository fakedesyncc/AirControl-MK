"""Движок жестов: превращает лендмарки руки в дискретные события и состояния.

Архитектурное разделение:
  * Дискретные ПОЗЫ (fist/open_palm/peace/point) распознаются сменным
    классификатором — эвристическим ИЛИ обученным ML (предмет сравнения в работе).
  * ЩИПКИ (большой↔палец) детектируются геометрически с гистерезисом
    (trigger/release) и конечными автоматами — это непрерывное измерение,
    которому нужны точные пороги, а не дискретная классификация.

Движок не выполняет действий сам — он возвращает FrameGestures, который
исполняет слой управления (cursor/actions). Это нужно для слияния модальностей.
"""

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import features as F
from .dynamic import DynamicGestureRecognizer
from .heuristic import HeuristicPoseClassifier
from .ml import MLPoseClassifier
from ..config import GestureConfig
from ..tracking.hand_tracker import (HandResult, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP,
                                      PINKY_TIP, RING_TIP, WRIST)


@dataclass
class GestureEvent:
    action: str
    data: dict = field(default_factory=dict)


@dataclass
class FrameGestures:
    hand_detected: bool = False
    cursor_norm: Optional[Tuple[float, float]] = None   # центр ладони [0..1]
    pose: str = "none"
    pose_confidence: float = 0.0
    frozen: bool = False                                 # «стоп», курсор заморожен
    is_dragging: bool = False
    scroll_delta: float = 0.0                            # шаги скролла за кадр
    listening_requested: bool = False                    # кулак → старт голоса
    events: List[GestureEvent] = field(default_factory=list)
    pinch_ratios: Dict[str, float] = field(default_factory=dict)


class _PinchFSM:
    """Конечный автомат щипка с гистерезисом и различением одиночного/двойного.

    Эмитит действия через колбэки. Поддерживает удержание (drag) и повтор."""

    def __init__(self, trigger: float, release: float, double_interval: float):
        self.trigger = trigger
        self.release = release
        self.double_interval = double_interval
        self.pinched = False
        self.last_release_time = 0.0
        self.last_down_time = 0.0
        self.ignore_release = False

    def reset(self):
        self.pinched = False
        self.ignore_release = False


class GestureEngine:
    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self.classifier = self._build_classifier(cfg)

        di = cfg.double_click_interval
        tr, rl = cfg.pinch_trigger_ratio, cfg.pinch_release_ratio
        self.index = _PinchFSM(tr, rl, di)
        self.middle = _PinchFSM(tr, rl, di)
        self.ring = _PinchFSM(tr, rl, di)
        self.pinky = _PinchFSM(tr, rl, di)

        # Скролл (поза «peace»).
        self._last_scroll_y: Optional[float] = None

        # Динамические жесты (свайпы открытой ладонью).
        self.dynamic = DynamicGestureRecognizer(
            min_dist=cfg.swipe_min_dist, max_time=cfg.swipe_max_time,
            cooldown=cfg.swipe_cooldown) if cfg.dynamic_enabled else None

        # Темпоральная стабилизация позы.
        self._pose_hist: deque = deque(maxlen=max(1, cfg.pose_smoothing_window))

        # Backspace-повтор при удержании.
        self._last_backspace = 0.0
        self._backspace_delay = 0.1

        # Комбо-жесты с кулдауном.
        self._last_screenshot = 0.0
        self._last_record = 0.0
        self._screenshot_cd = 1.0
        self._record_cd = 2.0

    def _stabilize(self, pose: str, conf: float) -> Tuple[str, float]:
        """Взвешенное по уверенности голосование за позу по окну кадров."""
        self._pose_hist.append((pose, conf))
        if len(self._pose_hist) < 2:
            return pose, conf
        weights: Dict[str, float] = {}
        for p, c in self._pose_hist:
            weights[p] = weights.get(p, 0.0) + max(c, 0.1)
        best = max(weights, key=weights.get)
        total = sum(weights.values()) or 1.0
        return best, weights[best] / total

    def _build_classifier(self, cfg: GestureConfig):
        if cfg.recognizer == "ml":
            clf = MLPoseClassifier.load(cfg.ml_model_path)
            if clf is not None:
                return clf
            print("[gestures] ML-модель не найдена — откат на эвристику")
        return HeuristicPoseClassifier()

    def set_classifier(self, classifier) -> None:
        """Горячая замена распознавателя (для сравнения в рантайме)."""
        self.classifier = classifier

    # ------------------------------------------------------------------ main

    def process(self, hand: Optional[HandResult]) -> FrameGestures:
        out = FrameGestures()
        if hand is None:
            self._release_all(out)
            self._last_scroll_y = None
            self._pose_hist.clear()
            if self.dynamic:
                self.dynamic.reset()
            return out

        lm = hand.landmarks
        out.hand_detected = True
        out.cursor_norm = F.palm_center(lm)

        # Поза (эвристика или ML) с доверительным гейтом и стабилизацией.
        pose, conf = self.classifier.predict(lm)
        if self.classifier.name == "ml" and conf < self.cfg.ml_min_confidence:
            pose, conf = "none", conf      # низкая уверенность ML → не реагируем
        pose, conf = self._stabilize(pose, conf)
        out.pose, out.pose_confidence = pose, conf

        # Соотношения щипков (для логики и HUD).
        r_index = F.pinch_ratio(lm, INDEX_TIP)
        r_middle = F.pinch_ratio(lm, MIDDLE_TIP)
        r_ring = F.pinch_ratio(lm, RING_TIP)
        r_pinky = F.pinch_ratio(lm, PINKY_TIP)
        out.pinch_ratios = {"index": r_index, "middle": r_middle,
                            "ring": r_ring, "pinky": r_pinky}

        now = time.time()

        if self.cfg.dwell_only_mode:
            out.frozen = pose == "open_palm"
            self._release_all(out)
            self._last_scroll_y = None
            if self.dynamic:
                self.dynamic.reset()
            return out

        # Поза-гейты: кулак/ладонь/мир перехватывают управление.
        if pose == "fist":
            out.listening_requested = True
            self._release_all(out)
            self._last_scroll_y = None
            if self.dynamic:
                self.dynamic.reset()
            return out

        if pose == "open_palm":
            out.frozen = True
            self._release_all(out)
            self._last_scroll_y = None
            # Свайп открытой ладонью — навигация взмахом (курсор и так заморожен).
            if self.dynamic and out.cursor_norm is not None:
                swipe = self.dynamic.update(out.cursor_norm[0], out.cursor_norm[1],
                                            now, active=True)
                if swipe:
                    out.events.append(GestureEvent(swipe))
            return out

        if pose == "peace":
            self._handle_scroll(lm, out)
            self._reset_taps()
            if self.dynamic:
                self.dynamic.reset()
            return out

        self._last_scroll_y = None
        if self.dynamic:
            self.dynamic.reset()

        # Комбо-жесты (скриншот / запись). Если сработали — пропускаем клики.
        if self._handle_combos(r_index, r_middle, r_ring, now, out):
            return out

        # Индивидуальные щипки.
        self._handle_index(r_index, now, out)
        self._handle_middle(r_middle, now, out)
        self._handle_ring(r_ring, now, out)
        self._handle_pinky(r_pinky, now, out)
        out.is_dragging = self.index.pinched and not self.index.ignore_release
        return out

    # --------------------------------------------------------------- helpers

    def _handle_scroll(self, lm: np.ndarray, out: FrameGestures) -> None:
        _, palm_y = F.palm_center(lm)
        if self._last_scroll_y is not None:
            dy = palm_y - self._last_scroll_y
            if abs(dy) >= self.cfg.scroll_threshold:
                out.scroll_delta = dy * self.cfg.scroll_speed * 10.0
        self._last_scroll_y = palm_y
        out.events.append(GestureEvent("scroll_mode"))

    def _handle_combos(self, r_index, r_middle, r_ring, now, out) -> bool:
        tr, rl = self.cfg.pinch_trigger_ratio, self.cfg.pinch_release_ratio
        # Скриншот: большой + указательный + средний.
        if r_index < tr and r_middle < tr and (now - self._last_screenshot) > self._screenshot_cd:
            self._last_screenshot = now
            out.events.append(GestureEvent("screenshot"))
            self._reset_all_fsm()
            return True
        # Запись: большой + указательный + безымянный (средний разогнут).
        if (r_index < tr and r_ring < tr and r_middle > rl
                and (now - self._last_record) > self._record_cd):
            self._last_record = now
            out.events.append(GestureEvent("toggle_record"))
            self._reset_all_fsm()
            return True
        return False

    def _handle_index(self, ratio, now, out) -> None:
        fsm = self.index
        if not fsm.pinched:
            if ratio < fsm.trigger:
                fsm.pinched = True
                if (now - fsm.last_down_time) < fsm.double_interval:
                    out.events.append(GestureEvent("double_click"))
                    fsm.ignore_release = True
                    fsm.last_down_time = 0.0
                else:
                    out.events.append(GestureEvent("left_down"))
                    fsm.ignore_release = False
                    fsm.last_down_time = now
        else:
            if ratio > fsm.release:
                fsm.pinched = False
                if not fsm.ignore_release:
                    out.events.append(GestureEvent("left_up"))
                fsm.last_down_time = now
                fsm.ignore_release = False

    def _handle_middle(self, ratio, now, out) -> None:
        fsm = self.middle
        if not fsm.pinched:
            if ratio < fsm.trigger:
                fsm.pinched = True
                if (now - fsm.last_down_time) < fsm.double_interval:
                    out.events.append(GestureEvent("middle_click"))
                    fsm.last_down_time = 0.0
                else:
                    out.events.append(GestureEvent("right_click"))
                    fsm.last_down_time = now
        else:
            if ratio > fsm.release:
                fsm.pinched = False

    def _handle_ring(self, ratio, now, out) -> None:
        fsm = self.ring
        if not fsm.pinched:
            if ratio < fsm.trigger:
                fsm.pinched = True
                if (now - fsm.last_down_time) < fsm.double_interval:
                    out.events.append(GestureEvent("copy"))
                    fsm.last_down_time = 0.0
                else:
                    out.events.append(GestureEvent("backspace"))
                    self._last_backspace = now
                    fsm.last_down_time = now
        else:
            if ratio > fsm.release:
                fsm.pinched = False
            elif now - self._last_backspace >= self._backspace_delay:
                out.events.append(GestureEvent("backspace"))
                self._last_backspace = now

    def _handle_pinky(self, ratio, now, out) -> None:
        fsm = self.pinky
        if not fsm.pinched:
            if ratio < fsm.trigger:
                fsm.pinched = True
                if (now - fsm.last_down_time) < fsm.double_interval:
                    out.events.append(GestureEvent("paste"))
                    fsm.last_down_time = 0.0
                else:
                    out.events.append(GestureEvent("enter"))
                    fsm.last_down_time = now
        else:
            if ratio > fsm.release:
                fsm.pinched = False

    # --- сброс состояний ---------------------------------------------------

    def _reset_taps(self):
        for fsm in (self.middle, self.ring, self.pinky):
            fsm.pinched = False

    def _reset_all_fsm(self):
        for fsm in (self.index, self.middle, self.ring, self.pinky):
            fsm.reset()

    def _release_all(self, out: FrameGestures) -> None:
        """Безопасно отпускает зажатую ЛКМ при потере руки/смене позы."""
        if self.index.pinched and not self.index.ignore_release:
            out.events.append(GestureEvent("left_up"))
        self._reset_all_fsm()
