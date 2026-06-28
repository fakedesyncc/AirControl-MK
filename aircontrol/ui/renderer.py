"""Отрисовка оверлеев на кадре камеры (OpenCV).

Чистые функции рисования поверх BGR-кадра: скелет руки, индикаторы жестов,
след движения, частицы, HUD-подсказки, метрики FPS, индикатор записи, прогресс
dwell-click. Логика управления здесь отсутствует — только визуализация.
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..gestures.engine import FrameGestures
from ..tracking.hand_tracker import HandResult

# Связи между лендмарками руки (для отрисовки скелета).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # большой
    (0, 5), (5, 6), (6, 7), (7, 8),            # указательный
    (5, 9), (9, 10), (10, 11), (11, 12),       # средний
    (9, 13), (13, 14), (14, 15), (15, 16),     # безымянный
    (13, 17), (17, 18), (18, 19), (19, 20),    # мизинец
    (0, 17),                                   # ладонь
]

POSE_COLORS = {
    "fist": (80, 80, 255), "open_palm": (0, 200, 255),
    "peace": (255, 200, 0), "point": (0, 255, 0), "none": (180, 180, 180),
}


def draw_landmarks(frame, hand: HandResult) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand.landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 120), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1)


class TrailRenderer:
    def __init__(self, max_len: int = 30):
        self.points: List[Tuple[int, int]] = []
        self.max_len = max_len

    def add(self, x: int, y: int) -> None:
        self.points.append((x, y))
        if len(self.points) > self.max_len:
            self.points.pop(0)

    def clear(self) -> None:
        self.points.clear()

    def draw(self, frame) -> None:
        for i in range(1, len(self.points)):
            alpha = i / len(self.points)
            color = (int(255 * alpha), int(100 * alpha), int(255 * alpha))
            cv2.line(frame, self.points[i - 1], self.points[i], color,
                     max(1, int(3 * alpha)))


class ParticleSystem:
    def __init__(self):
        self.particles: List[dict] = []

    def burst(self, x: int, y: int, color=(255, 255, 0), n: int = 20) -> None:
        for _ in range(n):
            self.particles.append({
                "x": float(x), "y": float(y), "color": color, "life": 30,
                "vx": np.random.uniform(-2, 2), "vy": np.random.uniform(-2, 2)})

    def draw(self, frame) -> None:
        alive = []
        for p in self.particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]; p["life"] -= 1
            if p["life"] > 0:
                a = p["life"] / 30.0
                cv2.circle(frame, (int(p["x"]), int(p["y"])), 3,
                           tuple(int(c * a) for c in p["color"]), -1)
                alive.append(p)
        self.particles = alive

    def clear(self) -> None:
        self.particles.clear()


def draw_pose_badge(frame, fg: FrameGestures) -> None:
    if not fg.hand_detected:
        return
    color = POSE_COLORS.get(fg.pose, (180, 180, 180))
    label = f"{fg.pose.upper()}"
    if fg.pose_confidence and fg.pose != "none":
        label += f" {fg.pose_confidence*100:.0f}%"
    cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if fg.frozen:
        cv2.putText(frame, "STOP (frozen)", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def draw_dwell_ring(frame, fg: FrameGestures, progress: float) -> None:
    """Кольцо прогресса dwell-click рядом с центром ладони."""
    if progress <= 0 or fg.cursor_norm is None:
        return
    h, w = frame.shape[:2]
    cx, cy = int(fg.cursor_norm[0] * w), int(fg.cursor_norm[1] * h)
    radius = 26
    angle = int(360 * min(1.0, progress))
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, (0, 255, 255), 4)
    cv2.circle(frame, (cx, cy), radius, (90, 90, 90), 1)


def draw_recording(frame, start_time: float) -> None:
    h, w = frame.shape[:2]
    cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (w - 62, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    sec = int(time.time() - start_time)
    cv2.putText(frame, f"{sec//60:02d}:{sec%60:02d}", (w - 95, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def draw_metrics(frame, fps: float, filter_type: str, recognizer: str) -> None:
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS {fps:4.1f} | filter:{filter_type} | rec:{recognizer}",
                (15, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 120), 1)


def draw_mode_status(frame, mode: str, hand_detected: bool, detect_ms: float) -> None:
    h, w = frame.shape[:2]
    if mode == "control":
        text = "CONTROL: hand detected" if hand_detected else "CONTROL: show your hand"
        color = (0, 255, 0) if hand_detected else (0, 220, 255)
        extra = f"detect {detect_ms:.0f} ms" if detect_ms else "detect waiting"
    else:
        text = "VIEW: hand detected (no input)" if hand_detected else "VIEW: show your hand (no input)"
        color = (0, 220, 255)
        extra = f"detect {detect_ms:.0f} ms | press 2 for control" if detect_ms else "press 2 for control"

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 42), (min(w - 8, 420), 88), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.putText(frame, text, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, extra, (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def draw_runtime_health(frame, lines: List[str]) -> None:
    if not lines:
        return
    h, w = frame.shape[:2]
    x0, y0 = 10, 94
    line_h = 18
    box_h = min(76, 14 + line_h * min(3, len(lines)))
    box_w = min(w - 20, 520)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    for i, line in enumerate(lines[:3]):
        color = (0, 220, 255) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line[:72], (x0 + 8, y0 + 22 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_assistive_status(frame, profile: str, input_status: str, dwell_enabled: bool,
                          last_action: str, last_action_age: Optional[float],
                          dwell_only_mode: bool = False,
                          last_input_error: str = "",
                          last_input_error_age: Optional[float] = None) -> None:
    h, w = frame.shape[:2]
    x0 = max(10, w - 255)
    y0 = 12

    input_ok = input_status.startswith("INPUT ") and input_status not in (
        "INPUT OFF", "INPUT RISK", "INPUT ERROR",
    )
    color = (0, 255, 0) if input_ok else (0, 220, 255)
    lines = [
        f"profile: {profile}",
        f"input: {input_status}",
        f"dwell: {'ON' if dwell_enabled else 'OFF'}",
        f"one: {'ON' if dwell_only_mode else 'OFF'}",
    ]
    if last_action and (last_action_age is None or last_action_age < 3.0):
        lines.append(f"action: {last_action}")
    if last_input_error and (last_input_error_age is None or last_input_error_age < 10.0):
        lines.append(f"error: {last_input_error}")

    overlay = frame.copy()
    height = 18 * len(lines) + 10
    cv2.rectangle(overlay, (x0 - 8, y0 - 6), (w - 8, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line[:32], (x0, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color if i == 1 else (220, 220, 220), 1)


def draw_hud(frame, mode_label: str) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 108), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y0 = h - 88
    lines = [
        "LMB: thumb+index | RMB: thumb+middle | 2x index: dbl-click | 2x middle: mid-btn",
        "Backspace: thumb+ring | Enter: thumb+pinky | 2x ring/pinky: Copy/Paste | Scroll: peace",
        "Stop: open palm | Swipe (open palm move): nav/volume | Zoom: two-hand pinch | Voice: fist",
        f"Screenshot/Record: 3-finger | f=filter g=ML d=dwell | 1=view 2=control | [{mode_label}]",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (10, y0 + i * 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200, 200, 200), 1)


def draw_sensitivity(frame, sensitivity: float) -> None:
    cv2.putText(frame, f"x{sensitivity:.2f}", (frame.shape[1] // 2 - 30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
