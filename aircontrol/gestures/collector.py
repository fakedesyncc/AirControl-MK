"""Сбор размеченного датасета поз руки для обучения ML-распознавателя.

Интерактивный инструмент (окно OpenCV): пользователь показывает позу и нажимает
соответствующую клавишу, чтобы добавить пример. Это и есть «персональная
калибровка» под конкретного пользователя — ключевой шаг для ассистивного
сценария и для эксперимента «эвристика vs ML»."""

import cv2

from .features import POSE_LABELS
from .ml import GestureDataset
from ..config import AppConfig
from ..tracking.camera import Camera
from ..tracking.hand_tracker import HandTracker
from ..ui.renderer import draw_landmarks

# Клавиша → метка позы.
KEY_TO_LABEL = {
    ord("0"): "none", ord("1"): "fist", ord("2"): "open_palm",
    ord("3"): "peace", ord("4"): "point",
}


def collect_dataset(cfg: AppConfig) -> None:
    cam = Camera(cfg.camera)
    tracker = HandTracker(cfg.tracking)
    dataset = GestureDataset.load(cfg.gestures.ml_dataset_path)

    print("=== Сбор датасета жестов ===")
    print("Покажите позу и нажмите цифру, чтобы добавить пример:")
    print("  0=none  1=fist  2=open_palm  3=peace  4=point")
    print("  h=удерживать (быстрый сбор), s=сохранить, q=выход")
    print(f"Текущий датасет: {len(dataset)} примеров {dataset.counts()}")

    hold_label = None
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        hands = tracker.detect(frame)
        hand = hands[0] if hands else None
        if hand is not None:
            draw_landmarks(frame, hand)
            if hold_label is not None:
                dataset.add(hand.landmarks, hold_label)

        counts = dataset.counts()
        y = 30
        cv2.putText(frame, f"Total: {len(dataset)}  {counts}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if hold_label:
            cv2.putText(frame, f"HOLD-CAPTURE: {hold_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("AirControl — Gesture Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            dataset.save(cfg.gestures.ml_dataset_path)
            print(f"✓ Сохранено: {cfg.gestures.ml_dataset_path} ({len(dataset)} примеров)")
        if key == ord("h"):
            hold_label = None  # сброс удержания
        if key in KEY_TO_LABEL and hand is not None:
            label = KEY_TO_LABEL[key]
            dataset.add(hand.landmarks, label)
            hold_label = label  # включаем удержание для быстрого набора
            print(f"+ {label} → {dataset.counts()[label]} примеров")

    dataset.save(cfg.gestures.ml_dataset_path)
    print(f"✓ Финальное сохранение: {len(dataset)} примеров {dataset.counts()}")
    cam.release()
    tracker.close()
    cv2.destroyAllWindows()
