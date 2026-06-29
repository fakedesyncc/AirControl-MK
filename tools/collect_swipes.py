"""Сбор реального датасета свайпов с веб-камеры для обучения/оценки (LOSO).

Записывает размеченные траектории взмахов открытой ладонью от конкретного
испытуемого и сохраняет .npz в формате, который понимают тренер
(`tools/train_swipe_model.py`) и LOSO-оценка (`aircontrol.evaluation.swipe_loso`):

    points  — object-массив траекторий, каждая [(x, y), ...] (норм. координаты);
    labels  — массив имён свайпов (SWIPE_LABELS);
    subject — массив id испытуемого (одинаковый для всей записи сессии).

Источник точек ТОТ ЖЕ, что в рантайме: центр ладони (`palm_center`) во время
позы «открытая ладонь» — ровно те координаты, которые приложение подаёт в
`DynamicGestureRecognizer.update`. Это гарантирует, что собранные траектории
совпадают по распределению с тем, что увидит модель в проде.

ВАЖНО: инструмент требует камеру и mediapipe и в headless/CI окружении не
запускается. Чистая логика (нормализация id, сборка и валидация траектории,
сохранение .npz) вынесена в функции уровня модуля и покрыта юнит-тестами —
их можно гонять без камеры. Импорт камеры/трекера отложен внутрь `capture`,
поэтому модуль импортируется (и py_compile-ится) без mediapipe.

Использование:
    # собрать по 15 свайпов каждого направления у испытуемого s01:
    python tools/collect_swipes.py --subject s01 --per-class 15 --out data/s01.npz

Управление в окне (ладонь раскрыта, делаете взмах, затем жмёте клавишу-метку):
    a = swipe_left   d = swipe_right   w = swipe_up   s = swipe_down
    u = отменить последнюю записанную траекторию
    q / ESC = завершить и сохранить
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Метки берём из рантайма (по файлу — без mediapipe), чтобы не размножать список.
import importlib.util


def _load_swipe_labels() -> Tuple[str, ...]:
    path = os.path.join(_ROOT, "aircontrol", "gestures", "dynamic.py")
    spec = importlib.util.spec_from_file_location("_aircontrol_dynamic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tuple(mod.SWIPE_LABELS)


SWIPE_LABELS: Tuple[str, ...] = _load_swipe_labels()

# Клавиша → метка свайпа (раскладка под привычные стрелки WASD).
KEY_TO_LABEL: Dict[str, str] = {
    "a": "swipe_left",
    "d": "swipe_right",
    "w": "swipe_up",
    "s": "swipe_down",
}


# ===========================================================================
# Чистая логика (тестируется без камеры)
# ===========================================================================

def normalize_subject_id(raw: str) -> str:
    """Нормализует id испытуемого: trim + строка. Пустой id запрещён.

    Анонимность: id — это произвольная метка вроде 's01', НЕ имя/почта."""
    sid = str(raw).strip()
    if not sid:
        raise ValueError("subject id не может быть пустым")
    return sid


def assemble_trajectory(
    raw_points: Sequence[Tuple[float, float]],
    min_points: int = 4,
    min_net: float = 0.05,
) -> List[Tuple[float, float]]:
    """Проверяет и собирает сырую траекторию в список [(x, y), ...] из float-пар.

    Отбраковывает слишком короткие записи и «дрожание на месте» (мелкое суммарное
    смещение). Возвращает очищенную траекторию; кидает ValueError, если запись
    не годится для разметки."""
    pts = [(float(x), float(y)) for x, y in raw_points]
    if len(pts) < min_points:
        raise ValueError(f"слишком короткая траектория: {len(pts)} < {min_points} точек")
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    net = (dx * dx + dy * dy) ** 0.5
    if net < min_net:
        raise ValueError(f"слишком малое смещение: {net:.3f} < {min_net}")
    return pts


def save_dataset(
    path: str,
    points: Sequence[Sequence[Tuple[float, float]]],
    labels: Sequence[str],
    subject: str,
) -> None:
    """Сохраняет датасет в .npz (object 'points', 'labels', 'subject').

    subject пишется ПОЭЛЕМЕНТНО (один и тот же id на каждую траекторию), чтобы
    файл можно было сконкатенировать с записями других испытуемых для LOSO."""
    if len(points) != len(labels):
        raise ValueError("points и labels разной длины")
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    obj_points = np.empty(len(points), dtype=object)
    for i, traj in enumerate(points):
        obj_points[i] = np.asarray(traj, dtype=float)
    np.savez(
        path,
        points=obj_points,
        labels=np.array([str(l) for l in labels]),
        subject=np.array([subject] * len(points)),
    )


def summarize_counts(labels: Sequence[str]) -> Dict[str, int]:
    """Считает число записанных траекторий по каждой метке (для прогресса в HUD)."""
    counts = {label: 0 for label in SWIPE_LABELS}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


# ===========================================================================
# Захват (требует камеру + mediapipe; импорты отложены)
# ===========================================================================

def capture(args: argparse.Namespace) -> int:
    """Интерактивный сбор траекторий с камеры. Возвращает код возврата процесса."""
    # Отложенные импорты: модуль должен импортироваться без mediapipe/cv2.
    import cv2

    from aircontrol.config import AppConfig
    from aircontrol.tracking.camera import Camera
    from aircontrol.tracking.hand_tracker import HandTracker
    from aircontrol.gestures.engine import GestureEngine

    subject = normalize_subject_id(args.subject)
    cfg = AppConfig.load()
    camera = Camera(cfg.camera)
    tracker = HandTracker(cfg.tracking)
    engine = GestureEngine(cfg.gestures)

    points: List[List[Tuple[float, float]]] = []
    labels: List[str] = []
    current: List[Tuple[float, float]] = []

    print(f"Сбор свайпов: субъект={subject!r}, цель={args.per_class}/класс")
    print("  a=left d=right w=up s=down | u=undo | q/ESC=выход")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue
            hands = tracker.detect(frame)
            hand = hands[0] if hands else None
            fg = engine.process(hand)

            # Пока ладонь раскрыта — копим точки центра ладони (как в рантайме).
            if fg.pose == "open_palm" and fg.cursor_norm is not None:
                current.append((float(fg.cursor_norm[0]), float(fg.cursor_norm[1])))
            elif fg.pose != "open_palm" and len(current) < 4:
                current = []  # рука ушла из позы, не успев нарисовать взмах

            _draw_hud(cv2, frame, subject, summarize_counts(labels), len(current))
            cv2.imshow("AirControl — сбор свайпов", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                if labels:
                    removed = labels.pop()
                    points.pop()
                    print(f"  отменено: {removed} (осталось {len(labels)})")
                continue
            ch = chr(key) if 32 <= key < 127 else ""
            if ch in KEY_TO_LABEL:
                label = KEY_TO_LABEL[ch]
                try:
                    traj = assemble_trajectory(current)
                except ValueError as exc:
                    print(f"  пропуск ({label}): {exc}")
                    current = []
                    continue
                points.append(traj)
                labels.append(label)
                current = []
                done = summarize_counts(labels)[label]
                print(f"  + {label}: {done}/{args.per_class}")
    finally:
        tracker.close()
        camera.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    if not labels:
        print("Ничего не записано — файл не сохранён.")
        return 1
    save_dataset(args.out, points, labels, subject)
    print(f"\nСохранено {len(labels)} траекторий → {args.out}")
    for label, n in summarize_counts(labels).items():
        print(f"  {label}: {n}")
    return 0


def _draw_hud(cv2, frame, subject: str, counts: Dict[str, int], buffered: int) -> None:
    """Рисует на кадре прогресс по классам и размер текущего буфера точек."""
    lines = [f"subj={subject}  buffer={buffered}"]
    lines.append("  ".join(f"{k.replace('swipe_', '')}={v}" for k, v in counts.items()))
    y = 24
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        y += 26


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Сбор реального датасета свайпов с камеры")
    p.add_argument("--subject", required=True,
                   help="id испытуемого (анонимная метка, напр. s01)")
    p.add_argument("--out", default="",
                   help="путь к .npz (по умолчанию data/swipes_<subject>.npz)")
    p.add_argument("--per-class", type=int, default=15, dest="per_class",
                   help="ориентир числа траекторий на класс (для HUD-прогресса)")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.out:
        args.out = os.path.join("data", f"swipes_{normalize_subject_id(args.subject)}.npz")
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
