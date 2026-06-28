"""Чистая математика оценки взгляда (БЕЗ MediaPipe/OpenCV).

Вынесено отдельно от gaze.py, чтобы отображение «сырой вектор глаз → экран»,
аффинную калибровку и сглаживание можно было импортировать и тестировать без
тяжёлых зависимостей (импорт MediaPipe долгий и тянет камеру). gaze.py
переэкспортирует эти символы и добавляет к ним сам MediaPipe-оценщик.

Индексы лендмарков — стандартные для MediaPipe Face Mesh / Face Landmarker
(478 точек, режим refine_landmarks): углы глаз, верх/низ век и центры радужек.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# Левый глаз.
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_IRIS = 468            # центр радужки левого глаза
# Правый глаз.
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_IRIS = 473          # центр радужки правого глаза

MIN_IRIS_INDEX = 477       # без iris-лендмарков (refine off) оценка невозможна


def clamp01(v: float) -> float:
    """Зажать значение в [0..1] (точка наведения не выходит за экран)."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def iris_ratio(iris: float, lo: float, hi: float) -> float:
    """Доля положения радужки между краями глаза по одной оси, [0..1].

    lo/hi — координаты краёв глаза (угол/веко). При вырожденном глазе (lo≈hi)
    возвращаем 0.5 (центр) — нейтральная оценка без деления на ноль."""
    span = hi - lo
    if abs(span) < 1e-6:
        return 0.5
    return clamp01((iris - lo) / span)


def eye_openness_quality(left_w: float, left_h: float,
                         right_w: float, right_h: float) -> float:
    """Грубая надёжность оценки по раскрытию глаз (EAR-подобная мера), [0..1].

    Открытый глаз: высота/ширина ≈ 0.25–0.45. Моргание/прищур резко занижает
    отношение → оценке взгляда лучше не доверять. 0.30 принято за «уверенно
    открыт» (quality≈1), ниже линейно падает к 0."""
    def ratio(w: float, h: float) -> float:
        if w < 1e-6:
            return 0.0
        return h / w
    avg = (ratio(left_w, left_h) + ratio(right_w, right_h)) / 2.0
    return clamp01(avg / 0.30)


def fit_affine_1d(raw: Sequence[float], target: Sequence[float]) -> Tuple[float, float]:
    """МНК-подгонка одномерного аффинного отображения target ≈ a*raw + b.

    Возвращает (a, b). При недостатке/вырожденности данных — тождество (1, 0),
    чтобы калибровка деградировала к «как есть», а не ломала наведение."""
    n = len(raw)
    if n < 2 or n != len(target):
        return (1.0, 0.0)
    mean_r = sum(raw) / n
    mean_t = sum(target) / n
    var = sum((r - mean_r) ** 2 for r in raw)
    if var < 1e-9:
        return (1.0, 0.0)
    cov = sum((r - mean_r) * (t - mean_t) for r, t in zip(raw, target))
    a = cov / var
    b = mean_t - a * mean_r
    return (a, b)


@dataclass
class GazeCalibration:
    """Аффинное отображение сырого вектора взгляда в экран [0..1] по осям.

    x = ax*raw_x + bx, y = ay*raw_y + by. По умолчанию ≈ тождество, поэтому
    без калибровки оценка уже работает (грубо), а fit() её уточняет."""

    ax: float = 1.0
    bx: float = 0.0
    ay: float = 1.0
    by: float = 0.0

    def apply(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        x = clamp01(self.ax * raw_x + self.bx)
        y = clamp01(self.ay * raw_y + self.by)
        return (x, y)

    def fit(self, samples: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
        """Подгонка по парам (raw, target), где target — точка экрана [0..1].

        Возвращает True, если калибровка обновлена (хватило данных). Оси
        подгоняются независимо (горизонталь и вертикаль глаза слабо связаны)."""
        if len(samples) < 2:
            return False
        raw_x = [s[0][0] for s in samples]
        raw_y = [s[0][1] for s in samples]
        tgt_x = [clamp01(s[1][0]) for s in samples]
        tgt_y = [clamp01(s[1][1]) for s in samples]
        self.ax, self.bx = fit_affine_1d(raw_x, tgt_x)
        self.ay, self.by = fit_affine_1d(raw_y, tgt_y)
        return True


def raw_gaze_from_landmarks(landmarks) -> Optional[Tuple[float, float, float]]:
    """Сырой вектор взгляда (raw_x, raw_y, quality) из лендмарков лица.

    landmarks — последовательность объектов с полями .x/.y (как у MediaPipe).
    raw_x/raw_y ∈ [0..1] — усреднённое по двум глазам положение радужки внутри
    глаза. quality ∈ [0..1] — надёжность по раскрытию глаз. Возвращает None,
    если iris-лендмарков нет (модель без refine)."""
    if landmarks is None or len(landmarks) <= MIN_IRIS_INDEX:
        return None

    def gx(i: int) -> float:
        return float(landmarks[i].x)

    def gy(i: int) -> float:
        return float(landmarks[i].y)

    # Горизонталь: доля радужки между внешним и внутренним углами глаза.
    lx = iris_ratio(gx(LEFT_IRIS), gx(LEFT_EYE_OUTER), gx(LEFT_EYE_INNER))
    rx = iris_ratio(gx(RIGHT_IRIS), gx(RIGHT_EYE_INNER), gx(RIGHT_EYE_OUTER))
    # Вертикаль: доля радужки между верхним и нижним веком.
    ly = iris_ratio(gy(LEFT_IRIS), gy(LEFT_EYE_TOP), gy(LEFT_EYE_BOTTOM))
    ry = iris_ratio(gy(RIGHT_IRIS), gy(RIGHT_EYE_TOP), gy(RIGHT_EYE_BOTTOM))

    raw_x = (lx + rx) / 2.0
    raw_y = (ly + ry) / 2.0

    left_w = abs(gx(LEFT_EYE_INNER) - gx(LEFT_EYE_OUTER))
    right_w = abs(gx(RIGHT_EYE_OUTER) - gx(RIGHT_EYE_INNER))
    left_h = abs(gy(LEFT_EYE_BOTTOM) - gy(LEFT_EYE_TOP))
    right_h = abs(gy(RIGHT_EYE_BOTTOM) - gy(RIGHT_EYE_TOP))
    quality = eye_openness_quality(left_w, left_h, right_w, right_h)
    return (raw_x, raw_y, quality)


class EMA2D:
    """Экспоненциальное сглаживание 2D-точки (взгляд шумнее руки → гасим сильнее)."""

    def __init__(self, alpha: float):
        self.alpha = max(0.01, min(1.0, alpha))
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None

    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        if self._sx is None:
            self._sx, self._sy = x, y
        else:
            self._sx = self.alpha * x + (1 - self.alpha) * self._sx
            self._sy = self.alpha * y + (1 - self.alpha) * self._sy
        return self._sx, self._sy

    def reset(self) -> None:
        self._sx = self._sy = None
