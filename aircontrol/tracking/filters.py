"""Фильтры стабилизации курсора.

Это один из центральных объектов исследования в работе: дрожание руки и шум
детектора лендмарков делают сырой сигнал непригодным для точного наведения.
Здесь реализованы четыре подхода, которые сравниваются экспериментально
(jitter в покое vs задержка при движении):

    NoFilter   — базовая линия (сырые координаты).
    EMAFilter  — экспоненциальное скользящее среднее (исходный метод проекта).
    OneEuroFilter — адаптивный фильтр Casiez, Roussel, Vogel (CHI 2012).
    KalmanFilter  — фильтр Калмана с моделью постоянной скорости.

Все фильтры работают в НОРМАЛИЗОВАННЫХ координатах [0..1], что делает их
параметры независимыми от разрешения камеры и экрана.
"""

import math
from typing import Tuple

import numpy as np


class Filter:
    """Базовый интерфейс фильтра 2D-точки."""

    def filter(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class NoFilter(Filter):
    """Без фильтрации — базовая линия для сравнения."""

    def filter(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        return x, y


class EMAFilter(Filter):
    """Экспоненциальное сглаживание: s_t = a*x_t + (1-a)*s_{t-1}.

    Простой и быстрый, но с фиксированным компромиссом: малое a гасит дрожание,
    но добавляет заметный лаг; большое a — наоборот."""

    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self._sx = None
        self._sy = None

    def filter(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        if self._sx is None:
            self._sx, self._sy = x, y
        else:
            self._sx = self.alpha * x + (1 - self.alpha) * self._sx
            self._sy = self.alpha * y + (1 - self.alpha) * self._sy
        return self._sx, self._sy

    def reset(self) -> None:
        self._sx = self._sy = None


class _LowPass:
    """Скалярный фильтр нижних частот с настраиваемым alpha."""

    def __init__(self):
        self.s = None

    def __call__(self, value: float, alpha: float) -> float:
        if self.s is None:
            self.s = value
        else:
            self.s = alpha * value + (1 - alpha) * self.s
        return self.s

    def reset(self):
        self.s = None


class _OneEuro1D:
    """One-Euro фильтр для одной координаты.

    Идея: частота среза low-pass фильтра адаптируется к скорости сигнала.
    На медленных движениях срез низкий (сильное сглаживание дрожания),
    на быстрых — высокий (минимальная задержка). cutoff = min_cutoff + beta*|dx|.
    """

    def __init__(self, freq: float, min_cutoff: float, beta: float, d_cutoff: float):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_value = None
        self._last_time = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, value: float, timestamp: float) -> float:
        if self._last_time is not None and timestamp > self._last_time:
            self.freq = 1.0 / (timestamp - self._last_time)
        self._last_time = timestamp

        prev = self._last_value if self._last_value is not None else value
        dvalue = (value - prev) * self.freq
        edvalue = self._dx(dvalue, self._alpha(self.d_cutoff, self.freq))

        cutoff = self.min_cutoff + self.beta * abs(edvalue)
        result = self._x(value, self._alpha(cutoff, self.freq))
        self._last_value = value
        return result

    def reset(self):
        self._x.reset()
        self._dx.reset()
        self._last_value = None
        self._last_time = None


class OneEuroFilter(Filter):
    """2D One-Euro фильтр (две независимые координаты).

    Параметры:
        min_cutoff — базовая частота среза (Гц). Меньше → сильнее сглаживание покоя.
        beta       — коэффициент адаптации к скорости. Больше → меньше лага на движении.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0, freq: float = 60.0):
        self._args = (freq, min_cutoff, beta, d_cutoff)
        self._fx = _OneEuro1D(*self._args)
        self._fy = _OneEuro1D(*self._args)

    def filter(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        return self._fx(x, timestamp), self._fy(y, timestamp)

    def reset(self) -> None:
        self._fx = _OneEuro1D(*self._args)
        self._fy = _OneEuro1D(*self._args)


class KalmanFilter(Filter):
    """Фильтр Калмана с моделью постоянной скорости (CV) для 2D-точки.

    Состояние: [x, y, vx, vy]. Хорошо предсказывает плавные траектории,
    но при резких сменах направления возможен небольшой overshoot.
    """

    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-1):
        self.q = process_noise
        self.r = measurement_noise
        self._init()

    def _init(self):
        self.x = np.zeros((4, 1))          # состояние
        self.P = np.eye(4) * 1.0           # ковариация ошибки
        self.H = np.array([[1, 0, 0, 0],   # матрица наблюдения
                           [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2) * self.r
        # Предпосчитанные константы (не аллоцируем их каждый кадр на горячем пути).
        self.Q = np.eye(4) * self.q        # шум процесса — постоянный
        self.I4 = np.eye(4)                 # единичная для коррекции ковариации
        self.F = np.eye(4)                  # шаблон CV-модели; меняем только dt
        self._last_time = None
        self._initialized = False

    def filter(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        dt = 1.0 / 60.0
        if self._last_time is not None:
            dt = max(1e-3, timestamp - self._last_time)
        self._last_time = timestamp

        if not self._initialized:
            self.x[0, 0], self.x[1, 0] = x, y
            self._initialized = True
            return x, y

        # Обновляем только зависящие от dt элементы шаблона F (без пересоздания).
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Прогноз
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Коррекция по измерению
        z = np.array([[x], [y]])
        y_res = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        # Closed-form инверсия 2x2 (S положительно определена при R = r*I, r>0):
        # дешевле и без общего LU-решателя на горячем пути.
        a, b = S[0, 0], S[0, 1]
        c, d = S[1, 0], S[1, 1]
        Sinv = np.array([[d, -b], [-c, a]]) / (a * d - b * c)
        K = self.P @ self.H.T @ Sinv
        self.x = self.x + K @ y_res
        self.P = (self.I4 - K @ self.H) @ self.P

        return float(self.x[0, 0]), float(self.x[1, 0])

    def reset(self) -> None:
        self._init()


def create_filter(filter_cfg) -> Filter:
    """Фабрика фильтра по объекту FilterConfig."""
    t = filter_cfg.type.lower()
    if t == "none":
        return NoFilter()
    if t == "ema":
        return EMAFilter(filter_cfg.ema_alpha)
    if t == "one_euro":
        return OneEuroFilter(
            min_cutoff=filter_cfg.one_euro_min_cutoff,
            beta=filter_cfg.one_euro_beta,
            d_cutoff=filter_cfg.one_euro_d_cutoff,
        )
    if t == "kalman":
        return KalmanFilter(
            process_noise=filter_cfg.kalman_process_noise,
            measurement_noise=filter_cfg.kalman_measurement_noise,
        )
    raise ValueError(f"Неизвестный тип фильтра: {filter_cfg.type}")
