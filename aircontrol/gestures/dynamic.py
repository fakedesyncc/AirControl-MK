"""Распознавание динамических (траекторных) жестов: свайпы.

В отличие от статических поз, динамические жесты определяются формой ТРАЕКТОРИИ
движения руки во времени. Реализован буфер последних позиций с таймстемпами и
детектор быстрых направленных взмахов (свайпов).

Чтобы не конфликтовать с позиционированием курсора, свайпы распознаются только
когда рука в позе «открытая ладонь» (в этой позе курсор и так заморожен), —
получается естественный жест «навигация взмахом».

Для продукта без обученной модели используется быстрый эвристический/backend
или template-classifier. Для исследовательской части предусмотрен реальный
инференс LSTM/TCN из .npz-весов: это позволяет подключить обученную временную
модель без PyTorch/TensorFlow в пользовательском приложении.
"""

import math
import os
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np


SWIPE_LABELS = ("swipe_left", "swipe_right", "swipe_up", "swipe_down")


def _resample_points(points: Sequence[Tuple[float, float]], length: int) -> np.ndarray:
    """Resample a 2D trajectory to a fixed-length sequence by path distance."""
    if not points:
        return np.zeros((length, 2), dtype=np.float32)
    arr = np.array(points, dtype=np.float32)
    if len(arr) == 1:
        return np.repeat(arr, length, axis=0)

    steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    dist = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(dist[-1])
    if total <= 1e-8:
        return np.repeat(arr[:1], length, axis=0)
    target = np.linspace(0.0, total, length)
    xs = np.interp(target, dist, arr[:, 0])
    ys = np.interp(target, dist, arr[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _sequence_features(points: Sequence[Tuple[float, float]], length: int) -> np.ndarray:
    """Build a sequence tensor [x, y, dx, dy] from normalized cursor points."""
    seq = _resample_points(points, length)
    centered = seq - seq[0]
    delta = np.diff(centered, axis=0, prepend=centered[:1])
    return np.concatenate([centered, delta], axis=1).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits -= np.max(logits)
    exp = np.exp(logits)
    total = float(np.sum(exp)) or 1.0
    return (exp / total).astype(np.float32)


def _npz_string(data, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    try:
        return str(value.item())
    except Exception:
        return str(value)


class TemplateSwipeClassifier:
    """No-dependency temporal classifier over a fixed-length trajectory.

    It is intentionally simple and deterministic, but unlike the raw heuristic
    it classifies a normalized time sequence rather than only final displacement.
    """

    name = "template"

    def __init__(self, sequence_length: int = 16):
        self.sequence_length = max(4, int(sequence_length))

    def predict(self, points: Sequence[Tuple[float, float]]) -> Tuple[Optional[str], float]:
        seq = _resample_points(points, self.sequence_length)
        dx = float(seq[-1, 0] - seq[0, 0])
        dy = float(seq[-1, 1] - seq[0, 1])
        net = math.hypot(dx, dy)
        if net <= 1e-8:
            return None, 0.0

        path = 0.0
        for i in range(1, len(seq)):
            path += math.hypot(float(seq[i, 0] - seq[i - 1, 0]),
                               float(seq[i, 1] - seq[i - 1, 1]))
        straightness = net / path if path > 0 else 0.0
        dominance = max(abs(dx), abs(dy)) / (abs(dx) + abs(dy) + 1e-8)
        confidence = max(0.0, min(1.0, 0.35 + 0.45 * straightness + 0.20 * dominance))

        if abs(dx) > abs(dy):
            return ("swipe_right" if dx > 0 else "swipe_left"), confidence
        return ("swipe_down" if dy > 0 else "swipe_up"), confidence


class TemporalSwipeSequenceModel:
    """Tiny numpy inference runtime for exported swipe LSTM/TCN models.

    Expected .npz fields:
      * common: backend ("lstm" or "tcn"), labels, sequence_length.
      * tcn: conv_w [kernel, in_ch, out_ch], conv_b [out_ch], W [out_ch, classes], b.
      * lstm: Wi/Wf/Wo/Wg [in_ch, hidden], Ui/Uf/Uo/Ug [hidden, hidden],
        bi/bf/bo/bg [hidden], W [hidden, classes], b.
    """

    def __init__(self, backend: str, labels: List[str], sequence_length: int, arrays):
        self.backend = backend
        self.labels = labels
        self.sequence_length = max(4, int(sequence_length))
        self.arrays = arrays
        self.name = backend

    @classmethod
    def load(cls, path: str, backend: str) -> Optional["TemporalSwipeSequenceModel"]:
        if backend not in ("lstm", "tcn") or not path or not os.path.exists(path):
            return None
        try:
            data = np.load(path, allow_pickle=False)
            file_backend = _npz_string(data, "backend", backend)
            if file_backend != backend:
                return None
            labels = [str(x) for x in data["labels"].tolist()]
            seq_len = int(data["sequence_length"]) if "sequence_length" in data else 16
            required = (
                ("conv_w", "conv_b", "W", "b")
                if backend == "tcn"
                else ("Wi", "Wf", "Wo", "Wg", "Ui", "Uf", "Uo", "Ug",
                      "bi", "bf", "bo", "bg", "W", "b")
            )
            if any(key not in data for key in required):
                return None
            return cls(backend, labels, seq_len, data)
        except Exception as exc:
            print(f"[dynamic] Не удалось загрузить temporal swipe model: {exc}")
            return None

    def predict(self, points: Sequence[Tuple[float, float]]) -> Tuple[Optional[str], float]:
        seq = _sequence_features(points, self.sequence_length)
        logits = self._predict_tcn(seq) if self.backend == "tcn" else self._predict_lstm(seq)
        proba = _softmax(logits)
        idx = int(np.argmax(proba))
        label = self.labels[idx]
        if label not in SWIPE_LABELS:
            return None, float(proba[idx])
        return label, float(proba[idx])

    def _predict_tcn(self, seq: np.ndarray) -> np.ndarray:
        conv_w = self.arrays["conv_w"].astype(np.float32)
        conv_b = self.arrays["conv_b"].astype(np.float32)
        kernel, in_ch, out_ch = conv_w.shape
        if seq.shape[1] != in_ch:
            raise ValueError(f"TCN input channels mismatch: {seq.shape[1]} != {in_ch}")

        pad = kernel // 2
        out = np.zeros((seq.shape[0], out_ch), dtype=np.float32)
        for t in range(seq.shape[0]):
            for k in range(kernel):
                src = t + k - pad
                if 0 <= src < seq.shape[0]:
                    out[t] += seq[src] @ conv_w[k]
        out = np.maximum(out + conv_b, 0.0)
        pooled = out.mean(axis=0)
        return pooled @ self.arrays["W"].astype(np.float32) + self.arrays["b"].astype(np.float32)

    def _predict_lstm(self, seq: np.ndarray) -> np.ndarray:
        hidden = int(self.arrays["bi"].shape[0])
        h = np.zeros(hidden, dtype=np.float32)
        c = np.zeros(hidden, dtype=np.float32)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        for x in seq:
            i = sigmoid(x @ self.arrays["Wi"] + h @ self.arrays["Ui"] + self.arrays["bi"])
            f = sigmoid(x @ self.arrays["Wf"] + h @ self.arrays["Uf"] + self.arrays["bf"])
            o = sigmoid(x @ self.arrays["Wo"] + h @ self.arrays["Uo"] + self.arrays["bo"])
            g = np.tanh(x @ self.arrays["Wg"] + h @ self.arrays["Ug"] + self.arrays["bg"])
            c = f * c + i * g
            h = o * np.tanh(c)
        return h @ self.arrays["W"].astype(np.float32) + self.arrays["b"].astype(np.float32)


class DynamicGestureRecognizer:
    def __init__(self, min_dist: float = 0.18, max_time: float = 0.5,
                 cooldown: float = 0.8, straightness: float = 0.7,
                 backend: str = "heuristic", model_path: str = "",
                 min_confidence: float = 0.65, sequence_length: int = 16):
        self.min_dist = min_dist          # мин. смещение (норм. координаты)
        self.max_time = max_time          # макс. длительность взмаха, с
        self.cooldown = cooldown          # пауза после срабатывания, с
        self.straightness = straightness  # доля «прямизны» пути (net/path)
        self.backend = backend if backend in ("heuristic", "template", "lstm", "tcn") else "heuristic"
        self.min_confidence = min_confidence
        self._buf: deque = deque()        # (t, x, y)
        self._last_fire = float("-inf")   # не блокировать первый свайп
        self._template = TemplateSwipeClassifier(sequence_length=sequence_length)
        self._model = TemporalSwipeSequenceModel.load(model_path, self.backend)
        if self.backend in ("lstm", "tcn") and self._model is None:
            print(f"[dynamic] {self.backend.upper()} модель свайпов не найдена — template fallback")

    def reset(self) -> None:
        self._buf.clear()

    def update(self, x: float, y: float, t: float,
               active: bool) -> Optional[str]:
        """Подаёт точку. Возвращает имя свайпа или None.

        active=True — рука в распознающей позе (открытая ладонь)."""
        if not active:
            self._buf.clear()
            return None
        if t - self._last_fire < self.cooldown:
            return None

        self._buf.append((t, x, y))
        # Держим только окно max_time.
        while self._buf and t - self._buf[0][0] > self.max_time:
            self._buf.popleft()
        if len(self._buf) < 4:
            return None

        x0, y0 = self._buf[0][1], self._buf[0][2]
        dx, dy = x - x0, y - y0
        net = math.hypot(dx, dy)
        if net < self.min_dist:
            return None

        # Прямизна: отношение прямого смещения к длине пути.
        path = 0.0
        for i in range(1, len(self._buf)):
            path += math.hypot(self._buf[i][1] - self._buf[i - 1][1],
                               self._buf[i][2] - self._buf[i - 1][2])
        if path > 0 and net / path < self.straightness:
            return None

        points = [(item[1], item[2]) for item in self._buf]
        if self.backend == "heuristic":
            if abs(dx) > abs(dy):
                label = "swipe_right" if dx > 0 else "swipe_left"
            else:
                label = "swipe_down" if dy > 0 else "swipe_up"
            return self._fire(label, t)

        classifier = self._model or self._template
        label, confidence = classifier.predict(points)
        if label is None or confidence < self.min_confidence:
            return None
        return self._fire(label, t)

    def _fire(self, label: str, timestamp: float) -> str:
        self._last_fire = timestamp
        self._buf.clear()
        return label
