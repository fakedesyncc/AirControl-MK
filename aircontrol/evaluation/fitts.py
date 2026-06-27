"""Тест по закону Фиттса (ISO 9241-9, multidirectional tapping).

Это экспериментальное ядро работы: стандартизованная методика оценки
указывающих устройств. Тест не зависит от способа ввода — мишени кликаются
любым курсором (жестовым или обычной мышью), поэтому позволяет напрямую
сравнить модальности.

Метрики (по ISO 9241-9):
  ID  = log2(A / W + 1)                  — индекс сложности (Shannon).
  MT  — время наведения (сек).
  We  = 4.133 * SD(отклонений эндпоинтов вдоль оси движения) — эффективная ширина.
  Ae  — эффективная амплитуда (средняя пройденная дистанция).
  IDe = log2(Ae / We + 1)               — эффективный индекс сложности.
  TP  = IDe / MT (бит/с)                 — пропускная способность (главный показатель).
"""

import csv
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Click:
    condition_id: int
    amplitude: float
    width: float
    from_pos: Tuple[float, float]
    target_pos: Tuple[float, float]
    click_pos: Tuple[float, float]
    movement_time: float
    hit: bool


@dataclass
class ConditionResult:
    amplitude: float
    width: float
    n_trials: int
    id_nominal: float
    mt_mean: float
    error_rate: float
    effective_width: float
    effective_amplitude: float
    id_effective: float
    throughput: float


class FittsTest:
    """Логика теста, отделённая от отрисовки.

    Использование: generate() → последовательность мишеней; register_click()
    на каждый клик; results() в конце для агрегированных метрик."""

    def __init__(self, cfg, screen_w: int, screen_h: int):
        self.cfg = cfg
        self.cx = screen_w / 2.0
        self.cy = screen_h / 2.0
        self.targets: List[dict] = []
        self.clicks: List[Click] = []
        self._idx = 0
        self._last_click_pos: Optional[Tuple[float, float]] = None
        self._last_click_time: Optional[float] = None
        self.generate()

    def generate(self) -> None:
        """Строит последовательность мишеней по «звёздному» порядку обхода."""
        self.targets.clear()
        cond_id = 0
        n = self.cfg.num_targets
        for amplitude in self.cfg.ring_amplitudes:
            for width in self.cfg.target_widths:
                radius = amplitude / 2.0
                positions = []
                for i in range(n):
                    theta = 2 * math.pi * i / n
                    positions.append((self.cx + radius * math.cos(theta),
                                      self.cy + radius * math.sin(theta)))
                # Обход по звезде: шаг (n+1)//2 даёт ~постоянную амплитуду.
                step = (n + 1) // 2
                order, idx = [], 0
                for _ in range(n):
                    order.append(idx % n)
                    idx += step
                for _ in range(self.cfg.repetitions):
                    for o in order:
                        self.targets.append({
                            "condition_id": cond_id,
                            "amplitude": amplitude,
                            "width": width,
                            "pos": positions[o],
                        })
                cond_id += 1

    @property
    def current_target(self) -> Optional[dict]:
        return self.targets[self._idx] if self._idx < len(self.targets) else None

    @property
    def progress(self) -> Tuple[int, int]:
        return self._idx, len(self.targets)

    @property
    def finished(self) -> bool:
        return self._idx >= len(self.targets)

    def register_click(self, x: float, y: float,
                       timestamp: Optional[float] = None) -> Optional[bool]:
        """Регистрирует клик по текущей мишени. Возвращает hit/None (если конец)."""
        tgt = self.current_target
        if tgt is None:
            return None
        if timestamp is None:
            timestamp = time.time()

        tx, ty = tgt["pos"]
        dist = math.hypot(x - tx, y - ty)
        hit = dist <= tgt["width"] / 2.0

        if self._last_click_pos is not None and self._last_click_time is not None:
            mt = timestamp - self._last_click_time
            self.clicks.append(Click(
                condition_id=tgt["condition_id"], amplitude=tgt["amplitude"],
                width=tgt["width"], from_pos=self._last_click_pos,
                target_pos=(tx, ty), click_pos=(x, y), movement_time=mt, hit=hit))

        self._last_click_pos = (x, y)
        self._last_click_time = timestamp
        self._idx += 1
        return hit

    # ---- агрегированные метрики -------------------------------------------

    def results(self) -> List[ConditionResult]:
        by_cond: dict = {}
        for c in self.clicks:
            by_cond.setdefault(c.condition_id, []).append(c)

        out: List[ConditionResult] = []
        for cond_id, clicks in sorted(by_cond.items()):
            a = clicks[0].amplitude
            w = clicks[0].width
            mts = [c.movement_time for c in clicks]
            errors = sum(1 for c in clicks if not c.hit)
            mt_mean = sum(mts) / len(mts)
            error_rate = errors / len(clicks)

            # Проекция эндпоинта на ось движения (from → target).
            projections, amplitudes = [], []
            for c in clicks:
                fx, fy = c.from_pos
                tx, ty = c.target_pos
                ax, ay = tx - fx, ty - fy
                norm = math.hypot(ax, ay) or 1.0
                ux, uy = ax / norm, ay / norm
                ex, ey = c.click_pos[0] - tx, c.click_pos[1] - ty
                projections.append(ex * ux + ey * uy)  # отклонение вдоль оси
                amplitudes.append(math.hypot(c.click_pos[0] - fx, c.click_pos[1] - fy))

            sd = _std(projections)
            we = 4.133 * sd if sd > 0 else w
            ae = sum(amplitudes) / len(amplitudes)
            id_nominal = math.log2(a / w + 1)
            id_eff = math.log2(ae / we + 1) if we > 0 else 0.0
            tp = id_eff / mt_mean if mt_mean > 0 else 0.0

            out.append(ConditionResult(
                amplitude=a, width=w, n_trials=len(clicks), id_nominal=id_nominal,
                mt_mean=mt_mean, error_rate=error_rate, effective_width=we,
                effective_amplitude=ae, id_effective=id_eff, throughput=tp))
        return out

    def summary(self) -> dict:
        res = self.results()
        if not res:
            return {"throughput_mean": 0.0, "conditions": []}
        tp_mean = sum(r.throughput for r in res) / len(res)
        err_mean = sum(r.error_rate for r in res) / len(res)
        mt_mean = sum(r.mt_mean for r in res) / len(res)
        return {
            "throughput_mean": tp_mean,
            "error_rate_mean": err_mean,
            "mt_mean": mt_mean,
            "n_conditions": len(res),
            "conditions": res,
        }

    def save_csv(self, path: str, participant_id: str = "", method: str = "") -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["participant", "method", "condition", "amplitude", "width",
                        "ID", "MT_s", "throughput_bps", "error_rate",
                        "We", "Ae", "IDe", "n_trials"])
            for i, r in enumerate(self.results()):
                w.writerow([participant_id, method, i, r.amplitude, r.width,
                            round(r.id_nominal, 3), round(r.mt_mean, 4),
                            round(r.throughput, 3), round(r.error_rate, 3),
                            round(r.effective_width, 2), round(r.effective_amplitude, 2),
                            round(r.id_effective, 3), r.n_trials])
        return path


def _std(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)
