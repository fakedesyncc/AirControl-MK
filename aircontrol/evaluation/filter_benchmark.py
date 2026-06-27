"""Сравнительный бенчмарк фильтров стабилизации (воспроизводимый, без участника).

Прогоняет синтетический сигнал через все фильтры и измеряет два конкурирующих
показателя:
  * jitter (СКО в покое) — насколько фильтр гасит дрожание;
  * latency (задержка реакции на ступенчатое движение) — насколько он «тормозит».

Хороший фильтр одновременно даёт низкий jitter и низкую задержку. Результаты
этого скрипта можно прямо вставлять в раздел работы про выбор фильтра.
Запуск: python -m aircontrol.evaluation.filter_benchmark
"""

import numpy as np

from ..config import FilterConfig
from ..tracking.filters import create_filter

FS = 60.0  # частота кадров, Гц


def _signal(seed: int = 0):
    """Сигнал: покой 0.3 (1.5 c) → ступень до 0.7 → покой (1.5 c), + шум."""
    rng = np.random.default_rng(seed)
    n_rest = int(1.5 * FS)
    rest1 = np.full(n_rest, 0.3)
    rest2 = np.full(n_rest, 0.7)
    sig = np.concatenate([rest1, rest2])
    noise = rng.normal(0, 0.01, sig.size)
    return sig + noise, n_rest


def _latency_samples(filtered, n_rest, target_from=0.3, target_to=0.7):
    """Число кадров до достижения 90% амплитуды ступени."""
    threshold = target_from + 0.9 * (target_to - target_from)
    for i in range(n_rest, len(filtered)):
        if filtered[i] >= threshold:
            return i - n_rest
    return len(filtered) - n_rest


def run_benchmark() -> list:
    clean, n_rest = _signal()
    results = []
    for ftype in ["none", "ema", "one_euro", "kalman"]:
        cfg = FilterConfig(); cfg.type = ftype
        f = create_filter(cfg)
        out = []
        t = 0.0
        for v in clean:
            t += 1.0 / FS
            fx, _ = f.filter(v, v, t)
            out.append(fx)
        out = np.array(out)

        # jitter — СКО на последних 0.8 c первого покоя.
        rest_slice = out[int(0.7 * FS):n_rest]
        jitter = float(rest_slice.std())
        lat_frames = _latency_samples(out, n_rest)
        results.append({
            "filter": ftype,
            "jitter_std": jitter,
            "latency_frames": lat_frames,
            "latency_ms": lat_frames * 1000.0 / FS,
        })
    return results


def main():
    rows = run_benchmark()
    print(f"{'filter':10s} {'jitter(СКО)':>14s} {'latency(кадров)':>16s} {'latency(мс)':>13s}")
    print("-" * 56)
    for r in rows:
        print(f"{r['filter']:10s} {r['jitter_std']:>14.5f} "
              f"{r['latency_frames']:>16d} {r['latency_ms']:>13.1f}")

    # Data-driven вывод: среди фильтрующих (кроме none) ищем лучший компромисс
    # по нормализованной сумме рангов jitter и latency.
    cand = [r for r in rows if r["filter"] != "none"]
    j = np.array([r["jitter_std"] for r in cand])
    lt = np.array([r["latency_ms"] for r in cand])
    score = (j - j.min()) / (np.ptp(j) or 1) + (lt - lt.min()) / (np.ptp(lt) or 1)
    best = cand[int(np.argmin(score))]["filter"]
    print(f"\nЛучший компромисс «jitter + latency»: {best}")


if __name__ == "__main__":
    main()
