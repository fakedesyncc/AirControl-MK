"""Анализ и построение исследовательских графиков.

Генерирует готовые к вставке в работу рисунки (PNG, 150 dpi) из артефактов
системы: датасета жестов, бенчмарка фильтров, CSV теста Фиттса и телеметрии.

Использует matplotlib с backend 'Agg' (без GUI), поэтому работает headless.
CLI: python -m aircontrol report
"""

import csv
import glob
import math
import os
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import AppConfig  # noqa: E402

try:
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import confusion_matrix  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    SKLEARN = True
except Exception:
    SKLEARN = False


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# --------------------------------------------------------- confusion matrix

def plot_confusion_matrix(dataset_path: str, out_path: str,
                          backend: str = "rf") -> Optional[float]:
    """Матрица ошибок ML-распознавателя на отложенной выборке. Возвращает accuracy."""
    from ..gestures.ml import GestureDataset, MLPoseClassifier

    ds = GestureDataset.load(dataset_path)
    if len(ds) < 20 or not SKLEARN:
        print("[analysis] Нет данных или scikit-learn — пропуск confusion matrix")
        return None

    X = np.array(ds.X, dtype=np.float32)
    y = np.array(ds.y)
    labels = sorted(set(y.tolist()))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=42, stratify=y)
    clf = MLPoseClassifier(backend=backend)
    clf.fit(X_tr, y_tr)
    y_pred = np.array([clf.predict_features(x) for x in X_te])
    acc = float((y_pred == y_te).mean())
    cm = confusion_matrix(y_te, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказано"); ax.set_ylabel("Истинно")
    ax.set_title(f"Матрица ошибок ({backend.upper()}), accuracy = {acc:.3f}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _ensure_dir(out_path); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[analysis] confusion matrix → {out_path} (acc={acc:.3f})")
    return acc


# --------------------------------------------------------- feature space (PCA)

def plot_feature_space(dataset_path: str, out_path: str) -> None:
    """2D-проекция (PCA) пространства признаков — наглядная разделимость поз."""
    from ..gestures.ml import GestureDataset

    ds = GestureDataset.load(dataset_path)
    if len(ds) < 20 or not SKLEARN:
        print("[analysis] Нет данных или scikit-learn — пропуск PCA")
        return
    X = np.array(ds.X, dtype=np.float32)
    y = np.array(ds.y)
    Z = PCA(n_components=2, random_state=42).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6))

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in sorted(set(y.tolist())):
        m = y == label
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.6, label=label)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Пространство признаков поз (PCA)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _ensure_dir(out_path); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[analysis] PCA пространства признаков → {out_path}")


# --------------------------------------------------------- filter benchmark

def plot_filter_benchmark(out_path: str) -> None:
    """Сравнение фильтров: jitter и latency на двух осях."""
    from .filter_benchmark import run_benchmark

    rows = run_benchmark()
    names = [r["filter"] for r in rows]
    jitter = [r["jitter_std"] for r in rows]
    latency = [r["latency_ms"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    ax1.bar(x - 0.2, jitter, 0.4, color="#4c72b0", label="jitter (СКО)")
    ax1.set_ylabel("jitter (СКО, норм.)", color="#4c72b0")
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, latency, 0.4, color="#dd8452", label="latency (мс)")
    ax2.set_ylabel("latency (мс)", color="#dd8452")
    ax1.set_title("Фильтры стабилизации: jitter vs latency")
    fig.tight_layout()
    _ensure_dir(out_path); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[analysis] бенчмарк фильтров → {out_path}")


# --------------------------------------------------------- Fitts regression

def _read_fitts_csvs(log_dir: str) -> dict:
    """Группирует строки fitts_*.csv по методу ввода: {method: [(ID, MT, TP), ...]}."""
    data: dict = {}
    for path in glob.glob(os.path.join(log_dir, "fitts_*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    method = row.get("method") or "unknown"
                    data.setdefault(method, []).append(
                        (float(row["ID"]), float(row["MT_s"]), float(row["throughput_bps"])))
                except (KeyError, ValueError):
                    continue
    return data


def plot_fitts_regression(log_dir: str, out_path: str) -> Optional[dict]:
    """Регрессия закона Фиттса MT = a + b·ID с R² для каждого метода ввода."""
    data = _read_fitts_csvs(log_dir)
    if not data:
        print("[analysis] Нет fitts_*.csv — пропуск регрессии (сначала прогоните 'fitts')")
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    results = {}
    for method, rows in data.items():
        ids = np.array([r[0] for r in rows])
        mts = np.array([r[1] for r in rows])
        if len(ids) < 2:
            continue
        b, a = np.polyfit(ids, mts, 1)               # MT = a + b*ID
        pred = a + b * ids
        ss_res = float(np.sum((mts - pred) ** 2))
        ss_tot = float(np.sum((mts - mts.mean()) ** 2)) or 1e-9
        r2 = 1 - ss_res / ss_tot
        tp = 1.0 / b if b > 0 else float("nan")      # пропускная способность (бит/с)
        results[method] = {"a": a, "b": b, "r2": r2, "throughput": tp}

        sc = ax.scatter(ids, mts, s=20, alpha=0.6, label=f"{method} (R²={r2:.2f}, TP={tp:.1f} б/с)")
        xs = np.linspace(ids.min(), ids.max(), 50)
        ax.plot(xs, a + b * xs, color=sc.get_facecolor()[0])
    ax.set_xlabel("Индекс сложности ID = log₂(A/W+1), бит")
    ax.set_ylabel("Время наведения MT, с")
    ax.set_title("Закон Фиттса: MT = a + b·ID")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _ensure_dir(out_path); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[analysis] регрессия Фиттса → {out_path}")
    return results


# --------------------------------------------------------- telemetry FPS

def plot_telemetry(log_dir: str, out_path: str) -> None:
    files = sorted(glob.glob(os.path.join(log_dir, "telemetry_*.csv")))
    if not files:
        print("[analysis] Нет telemetry_*.csv — пропуск")
        return
    rows = list(csv.DictReader(open(files[-1], newline="", encoding="utf-8")))
    if not rows:
        return
    t0 = float(rows[0]["timestamp"])
    t = [float(r["timestamp"]) - t0 for r in rows]
    fps = [float(r["fps"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, fps, color="#55a868")
    ax.set_xlabel("Время, с"); ax.set_ylabel("FPS")
    ax.set_title(f"Производительность (filter={rows[0].get('filter')}, "
                 f"rec={rows[0].get('recognizer')})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _ensure_dir(out_path); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[analysis] телеметрия FPS → {out_path}")


# --------------------------------------------------------- SUS / NASA-TLX

def plot_usability(log_dir: str, out_path: str) -> Optional[dict]:
    """Plot SUS and NASA-TLX means by study condition."""
    from .usability import summarize_usability_rows

    rows = []
    for path in glob.glob(os.path.join(log_dir, "usability*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    summary = summarize_usability_rows(rows)
    if not summary:
        print("[analysis] Нет usability*.csv — пропуск SUS/NASA-TLX")
        return None

    conditions = sorted(summary)
    sus = [summary[c]["sus_mean"] for c in conditions]
    tlx = [summary[c]["nasa_tlx_mean"] for c in conditions]
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.18, sus, 0.36, label="SUS", color="#4c72b0")
    ax.bar(x + 0.18, tlx, 0.36, label="NASA-TLX", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Баллы, 0..100")
    ax.set_title("Пользовательская оценка удобства")
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _ensure_dir(out_path); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[analysis] SUS/NASA-TLX → {out_path}")
    return summary


# --------------------------------------------------------- сводный отчёт

def generate_report(cfg: AppConfig, out_dir: str = None) -> None:
    out_dir = out_dir or os.path.join(os.path.dirname(cfg.gestures.ml_dataset_path), "figures")
    print(f"=== Генерация графиков → {out_dir} ===")

    # Если датасета нет — генерируем синтетику, чтобы рисунки были сразу.
    from ..gestures.ml import GestureDataset
    if len(GestureDataset.load(cfg.gestures.ml_dataset_path)) < 20:
        from ..gestures.synthetic import generate_synthetic_dataset
        print("[analysis] Датасет пуст — генерирую синтетический для демонстрации")
        generate_synthetic_dataset(per_pose=300).save(cfg.gestures.ml_dataset_path)

    plot_filter_benchmark(os.path.join(out_dir, "filters.png"))
    plot_confusion_matrix(cfg.gestures.ml_dataset_path,
                          os.path.join(out_dir, "confusion_matrix.png"))
    plot_feature_space(cfg.gestures.ml_dataset_path,
                       os.path.join(out_dir, "feature_space_pca.png"))
    plot_fitts_regression(cfg.evaluation.log_dir, os.path.join(out_dir, "fitts_regression.png"))
    plot_telemetry(cfg.telemetry.log_dir, os.path.join(out_dir, "telemetry_fps.png"))
    plot_usability(cfg.evaluation.log_dir, os.path.join(out_dir, "usability.png"))
    print("=== Готово ===")


if __name__ == "__main__":
    generate_report(AppConfig.load())
