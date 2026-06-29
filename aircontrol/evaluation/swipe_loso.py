"""Leave-one-subject-out (LOSO) оценка временных моделей свайпов.

Зачем именно LOSO. Случайное train/test разбиение перемешивает кадры одного
человека между обучением и тестом, поэтому модель может «подсмотреть» личный
почерк взмаха и завысить точность. Для заявления «жесты работают у незнакомого
пользователя» нужен протокол, где каждый испытуемый по очереди полностью
выносится в тест, а модель учится на остальных. Это и есть LOSO: честная оценка
обобщения на нового пользователя.

Модуль чисто numpy-шный и НЕ тянет mediapipe: тренер (`tools/train_swipe_model.py`)
и рантайм-признаки (`aircontrol/gestures/dynamic.py`) загружаются ПО ФАЙЛУ, минуя
пакетный `aircontrol.gestures.__init__`, который импортирует движок → mediapipe.
Тем самым LOSO запускается на любом окружении, в т.ч. без камеры и mediapipe.

Формат датасета (.npz, allow_pickle=True):
  * points  — object-массив траекторий, каждая [(x, y), ...] в норм. координатах;
  * labels  — массив имён свайпов (SWIPE_LABELS);
  * subject — массив id испытуемого (строка/число) длиной как points/labels.

CLI:
    python -m aircontrol.evaluation.swipe_loso --dataset data/swipes.npz
    python -m aircontrol.evaluation.swipe_loso --dataset d.npz --backend tcn --epochs 80
"""

import argparse
import importlib.util
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_module_by_path(name: str, rel_path: str):
    """Загружает модуль НАПРЯМУЮ по файлу, минуя пакетные __init__.

    `aircontrol.gestures.__init__` тянет движок → mediapipe (тяжело, а на части
    окружений подвисает). И тренер, и рантайм-признаки зависят только от
    numpy/stdlib, поэтому грузим их как отдельные модули и переиспользуем 1-в-1."""
    path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    spec.loader.exec_module(mod)
    return mod


# Тренер сам грузит dynamic.py по файлу и реэкспортирует SWIPE_LABELS,
# build_tensors, _sequence_features, train_tcn — переиспользуем их напрямую.
_trainer = _load_module_by_path("_swipe_trainer", os.path.join("tools", "train_swipe_model.py"))

SWIPE_LABELS: Tuple[str, ...] = _trainer.SWIPE_LABELS
build_tensors = _trainer.build_tensors
train_tcn = _trainer.train_tcn
train_lstm_numpy = _trainer.train_lstm_numpy
_sequence_features = _trainer._sequence_features


# ===========================================================================
# Загрузка многосубъектного датасета
# ===========================================================================

def load_multisubject_dataset(
    path: str,
) -> Tuple[List[List[Tuple[float, float]]], List[str], List[str]]:
    """Грузит .npz с object-массивом `points`, `labels` и `subject`.

    Возвращает (points, labels, subjects). subject приводится к строке, чтобы
    числовые и строковые id вели себя одинаково."""
    data = np.load(path, allow_pickle=True)
    for key in ("points", "labels", "subject"):
        if key not in data:
            raise ValueError(f"в датасете нет обязательного ключа '{key}'")
    points = [list(map(tuple, np.asarray(p, dtype=float).reshape(-1, 2)))
              for p in data["points"]]
    labels = [str(x) for x in data["labels"].tolist()]
    subjects = [str(x) for x in data["subject"].tolist()]
    if not (len(points) == len(labels) == len(subjects)):
        raise ValueError("points, labels и subject должны быть одной длины")
    return points, labels, subjects


# ===========================================================================
# Чистый numpy-инференс обученных весов (без рантайм-класса, чтобы не зависеть
# от пакета gestures). Раскладка весов = ровно то, что отдаёт тренер.
# ===========================================================================

def _predict_logits_tcn(X: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """Батч-инференс TCN: conv(pad=k//2) → ReLU → mean-pool → linear. X[N,T,in]."""
    conv_w, conv_b = w["conv_w"], w["conv_b"]
    kernel, in_ch, _ = conv_w.shape
    if X.shape[2] != in_ch:
        raise ValueError(f"несовпадение каналов TCN: {X.shape[2]} != {in_ch}")
    pad = kernel // 2
    N, T, _ = X.shape
    Xpad = np.pad(X, ((0, 0), (pad, pad), (0, 0)))
    cols = np.stack([Xpad[:, t:t + kernel, :] for t in range(T)], axis=1)  # [N,T,k,in]
    conv = np.tensordot(cols, conv_w, axes=([2, 3], [0, 1])) + conv_b      # [N,T,out]
    pooled = np.maximum(conv, 0.0).mean(axis=1)                            # [N,out]
    return pooled @ w["W"] + w["b"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _predict_logits_lstm(X: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """Батч-инференс LSTM (последнее h → линейный слой). X[N,T,in]."""
    N, T, _ = X.shape
    hidden = int(w["bi"].shape[0])
    h = np.zeros((N, hidden))
    c = np.zeros((N, hidden))
    for t in range(T):
        x = X[:, t, :]
        i = _sigmoid(x @ w["Wi"] + h @ w["Ui"] + w["bi"])
        f = _sigmoid(x @ w["Wf"] + h @ w["Uf"] + w["bf"])
        o = _sigmoid(x @ w["Wo"] + h @ w["Uo"] + w["bo"])
        g = np.tanh(x @ w["Wg"] + h @ w["Ug"] + w["bg"])
        c = f * c + i * g
        h = o * np.tanh(c)
    return h @ w["W"] + w["b"]


def _predict_labels(X: np.ndarray, weights: Dict[str, np.ndarray], backend: str,
                    classes: Sequence[str]) -> np.ndarray:
    logits = (_predict_logits_tcn(X, weights) if backend == "tcn"
              else _predict_logits_lstm(X, weights))
    return np.argmax(logits, axis=1)


# ===========================================================================
# LOSO
# ===========================================================================

def run_loso(
    points: Sequence[Sequence[Tuple[float, float]]],
    labels: Sequence[str],
    subjects: Sequence[str],
    *,
    backend: str = "tcn",
    seq_len: int = 24,
    epochs: int = 80,
    lr: float = 0.01,
    seed: int = 1234,
    out_ch: int = 12,
    kernel: int = 3,
    hidden: int = 16,
    verbose: bool = False,
) -> Dict[str, object]:
    """Прогоняет leave-one-subject-out по уникальным id испытуемых.

    Для каждого вынесенного субъекта: модель обучается на остальных и
    оценивается на нём. Возвращает dict с per-subject accuracy, mean ± std,
    агрегированной матрицей ошибок и метаданными протокола."""
    if backend not in ("tcn", "lstm"):
        raise ValueError(f"неизвестный backend: {backend}")
    classes = list(SWIPE_LABELS)
    n_cls = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    subj_arr = np.array([str(s) for s in subjects])
    unique_subjects = sorted(set(subj_arr.tolist()))
    if len(unique_subjects) < 2:
        raise ValueError("для LOSO нужно минимум 2 испытуемых")

    # Признаки строим один раз для всего датасета — фолд лишь выбирает индексы.
    X, y = build_tensors(points, labels, seq_len, classes)

    per_subject: List[Dict[str, object]] = []
    confusion = np.zeros((n_cls, n_cls), dtype=int)

    for fold, held in enumerate(unique_subjects):
        test_mask = subj_arr == held
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            continue

        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]

        # Детерминированный, но различающийся по фолду сид — фолды независимы.
        fold_seed = seed + fold
        if backend == "tcn":
            weights = train_tcn(Xtr, ytr, n_cls, out_ch=out_ch, kernel=kernel,
                                epochs=epochs, lr=lr, seed=fold_seed)
        else:
            weights = train_lstm_numpy(Xtr, ytr, n_cls, hidden=hidden,
                                       epochs=epochs, lr=lr, seed=fold_seed)

        pred = _predict_labels(Xte, weights, backend, classes)
        acc = float(np.mean(pred == yte))
        for true_i, pred_i in zip(yte, pred):
            confusion[int(true_i), int(pred_i)] += 1

        per_subject.append({
            "subject": held,
            "accuracy": acc,
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
        })
        if verbose:
            print(f"[loso] субъект {held!r}: acc={acc:.3f} "
                  f"(test={int(test_mask.sum())}, train={int(train_mask.sum())})")

    accs = np.array([r["accuracy"] for r in per_subject], dtype=float)
    return {
        "backend": backend,
        "classes": classes,
        "n_subjects": len(per_subject),
        "per_subject": per_subject,
        "mean_accuracy": float(accs.mean()) if accs.size else 0.0,
        "std_accuracy": float(accs.std()) if accs.size else 0.0,
        "chance": 1.0 / n_cls,
        "confusion_matrix": confusion,
        "seq_len": seq_len,
    }


# ===========================================================================
# Отчёт
# ===========================================================================

def print_report(results: Dict[str, object]) -> None:
    """Печатает per-subject accuracy, mean ± std и агрегированную матрицу ошибок."""
    classes = results["classes"]
    print(f"\nLOSO ({results['backend']}, {results['n_subjects']} испытуемых, "
          f"seq_len={results['seq_len']})")
    print("  Точность по вынесенным испытуемым:")
    for r in results["per_subject"]:
        print(f"    {str(r['subject']):>12}: acc={r['accuracy']:.3f} "
              f"(test={r['n_test']})")
    print(f"  Среднее ± std: {results['mean_accuracy']:.3f} ± "
          f"{results['std_accuracy']:.3f}  (chance ≈ {results['chance']:.3f})")

    short = [c.replace("swipe_", "") for c in classes]
    cm = results["confusion_matrix"]
    print("\n  Агрегированная матрица ошибок (строки=истина, столбцы=предсказание):")
    print("           " + "".join(f"{s:>8}" for s in short))
    for i, c in enumerate(short):
        print(f"  {c:>8} " + "".join(f"{v:>8}" for v in cm[i]))


def run(args: argparse.Namespace) -> int:
    if not args.dataset or not os.path.exists(args.dataset):
        print(f"Датасет не найден: {args.dataset!r}")
        return 2
    np.random.seed(args.seed)
    points, labels, subjects = load_multisubject_dataset(args.dataset)
    n_subj = len(set(subjects))
    print(f"Датасет: {len(points)} траекторий, {n_subj} испытуемых, "
          f"классы={list(SWIPE_LABELS)}")
    results = run_loso(points, labels, subjects, backend=args.backend,
                       seq_len=args.seq_len, epochs=args.epochs, lr=args.lr,
                       seed=args.seed, out_ch=args.out_ch, kernel=args.kernel,
                       hidden=args.hidden, verbose=args.verbose)
    print_report(results)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LOSO-оценка временной модели свайпов (leave-one-subject-out)")
    p.add_argument("--dataset", required=True,
                   help=".npz с object 'points' + 'labels' + 'subject'")
    p.add_argument("--backend", choices=["tcn", "lstm"], default="tcn")
    p.add_argument("--seq-len", type=int, default=24, dest="seq_len")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-ch", type=int, default=12, dest="out_ch")
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: List[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
