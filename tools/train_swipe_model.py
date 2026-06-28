"""Обучение и экспорт временных моделей свайпов (TCN/LSTM) для AirControl.

Research-сторона функции «динамические жесты через временные модели». Рантайм
(`aircontrol/gestures/dynamic.py::TemporalSwipeSequenceModel`) умеет только ПРЯМОЙ
проход на numpy; этот инструмент обучает веса офлайн и сохраняет их в .npz в
ТОЧНОМ формате, который ожидает рантайм.

Возможности:
  * Синтетический генератор траекторий 4 направлений (шум, дрожание, кривизна,
    переменная длина) — инструмент работает без единого собранного примера.
  * Опциональный реальный датасет: .npz с object-массивом `points` (список
    траекторий [(x, y), ...]) и `labels` (имена свайпов).
  * TCN на ЧИСТОМ numpy: ручной forward + backprop + Adam, детерминирован (seed).
  * LSTM: numpy-тренер (по умолчанию) либо PyTorch, если он установлен
    (requirements-optional, в продуктовый бандл НЕ входит); веса конвертируются
    в numpy-раскладку рантайма.
  * Печать accuracy на отложенной выборке и матрицы ошибок.

Признаки берутся из `_sequence_features` / `_resample_points` рантайма — train и
inference используют ОДНУ функцию, что гарантирует совпадение точности.

Пример:
    python tools/train_swipe_model.py --backend tcn --epochs 60
    python tools/train_swipe_model.py --backend lstm --out /tmp/swipe.npz
"""

import argparse
import importlib.util
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from aircontrol.config import DEFAULT_TEMPORAL_SWIPE_MODEL_PATH


def _load_dynamic_module():
    """Импортирует gestures/dynamic.py НАПРЯМУЮ по файлу, минуя пакетный __init__.

    Пакет `aircontrol.gestures` в __init__ тянет engine → mediapipe, что не нужно
    research-инструменту (тяжело, а на части окружений и вовсе подвисает). Сам
    dynamic.py зависит только от numpy/stdlib, поэтому загружаем его как
    отдельный модуль и переиспользуем рантайм-функции признаков 1-в-1."""
    path = os.path.join(_ROOT, "aircontrol", "gestures", "dynamic.py")
    spec = importlib.util.spec_from_file_location("_aircontrol_dynamic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dynamic = _load_dynamic_module()
SWIPE_LABELS = _dynamic.SWIPE_LABELS
TemporalSwipeSequenceModel = _dynamic.TemporalSwipeSequenceModel
_sequence_features = _dynamic._sequence_features

# Единичные направления свайпов в нормированных координатах кадра (y вниз).
_DIRECTIONS: Dict[str, Tuple[float, float]] = {
    "swipe_left":  (-1.0, 0.0),
    "swipe_right": (1.0, 0.0),
    "swipe_up":    (0.0, -1.0),
    "swipe_down":  (0.0, 1.0),
}


# ===========================================================================
# Синтетические данные
# ===========================================================================

def _make_swipe(label: str, rng: np.random.Generator) -> List[Tuple[float, float]]:
    """Генерирует одну правдоподобную траекторию свайпа заданного направления.

    Модель учитывает: переменную длину взмаха и число кадров, перпендикулярную
    кривизну (дуга), покадровое дрожание руки и небольшой поворот направления."""
    dx, dy = _DIRECTIONS[label]
    # Поворот основного направления на малый угол (рука редко идёт идеально ровно).
    ang = rng.normal(0.0, 0.12)
    ca, sa = np.cos(ang), np.sin(ang)
    main = np.array([dx * ca - dy * sa, dx * sa + dy * ca])
    perp = np.array([-main[1], main[0]])               # нормаль к направлению

    n = int(rng.integers(6, 16))                       # число кадров взмаха
    amp = rng.uniform(0.22, 0.40)                      # амплитуда смещения
    curve = rng.normal(0.0, 0.05)                      # величина дуги
    start = np.array([rng.uniform(0.35, 0.65),
                      rng.uniform(0.35, 0.65)])        # старт около центра

    pts: List[Tuple[float, float]] = []
    for i in range(n):
        s = i / (n - 1)
        # Лёгкое ease-in-out: взмах ускоряется и тормозит (не идеально линейный).
        prog = 0.5 - 0.5 * np.cos(np.pi * s)
        bow = curve * np.sin(np.pi * s)                # дуга: 0 на концах
        p = start + main * amp * prog + perp * bow
        p = p + rng.normal(0.0, 0.006, size=2)         # дрожание руки
        pts.append((float(p[0]), float(p[1])))
    return pts


def generate_synthetic(n_per_class: int, seed: int,
                       ) -> Tuple[List[List[Tuple[float, float]]], List[str]]:
    """Сбалансированный синтетический датасет: n_per_class траекторий на класс."""
    rng = np.random.default_rng(seed)
    points: List[List[Tuple[float, float]]] = []
    labels: List[str] = []
    for label in SWIPE_LABELS:
        for _ in range(n_per_class):
            points.append(_make_swipe(label, rng))
            labels.append(label)
    return points, labels


def load_real_dataset(path: str) -> Tuple[List[List[Tuple[float, float]]], List[str]]:
    """Грузит реальный датасет .npz: object-массив `points` + массив `labels`."""
    data = np.load(path, allow_pickle=True)
    points = [list(map(tuple, np.asarray(p, dtype=float).reshape(-1, 2)))
              for p in data["points"]]
    labels = [str(x) for x in data["labels"].tolist()]
    if len(points) != len(labels):
        raise ValueError("points и labels разной длины")
    return points, labels


# ===========================================================================
# Подготовка тензоров
# ===========================================================================

def build_tensors(points: Sequence[Sequence[Tuple[float, float]]],
                  labels: Sequence[str], seq_len: int, classes: Sequence[str],
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Превращает траектории в X[N, T, 4] и целочисленные метки y[N]."""
    idx = {c: i for i, c in enumerate(classes)}
    X = np.stack([_sequence_features(p, seq_len) for p in points]).astype(np.float64)
    y = np.array([idx[l] for l in labels], dtype=np.int64)
    return X, y


def stratified_split(y: np.ndarray, test_frac: float, seed: int,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Стратифицированное по классам детерминированное разбиение на индексы."""
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        cut = max(1, int(round(len(ci) * test_frac)))
        te_idx.extend(ci[:cut].tolist())
        tr_idx.extend(ci[cut:].tolist())
    tr_idx = np.array(tr_idx); te_idx = np.array(te_idx)
    rng.shuffle(tr_idx)
    return tr_idx, te_idx


# ===========================================================================
# Утилиты обучения
# ===========================================================================

class _Adam:
    """Минимальный Adam-оптимизатор для словаря параметров."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 0.01,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _onehot(y: np.ndarray, n: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], n))
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


# ===========================================================================
# TCN на чистом numpy (forward + backprop + Adam)
# ===========================================================================

def train_tcn(Xtr: np.ndarray, ytr: np.ndarray, n_classes: int, *,
              out_ch: int, kernel: int, epochs: int, lr: float, seed: int,
              ) -> Dict[str, np.ndarray]:
    """Обучает TCN, идентичную рантайм-инференсу: conv(pad=k//2)→ReLU→mean→linear.

    Возвращает веса в раскладке рантайма: conv_w[k, in, out], conv_b[out],
    W[out, classes], b[classes]."""
    rng = np.random.default_rng(seed)
    N, T, in_ch = Xtr.shape
    pad = kernel // 2

    params = {
        "conv_w": rng.normal(0, 1.0 / np.sqrt(kernel * in_ch),
                             (kernel, in_ch, out_ch)),
        "conv_b": np.zeros(out_ch),
        "W": rng.normal(0, 1.0 / np.sqrt(out_ch), (out_ch, n_classes)),
        "b": np.zeros(n_classes),
    }
    opt = _Adam(params, lr=lr)
    Y = _onehot(ytr, n_classes)

    # Развёртка conv в матрицу окон: Xpad[N, T+2pad, in] → cols[N, T, k, in].
    Xpad = np.pad(Xtr, ((0, 0), (pad, pad), (0, 0)))
    cols = np.stack([Xpad[:, t:t + kernel, :] for t in range(T)], axis=1)  # [N,T,k,in]

    for ep in range(epochs):
        # --- forward ---
        # conv[N, T, out] = sum_{k,in} cols * conv_w
        conv = np.tensordot(cols, params["conv_w"], axes=([2, 3], [0, 1]))
        conv = conv + params["conv_b"]
        relu = np.maximum(conv, 0.0)
        pooled = relu.mean(axis=1)                       # [N, out]
        logits = pooled @ params["W"] + params["b"]      # [N, classes]
        probs = _softmax_rows(logits)

        # --- backward ---
        dlogits = (probs - Y) / N                        # [N, classes]
        gW = pooled.T @ dlogits
        gb = dlogits.sum(axis=0)
        dpooled = dlogits @ params["W"].T                # [N, out]
        drelu = np.repeat(dpooled[:, None, :], T, axis=1) / T   # [N, T, out]
        dconv = drelu * (conv > 0)                       # ReLU grad
        gconv_w = np.tensordot(cols, dconv, axes=([0, 1], [0, 1]))  # [k, in, out]
        gconv_b = dconv.sum(axis=(0, 1))

        opt.step(params, {"conv_w": gconv_w, "conv_b": gconv_b, "W": gW, "b": gb})

        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            loss = -np.mean(np.log(probs[np.arange(N), ytr] + 1e-12))
            acc = float(np.mean(np.argmax(probs, axis=1) == ytr))
            print(f"  [tcn] epoch {ep + 1:3d}/{epochs}  loss={loss:.4f}  train_acc={acc:.3f}")

    return params


# ===========================================================================
# LSTM на чистом numpy (BPTT + Adam) — корректный fallback при отсутствии torch
# ===========================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_lstm_numpy(Xtr: np.ndarray, ytr: np.ndarray, n_classes: int, *,
                     hidden: int, epochs: int, lr: float, seed: int,
                     ) -> Dict[str, np.ndarray]:
    """Обучает LSTM на numpy с полным BPTT. Раскладка весов = рантайм-инференс.

    Ячейка: i,f,o = sigmoid(x@Wx + h@Uh + b); g = tanh(...); c = f*c + i*g;
    h = o*tanh(c). Последнее h → линейный слой. Батч — все примеры разом."""
    rng = np.random.default_rng(seed)
    N, T, in_ch = Xtr.shape
    sx = 1.0 / np.sqrt(in_ch)
    sh = 1.0 / np.sqrt(hidden)
    p: Dict[str, np.ndarray] = {}
    for g in ("i", "f", "o", "g"):
        p[f"W{g}"] = rng.normal(0, sx, (in_ch, hidden))
        p[f"U{g}"] = rng.normal(0, sh, (hidden, hidden))
        p[f"b{g}"] = np.zeros(hidden)
    p["bf"] += 1.0                                       # forget-bias для памяти
    p["W"] = rng.normal(0, sh, (hidden, n_classes))
    p["b"] = np.zeros(n_classes)

    opt = _Adam(p, lr=lr)
    Y = _onehot(ytr, n_classes)

    for ep in range(epochs):
        # --- forward, кэшируем вентили для BPTT ---
        h = np.zeros((N, hidden)); c = np.zeros((N, hidden))
        cache = []
        for t in range(T):
            x = Xtr[:, t, :]
            i = _sigmoid(x @ p["Wi"] + h @ p["Ui"] + p["bi"])
            f = _sigmoid(x @ p["Wf"] + h @ p["Uf"] + p["bf"])
            o = _sigmoid(x @ p["Wo"] + h @ p["Uo"] + p["bo"])
            g = np.tanh(x @ p["Wg"] + h @ p["Ug"] + p["bg"])
            c_prev = c
            c = f * c + i * g
            tc = np.tanh(c)
            h_prev = h
            h = o * tc
            cache.append((x, h_prev, c_prev, i, f, o, g, c, tc))

        logits = h @ p["W"] + p["b"]
        probs = _softmax_rows(logits)

        # --- backward ---
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        dlogits = (probs - Y) / N
        grads["W"] = h.T @ dlogits
        grads["b"] = dlogits.sum(axis=0)
        dh = dlogits @ p["W"].T
        dc = np.zeros((N, hidden))
        for t in reversed(range(T)):
            x, h_prev, c_prev, i, f, o, g, c_t, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc * tc)
            di = dc * g
            dg = dc * i
            df = dc * c_prev
            dc_prev = dc * f
            # пред-активационные градиенты вентилей
            dai = di * i * (1 - i)
            daf = df * f * (1 - f)
            dao = do * o * (1 - o)
            dag = dg * (1 - g * g)
            for name, da in (("i", dai), ("f", daf), ("o", dao), ("g", dag)):
                grads[f"W{name}"] += x.T @ da
                grads[f"U{name}"] += h_prev.T @ da
                grads[f"b{name}"] += da.sum(axis=0)
            # Градиент по h_prev идёт через РЕКУРРЕНТНЫЕ матрицы U (hidden→hidden),
            # а не через входные W (Wx — это путь к входу x, не к прошлому h).
            dh = (dai @ p["Ui"].T + daf @ p["Uf"].T
                  + dao @ p["Uo"].T + dag @ p["Ug"].T)
            dc = dc_prev

        opt.step(p, grads)

        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            loss = -np.mean(np.log(probs[np.arange(N), ytr] + 1e-12))
            acc = float(np.mean(np.argmax(probs, axis=1) == ytr))
            print(f"  [lstm] epoch {ep + 1:3d}/{epochs}  loss={loss:.4f}  train_acc={acc:.3f}")

    return p


def train_lstm_torch(Xtr: np.ndarray, ytr: np.ndarray, n_classes: int, *,
                     hidden: int, epochs: int, lr: float, seed: int,
                     ) -> Dict[str, np.ndarray]:
    """Обучает LSTM на PyTorch и конвертирует веса в numpy-раскладку рантайма."""
    import torch                                          # noqa: локальный импорт
    import torch.nn as nn

    torch.manual_seed(seed)
    in_ch = Xtr.shape[2]
    X = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.long)

    lstm = nn.LSTM(in_ch, hidden, batch_first=True)
    head = nn.Linear(hidden, n_classes)
    opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        opt.zero_grad()
        out, _ = lstm(X)
        logits = head(out[:, -1, :])
        loss = lossf(logits, y)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            acc = float((logits.argmax(1) == y).float().mean())
            print(f"  [lstm/torch] epoch {ep + 1:3d}/{epochs}  "
                  f"loss={loss.item():.4f}  train_acc={acc:.3f}")

    # torch LSTM хранит веса как [4*hidden, in] в порядке i,f,g,o → переразложим.
    wih = lstm.weight_ih_l0.detach().numpy()              # [4H, in]
    whh = lstm.weight_hh_l0.detach().numpy()              # [4H, H]
    bih = lstm.bias_ih_l0.detach().numpy()
    bhh = lstm.bias_hh_l0.detach().numpy()
    H = hidden
    sl = {"i": slice(0, H), "f": slice(H, 2 * H),
          "g": slice(2 * H, 3 * H), "o": slice(3 * H, 4 * H)}
    p: Dict[str, np.ndarray] = {}
    for g in ("i", "f", "o", "g"):
        p[f"W{g}"] = wih[sl[g]].T.astype(np.float64)     # [in, H]
        p[f"U{g}"] = whh[sl[g]].T.astype(np.float64)     # [H, H]
        p[f"b{g}"] = (bih[sl[g]] + bhh[sl[g]]).astype(np.float64)
    p["W"] = head.weight.detach().numpy().T.astype(np.float64)   # [H, classes]
    p["b"] = head.bias.detach().numpy().astype(np.float64)
    return p


# ===========================================================================
# Оценка и экспорт
# ===========================================================================

def evaluate(model: TemporalSwipeSequenceModel,
             points: Sequence[Sequence[Tuple[float, float]]],
             labels: Sequence[str]) -> Tuple[float, np.ndarray, List[str]]:
    """Accuracy и матрица ошибок на отложенной выборке (по сырым траекториям)."""
    classes = model.labels
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    correct = 0
    for pts, true in zip(points, labels):
        pred, _ = model.predict(pts)
        cm[idx[true], idx[pred]] += 1
        correct += int(pred == true)
    acc = correct / max(1, len(labels))
    return acc, cm, classes


def print_confusion(cm: np.ndarray, classes: Sequence[str]) -> None:
    short = [c.replace("swipe_", "") for c in classes]
    print("\nМатрица ошибок (строки=истина, столбцы=предсказание):")
    print("           " + "".join(f"{s:>8}" for s in short))
    for i, c in enumerate(short):
        print(f"  {c:>8} " + "".join(f"{v:>8}" for v in cm[i]))


def export_npz(path: str, backend: str, classes: Sequence[str], seq_len: int,
               weights: Dict[str, np.ndarray]) -> None:
    """Сохраняет модель в .npz в ТОЧНОМ формате рантайма."""
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"backend": backend, "labels": np.array(list(classes)),
               "sequence_length": int(seq_len)}
    payload.update({k: np.asarray(v, dtype=np.float64) for k, v in weights.items()})
    np.savez(path, **payload)


def run(args: argparse.Namespace) -> int:
    classes = list(SWIPE_LABELS)
    np.random.seed(args.seed)                            # глобальный сид на всякий

    # --- данные ---
    points, labels = generate_synthetic(args.samples_per_class, args.seed)
    if args.dataset and os.path.exists(args.dataset):
        rp, rl = load_real_dataset(args.dataset)
        points += rp
        labels += rl
        print(f"Реальный датасет: +{len(rp)} траекторий из {args.dataset}")
    print(f"Всего траекторий: {len(points)} | классы: {classes} | "
          f"sequence_length={args.seq_len}")

    X, y = build_tensors(points, labels, args.seq_len, classes)
    tr_idx, te_idx = stratified_split(y, args.test_frac, args.seed)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    pts_te = [points[i] for i in te_idx]
    lab_te = [labels[i] for i in te_idx]
    print(f"Обучение: {len(tr_idx)} | отложено: {len(te_idx)}")

    # --- обучение ---
    if args.backend == "tcn":
        weights = train_tcn(Xtr, ytr, len(classes), out_ch=args.out_ch,
                            kernel=args.kernel, epochs=args.epochs, lr=args.lr,
                            seed=args.seed)
    elif args.backend == "lstm":
        use_torch = False
        if not args.no_torch:
            try:
                import torch  # noqa: F401
                use_torch = True
            except Exception:
                use_torch = False
        if use_torch:
            print("PyTorch найден — обучаю LSTM на torch, экспортирую в numpy.")
            weights = train_lstm_torch(Xtr, ytr, len(classes), hidden=args.hidden,
                                       epochs=args.epochs, lr=args.lr, seed=args.seed)
        else:
            print("PyTorch недоступен — обучаю LSTM на чистом numpy (BPTT).")
            weights = train_lstm_numpy(Xtr, ytr, len(classes), hidden=args.hidden,
                                       epochs=args.epochs, lr=args.lr, seed=args.seed)
    else:
        print(f"Неизвестный backend: {args.backend}")
        return 2

    # --- экспорт и оценка через РАНТАЙМ-загрузчик (проверка формата) ---
    export_npz(args.out, args.backend, classes, args.seq_len, weights)
    print(f"\nМодель сохранена: {args.out}")

    model = TemporalSwipeSequenceModel.load(args.out, args.backend)
    if model is None:
        print("ОШИБКА: экспортированную модель не удалось загрузить рантаймом.")
        return 1

    acc, cm, _ = evaluate(model, pts_te, lab_te)
    print(f"\nТочность на отложенной выборке: {acc:.3f}  (chance ≈ {1/len(classes):.3f})")
    print_confusion(cm, classes)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Обучение/экспорт временной модели свайпов")
    p.add_argument("--backend", choices=["tcn", "lstm"], default="tcn")
    p.add_argument("--out", default=DEFAULT_TEMPORAL_SWIPE_MODEL_PATH,
                   help="путь к экспортируемому .npz")
    p.add_argument("--dataset", default="",
                   help="опц. .npz реальных траекторий (object 'points' + 'labels')")
    p.add_argument("--samples-per-class", type=int, default=200, dest="samples_per_class")
    p.add_argument("--seq-len", type=int, default=24, dest="seq_len")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--test-frac", type=float, default=0.2, dest="test_frac")
    p.add_argument("--seed", type=int, default=1234)
    # TCN
    p.add_argument("--out-ch", type=int, default=12, dest="out_ch")
    p.add_argument("--kernel", type=int, default=3)
    # LSTM
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--no-torch", action="store_true", dest="no_torch",
                   help="принудительно numpy-LSTM, даже если torch установлен")
    return p


def main(argv: List[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
