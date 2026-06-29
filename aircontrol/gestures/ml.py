"""Обучаемый ML-распознаватель поз руки.

Назначение для работы: показать, что распознавание жестов можно вынести из
жёстко закодированных порогов в обучаемую модель, и сравнить точность с
эвристикой (confusion matrix, accuracy, устойчивость к новым пользователям).

Реализация:
  * GestureDataset — сбор/хранение размеченных примеров (.npz).
  * MLPoseClassifier — kNN на numpy (всегда доступен, нулевые зависимости) либо
    MLP/RandomForest из scikit-learn, если он установлен.
  * train_from_dataset / evaluate — обучение и оценка с метриками.
"""

import os
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

from .features import POSE_LABELS, extract_features


def _ensure_parent_dir(path: str) -> None:
    """Создаёт родительский каталог для ``path``, если он указан.

    ``os.path.dirname`` для голого имени файла (например ``"model"``) возвращает
    пустую строку, на которой ``os.makedirs`` падает с ``FileNotFoundError``.
    Здесь это аккуратно обходится, чтобы сохранение по относительному пути в
    текущем каталоге не ломалось.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

try:
    from sklearn.neural_network import MLPClassifier  # type: ignore
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import accuracy_score, confusion_matrix  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class GestureDataset:
    """Хранилище размеченных примеров признаков жестов."""

    def __init__(self):
        self.X: List[np.ndarray] = []
        self.y: List[str] = []

    def add(self, landmarks: np.ndarray, label: str) -> None:
        self.X.append(extract_features(landmarks))
        self.y.append(label)

    def counts(self) -> dict:
        return dict(Counter(self.y))

    def save(self, path: str) -> None:
        _ensure_parent_dir(path)
        np.savez(path, X=np.array(self.X, dtype=np.float32),
                 y=np.array(self.y))

    @classmethod
    def load(cls, path: str) -> "GestureDataset":
        """Загружает датасет; на отсутствующем/битом файле возвращает пустой.

        Возврат пустого ``GestureDataset`` (а не исключение) позволяет вызывающему
        коду единообразно проверять ``len(ds)`` — например, ``train_from_dataset``
        затем выдаёт понятную ошибку «слишком мало примеров».
        """
        ds = cls()
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                ds.X = list(data["X"])
                ds.y = list(data["y"])
            except Exception as exc:
                print(f"[ml] Не удалось прочитать датасет ({exc}) — пустой набор")
                ds.X, ds.y = [], []
        return ds

    def __len__(self) -> int:
        return len(self.y)


class MLPoseClassifier:
    """Классификатор поз. backend: 'knn' | 'mlp' | 'rf'."""

    name = "ml"

    def __init__(self, backend: str = "knn", k: int = 5):
        self.backend = backend if (backend == "knn" or SKLEARN_AVAILABLE) else "knn"
        self.k = k
        self._labels: List[str] = []
        self._X: Optional[np.ndarray] = None     # для kNN
        self._y: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._model = None                        # для sklearn
        self._trained = False

    # ---- обучение ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._labels = sorted(set(y.tolist()))
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-6
        Xn = (X - self._mean) / self._std

        if self.backend == "knn":
            self._X, self._y = Xn, y
        elif self.backend == "mlp":
            self._model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                        random_state=42)
            self._model.fit(Xn, y)
        elif self.backend == "rf":
            self._model = RandomForestClassifier(n_estimators=200, random_state=42)
            self._model.fit(Xn, y)
        self._trained = True

    # ---- предсказание ------------------------------------------------------

    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Предсказание по сырым лендмаркам (используется в реальном времени)."""
        if not self._trained:
            return "none", 0.0
        return self._predict_vec(extract_features(landmarks))

    def predict_features(self, feat: np.ndarray) -> str:
        """Предсказание по готовому вектору признаков (для оценки на выборке)."""
        return self._predict_vec(feat)[0]

    def predict_confident(self, landmarks: np.ndarray,
                          min_confidence: float = 0.6) -> Tuple[str, float]:
        """Предсказание с порогом уверенности (опциональное гейтирование).

        Поведение-сохраняющая надстройка над :meth:`predict`: сначала вызывает
        обычный :meth:`predict`, и если уверенность ниже ``min_confidence``,
        возвращает ``("none", conf)`` вместо предсказанной метки. Это полезно,
        когда движок предпочитает «промолчать», а не сработать на неуверенной
        классификации. Сам :meth:`predict` при этом не меняется — старый код,
        который его вызывает, работает как прежде.

        ``min_confidence`` ожидается в диапазоне [0, 1]; уверенность kNN — это
        доля голосов соседей, у sklearn — максимум predict_proba.
        """
        label, conf = self.predict(landmarks)
        if label == "none" or conf < min_confidence:
            return "none", conf
        return label, conf

    def _predict_vec(self, feat: np.ndarray) -> Tuple[str, float]:
        xn = (feat - self._mean) / self._std

        if self.backend == "knn":
            dists = np.linalg.norm(self._X - xn, axis=1)
            idx = np.argsort(dists)[: self.k]
            votes = Counter(self._y[idx].tolist())
            label, count = votes.most_common(1)[0]
            return label, count / self.k

        proba = self._model.predict_proba([xn])[0]
        j = int(np.argmax(proba))
        return self._model.classes_[j], float(proba[j])

    # ---- сохранение/загрузка (только kNN; sklearn — через joblib опц.) ------

    def save(self, path: str) -> None:
        _ensure_parent_dir(path)
        if self.backend == "knn":
            np.savez(path, backend="knn", k=self.k, X=self._X, y=self._y,
                     mean=self._mean, std=self._std, labels=np.array(self._labels))
        else:
            import pickle
            with open(path + ".pkl", "wb") as f:
                pickle.dump({"backend": self.backend, "model": self._model,
                             "mean": self._mean, "std": self._std,
                             "labels": self._labels}, f)

    @classmethod
    def load(cls, path: str) -> Optional["MLPoseClassifier"]:
        # Любая ошибка загрузки (несовместимый формат, отсутствие sklearn для
        # распиновки модели и т.п.) → None, чтобы движок откатился на эвристику.
        try:
            # np.savez добавляет .npz, если расширения нет — учитываем оба варианта.
            npz_path = path if os.path.exists(path) else path + ".npz"
            if os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                clf = cls(backend="knn", k=int(data["k"]))
                clf._X, clf._y = data["X"], data["y"]
                clf._mean, clf._std = data["mean"], data["std"]
                clf._labels = list(data["labels"])
                clf._trained = True
                return clf
            if os.path.exists(path + ".pkl"):
                import pickle
                with open(path + ".pkl", "rb") as f:
                    d = pickle.load(f)
                clf = cls(backend=d["backend"])
                clf._model = d["model"]; clf._mean = d["mean"]; clf._std = d["std"]
                clf._labels = d["labels"]; clf._trained = True
                return clf
        except Exception as exc:
            print(f"[ml] Не удалось загрузить модель ({exc}) — откат на эвристику")
        return None


def train_from_dataset(dataset_path: str, model_path: str, backend: str = "knn",
                       test_size: float = 0.2) -> dict:
    """Обучает классификатор и возвращает метрики оценки (для отчёта в работе)."""
    ds = GestureDataset.load(dataset_path)
    if len(ds) < 10:
        raise ValueError(f"Слишком мало примеров для обучения: {len(ds)}")

    X = np.array(ds.X, dtype=np.float32)
    y = np.array(ds.y)

    n_classes = len(set(y.tolist()))
    if n_classes < 2:
        # Один класс нельзя ни обучить осмысленно (нечего различать), ни оценить
        # (нет матрицы ошибок). Сообщаем явно, а не молча сваливаемся в kNN.
        only = sorted(set(y.tolist()))
        raise ValueError(
            f"Нужно минимум 2 класса для обучения, получен 1: {only}")

    metrics = {"backend": backend, "n_samples": len(ds), "class_counts": ds.counts()}

    if SKLEARN_AVAILABLE and len(set(y)) > 1:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                                  random_state=42, stratify=y)
        clf = MLPoseClassifier(backend=backend)
        clf.fit(X_tr, y_tr)
        y_pred = np.array([clf.predict_features(x) for x in X_te])
        metrics["accuracy"] = float(accuracy_score(y_te, y_pred))
        labels = sorted(set(y.tolist()))
        metrics["labels"] = labels
        metrics["confusion_matrix"] = confusion_matrix(y_te, y_pred, labels=labels).tolist()
        clf.fit(X, y)  # финальная модель на всех данных
    else:
        clf = MLPoseClassifier(backend="knn")
        clf.fit(X, y)
        metrics["accuracy"] = None
        metrics["note"] = "scikit-learn недоступен — обучён kNN без отложенной выборки"

    clf.save(model_path)
    return metrics
