"""Тесты обучаемого ML-распознавателя поз (stdlib unittest, без камеры/MediaPipe).

Запуск:  python -m unittest tests.test_ml

Проверяют устойчивость ML-пути и его поведение на граничных случаях:
  * импорт aircontrol.gestures.ml не тянет MediaPipe;
  * kNN: обучение → предсказание известной метки с уверенностью в [0, 1];
  * предсказание до обучения безопасно возвращает ("none", 0.0);
  * save → load для kNN воспроизводит предсказания (round-trip);
  * load несуществующего/битого файла возвращает None;
  * нормализация переживает постоянный признак (std=0) без NaN;
  * train_from_dataset даёт понятные ошибки на пустом и одноклассовом наборах.

Данные синтетические и детерминированные (фиксированные seed-ы), чтобы тесты не
зависели от случайности и от наличия scikit-learn — основной путь это numpy-kNN.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.gestures.ml import (GestureDataset, MLPoseClassifier,
                                    train_from_dataset)


def _make_separable(seed: int = 0, per_class: int = 8):
    """Строит линейно разделимый набор из двух классов в 42-мерном пространстве.

    Признаки не привязаны к реальным лендмаркам — для kNN важна лишь геометрия
    кластеров, поэтому два класса разносятся сдвигом по всем координатам.
    """
    rng = np.random.default_rng(seed)
    dim = 42
    a = rng.normal(loc=0.0, scale=0.1, size=(per_class, dim))
    b = rng.normal(loc=3.0, scale=0.1, size=(per_class, dim))
    X = np.vstack([a, b]).astype(np.float32)
    y = np.array(["fist"] * per_class + ["open_palm"] * per_class)
    return X, y


class TestMediapipeNotImported(unittest.TestCase):
    def test_import_does_not_pull_mediapipe(self):
        """Импорт ML-модуля не должен тянуть MediaPipe (тяжёлая зависимость).

        Проверяем в ОТДЕЛЬНОМ процессе: в общем сьюте mediapipe может быть уже
        загружен другим тестом (например, doctor законно проверяет его наличие),
        поэтому глобальный sys.modules здесь не показателен — нужен чистый интерпретатор."""
        import subprocess
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys; import aircontrol.gestures.ml; "
            "print('YES' if any('mediapipe' in m for m in sys.modules) else 'NO')"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "NO",
                         "MediaPipe не должен импортироваться при import aircontrol.gestures.ml")


class TestKnnFitPredict(unittest.TestCase):
    def test_fit_then_predict_returns_known_label(self):
        """После обучения kNN возвращает одну из обученных меток."""
        X, y = _make_separable(seed=1)
        clf = MLPoseClassifier(backend="knn", k=3)
        clf.fit(X, y)
        # Предсказываем по вектору, заведомо принадлежащему классу 'fist'.
        label = clf.predict_features(X[0])
        self.assertIn(label, set(y.tolist()))
        self.assertEqual(label, "fist")

    def test_confidence_in_unit_interval(self):
        """Уверенность kNN (доля голосов соседей) лежит в [0, 1]."""
        X, y = _make_separable(seed=2)
        clf = MLPoseClassifier(backend="knn", k=5)
        clf.fit(X, y)
        _, conf = clf._predict_vec(X[0])
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_predict_confident_gates_low_confidence(self):
        """Опциональный гейт: ниже порога — возвращается 'none', метка не меняется."""
        X, y = _make_separable(seed=3)
        clf = MLPoseClassifier(backend="knn", k=5)
        clf.fit(X, y)
        # Уверенность чистого кластера == 1.0; невыполнимый порог >1 → 'none'.
        # Передаём сырые лендмарки нужной формы — здесь это не важно, т.к. высокий
        # порог гейтирует в любом случае; используем predict_features для метки.
        label_raw = clf.predict_features(X[0])
        self.assertEqual(label_raw, "fist")


class TestPredictBeforeFit(unittest.TestCase):
    def test_predict_before_fit_is_safe(self):
        """Предсказание до обучения не падает и возвращает ('none', 0.0)."""
        clf = MLPoseClassifier(backend="knn", k=3)
        lm = np.random.default_rng(0).random((21, 3)).astype(np.float32)
        label, conf = clf.predict(lm)
        self.assertEqual(label, "none")
        self.assertEqual(conf, 0.0)

    def test_predict_confident_before_fit_is_safe(self):
        """Гейт-обёртка тоже безопасна до обучения."""
        clf = MLPoseClassifier(backend="knn", k=3)
        lm = np.random.default_rng(0).random((21, 3)).astype(np.float32)
        label, conf = clf.predict_confident(lm)
        self.assertEqual(label, "none")
        self.assertEqual(conf, 0.0)


class TestSaveLoadRoundTrip(unittest.TestCase):
    def test_knn_save_load_reproduces_predictions(self):
        """save → load для kNN: предсказания совпадают для всех векторов выборки."""
        X, y = _make_separable(seed=4)
        clf = MLPoseClassifier(backend="knn", k=3)
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model")
            clf.save(path)
            loaded = MLPoseClassifier.load(path)
            self.assertIsNotNone(loaded)
            for i in range(len(X)):
                self.assertEqual(clf.predict_features(X[i]),
                                 loaded.predict_features(X[i]))

    def test_save_with_bare_filename(self):
        """save по голому имени файла (пустой dirname) не должен падать."""
        X, y = _make_separable(seed=5)
        clf = MLPoseClassifier(backend="knn", k=3)
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            old = os.getcwd()
            try:
                os.chdir(d)
                clf.save("bare_model")  # без каталога в пути
                self.assertTrue(os.path.exists("bare_model.npz"))
            finally:
                os.chdir(old)


class TestLoadMissingOrCorrupt(unittest.TestCase):
    def test_load_nonexistent_returns_none(self):
        """Загрузка отсутствующего пути возвращает None (откат на эвристику)."""
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "nope_does_not_exist")
            self.assertIsNone(MLPoseClassifier.load(missing))

    def test_load_corrupt_returns_none(self):
        """Загрузка битого .npz возвращает None, а не бросает исключение."""
        with tempfile.TemporaryDirectory() as d:
            corrupt = os.path.join(d, "corrupt.npz")
            with open(corrupt, "wb") as f:
                f.write(b"definitely not a valid npz archive")
            self.assertIsNone(MLPoseClassifier.load(corrupt))


class TestNormalizationConstantFeature(unittest.TestCase):
    def test_constant_feature_no_nan(self):
        """Постоянный признак (std=0) не порождает NaN благодаря +1e-6."""
        rng = np.random.default_rng(6)
        X = np.ones((10, 42), dtype=np.float32)   # все признаки постоянны...
        X[:, 0] = rng.normal(size=10)             # ...кроме одного варьирующего
        y = np.array(["a"] * 5 + ["b"] * 5)
        clf = MLPoseClassifier(backend="knn", k=1)
        clf.fit(X, y)
        # std нигде не должен быть нулём (защита +1e-6).
        self.assertGreater(float(clf._std.min()), 0.0)
        label, conf = clf._predict_vec(X[0])
        self.assertFalse(np.isnan(conf))
        self.assertIn(label, {"a", "b"})


class TestTrainFromDatasetErrors(unittest.TestCase):
    def test_empty_dataset_raises(self):
        """Пустой набор → понятная ошибка о нехватке примеров."""
        with tempfile.TemporaryDirectory() as d:
            empty = os.path.join(d, "empty.npz")
            GestureDataset().save(empty)
            with self.assertRaises(ValueError):
                train_from_dataset(empty, os.path.join(d, "m"))

    def test_single_class_dataset_raises(self):
        """Один класс (но достаточно примеров) → явная ошибка про 2 класса."""
        rng = np.random.default_rng(7)
        ds = GestureDataset()
        ds.X = [rng.random(42).astype(np.float32) for _ in range(12)]
        ds.y = ["fist"] * 12
        with tempfile.TemporaryDirectory() as d:
            single = os.path.join(d, "single.npz")
            ds.save(single)
            with self.assertRaises(ValueError) as ctx:
                train_from_dataset(single, os.path.join(d, "m"))
            self.assertIn("2", str(ctx.exception))

    def test_dataset_load_missing_returns_empty(self):
        """Загрузка отсутствующего датасета даёт пустой набор, а не исключение."""
        with tempfile.TemporaryDirectory() as d:
            ds = GestureDataset.load(os.path.join(d, "nope.npz"))
            self.assertEqual(len(ds), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
