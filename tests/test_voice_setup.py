"""Тесты помощника установки модели Vosk (stdlib unittest).

Запуск:  python -m unittest tests.test_voice_setup

Здесь проверяем чистую логику без звука, сети и mediapipe:
  * структурная валидация каталога: «похожий на Vosk» проходит, пустой/случайный — нет;
  * установка из локального каталога и из zip-архива (включая каталог-обёртку);
  * отказ на не-Vosk источнике, без оставления «битой» модели в dest;
  * vosk_model_status из recognizer для присутствующей и отсутствующей модели.

Реальная модель не нужна — собираем минимальную правдоподобную структуру
(am/final.mdl + conf/mfcc.conf) во временном каталоге.
"""

import os
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.config import VoiceConfig
from aircontrol.voice import recognizer as rec
from tools import setup_vosk


def _make_fake_model(root: str) -> str:
    """Собрать минимальную правдоподобную структуру модели Vosk в ``root``."""
    os.makedirs(os.path.join(root, "am"), exist_ok=True)
    os.makedirs(os.path.join(root, "conf"), exist_ok=True)
    with open(os.path.join(root, "am", "final.mdl"), "wb") as f:
        f.write(b"\x00fake-acoustic-model")
    with open(os.path.join(root, "conf", "mfcc.conf"), "w", encoding="utf-8") as f:
        f.write("--sample-frequency=16000\n")
    return root


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_real_looking_dir_passes(self):
        model = _make_fake_model(os.path.join(self.tmp, "model"))
        ok, reason = setup_vosk.looks_like_vosk_model(model)
        self.assertTrue(ok, reason)

    def test_empty_dir_fails(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        ok, _ = setup_vosk.looks_like_vosk_model(empty)
        self.assertFalse(ok)

    def test_random_dir_fails(self):
        rnd = os.path.join(self.tmp, "rnd")
        os.makedirs(os.path.join(rnd, "junk"))
        with open(os.path.join(rnd, "readme.txt"), "w", encoding="utf-8") as f:
            f.write("not a model")
        ok, _ = setup_vosk.looks_like_vosk_model(rnd)
        self.assertFalse(ok)

    def test_missing_conf_fails(self):
        partial = os.path.join(self.tmp, "partial")
        os.makedirs(os.path.join(partial, "am"))
        with open(os.path.join(partial, "am", "final.mdl"), "wb") as f:
            f.write(b"x")
        ok, _ = setup_vosk.looks_like_vosk_model(partial)
        self.assertFalse(ok)

    def test_nonexistent_dir_fails(self):
        ok, _ = setup_vosk.looks_like_vosk_model(os.path.join(self.tmp, "nope"))
        self.assertFalse(ok)


class InstallTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_install_from_directory(self):
        src = _make_fake_model(os.path.join(self.tmp, "src"))
        dest = os.path.join(self.tmp, "installed", "vosk-model")
        out = setup_vosk.install_model(src, dest)
        self.assertEqual(os.path.abspath(dest), out)
        self.assertTrue(setup_vosk.looks_like_vosk_model(dest)[0])

    def test_install_from_wrapper_directory(self):
        # Источник — каталог-обёртка, модель лежит на уровень глубже.
        wrapper = os.path.join(self.tmp, "download")
        _make_fake_model(os.path.join(wrapper, "vosk-model-small-ru-0.22"))
        dest = os.path.join(self.tmp, "installed")
        out = setup_vosk.install_model(wrapper, dest)
        self.assertTrue(setup_vosk.looks_like_vosk_model(out)[0])

    def test_install_from_zip(self):
        model = _make_fake_model(os.path.join(self.tmp, "model"))
        zip_path = os.path.join(self.tmp, "model.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for base, _dirs, files in os.walk(model):
                for name in files:
                    full = os.path.join(base, name)
                    arc = os.path.join(
                        "vosk-model-small-ru-0.22",
                        os.path.relpath(full, model),
                    )
                    zf.write(full, arc)
        dest = os.path.join(self.tmp, "installed")
        out = setup_vosk.install_model(zip_path, dest)
        self.assertTrue(setup_vosk.looks_like_vosk_model(out)[0])

    def test_install_rejects_non_model(self):
        bad = os.path.join(self.tmp, "bad")
        os.makedirs(bad)
        with open(os.path.join(bad, "hello.txt"), "w", encoding="utf-8") as f:
            f.write("nope")
        dest = os.path.join(self.tmp, "installed")
        with self.assertRaises(ValueError):
            setup_vosk.install_model(bad, dest)
        # «Битая» модель не должна остаться в dest.
        self.assertFalse(os.path.exists(dest))

    def test_install_missing_source_raises(self):
        with self.assertRaises(FileNotFoundError):
            setup_vosk.install_model(
                os.path.join(self.tmp, "ghost"),
                os.path.join(self.tmp, "installed"),
            )


class StatusTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_status_present_model_ok(self):
        model = _make_fake_model(os.path.join(self.tmp, "vosk-model"))
        cfg = VoiceConfig(engine="vosk", vosk_model_path=model)
        ok, detail = rec.vosk_model_status(cfg)
        self.assertTrue(ok, detail)

    def test_status_missing_model_not_ok(self):
        cfg = VoiceConfig(
            engine="vosk",
            vosk_model_path=os.path.join(self.tmp, "absent"),
        )
        ok, detail = rec.vosk_model_status(cfg)
        self.assertFalse(ok)
        self.assertIn("unavailable", detail)

    def test_status_invalid_model_not_ok(self):
        junk = os.path.join(self.tmp, "junk")
        os.makedirs(junk)
        cfg = VoiceConfig(engine="vosk", vosk_model_path=junk)
        ok, detail = rec.vosk_model_status(cfg)
        self.assertFalse(ok)
        self.assertIn("invalid", detail)


if __name__ == "__main__":
    unittest.main()
