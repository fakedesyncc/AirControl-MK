"""Помощник установки офлайн-модели Vosk для голосового ввода AirControl.

Голос в AirControl опционален и ориентирован на приватность. Онлайн-движок
Google отправляет звук в сеть; офлайн-движок Vosk работает локально, но
требует заранее скачанной модели. Этот скрипт НЕ скачивает модель из сети сам:
он лишь печатает понятные инструкции, а затем (по вашему указанию на локальный
каталог или zip) проверяет и размещает модель в нужном месте.

Использование::

    # Показать инструкции, куда и какую модель скачать:
    python -m tools.setup_vosk

    # Установить из уже скачанного каталога модели:
    python -m tools.setup_vosk --source /path/to/vosk-model-small-ru-0.22

    # ...или из zip-архива модели:
    python -m tools.setup_vosk --source vosk-model-small-en-us-0.15.zip

    # Переопределить каталог назначения (по умолчанию voice.vosk_model_path):
    python -m tools.setup_vosk --source ./model --dest /custom/path

Скрипт намеренно зависит только от стандартной библиотеки и НЕ выполняет
скрытых сетевых загрузок: реальные данные берутся исключительно из указанного
локального --source. Логика проверки и размещения вынесена в чистые функции,
чтобы её было удобно тестировать без звука, сети и mediapipe.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from typing import Optional

# Запуск как `python -m tools.setup_vosk` (пакет) и как `python tools/setup_vosk.py`
# (скрипт) — в обоих случаях корень репозитория должен быть на sys.path, чтобы
# импортировать конфиг приложения.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from aircontrol.config import DEFAULT_VOSK_MODEL_PATH  # noqa: E402

# Ссылки на официальную страницу моделей и пара популярных «small»-моделей.
# Печатаются только как подсказка — скрипт сам ничего не качает.
MODELS_PAGE = "https://alphacephei.com/vosk/models"
SUGGESTED_MODELS = (
    ("ru", "vosk-model-small-ru-0.22", "русский, компактная (~45 МБ)"),
    ("en", "vosk-model-small-en-us-0.15", "english, small (~40 МБ)"),
)


def looks_like_vosk_model(model_dir: str) -> tuple[bool, str]:
    """Поверхностно проверить, что каталог похож на распакованную модель Vosk.

    Это структурная эвристика, а не загрузка модели. У настоящей модели есть
    акустическая модель ``am/final.mdl`` и каталог ``conf/`` с ``mfcc.conf``
    либо ``model.conf``. Возвращает ``(ok, reason)``; при неуспехе ``reason``
    кратко поясняет, чего не хватает (удобно показать пользователю).
    """
    if not os.path.isdir(model_dir):
        return False, "не каталог"
    am_model = os.path.join(model_dir, "am", "final.mdl")
    if not os.path.isfile(am_model):
        return False, "нет am/final.mdl"
    conf_dir = os.path.join(model_dir, "conf")
    if not os.path.isdir(conf_dir):
        return False, "нет каталога conf/"
    has_conf = any(
        os.path.isfile(os.path.join(conf_dir, name))
        for name in ("mfcc.conf", "model.conf")
    )
    if not has_conf:
        return False, "нет conf/mfcc.conf или conf/model.conf"
    return True, "ok"


def find_model_root(base_dir: str) -> Optional[str]:
    """Найти корень модели внутри каталога.

    Архивы Vosk обычно распаковываются в один вложенный каталог
    (``vosk-model-small-ru-0.22/...``), поэтому модель может быть как в самом
    ``base_dir``, так и на один уровень глубже. Проверяем оба варианта и
    возвращаем путь к корню модели либо ``None``, если ничего подходящего нет.
    """
    if not os.path.isdir(base_dir):
        return None
    if looks_like_vosk_model(base_dir)[0]:
        return base_dir
    try:
        entries = sorted(os.listdir(base_dir))
    except OSError:
        return None
    for name in entries:
        candidate = os.path.join(base_dir, name)
        if os.path.isdir(candidate) and looks_like_vosk_model(candidate)[0]:
            return candidate
    return None


def extract_zip_to(zip_path: str, dest_dir: str) -> str:
    """Безопасно распаковать zip-архив модели в ``dest_dir`` и вернуть его путь.

    Защита от Zip Slip: записи, ведущие за пределы ``dest_dir`` (абсолютные
    пути или ``..``), отклоняются с ``ValueError``. Сетью не пользуемся —
    источник строго локальный.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_abs = os.path.abspath(dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = os.path.abspath(os.path.join(dest_abs, member))
            if target != dest_abs and not target.startswith(dest_abs + os.sep):
                raise ValueError(f"небезопасный путь в архиве: {member}")
        zf.extractall(dest_abs)
    return dest_abs


def install_model(source: str, dest: str) -> str:
    """Разместить модель Vosk из локального ``source`` в каталоге ``dest``.

    ``source`` — каталог модели (или каталог-обёртка с моделью внутри) либо
    zip-архив. После размещения каталог ``dest`` валидируется как настоящая
    модель Vosk; иначе бросается ``ValueError`` и частично скопированное
    удаляется, чтобы не оставлять «битую» модель.

    Возвращает абсолютный путь к установленной модели. Существующий ``dest``
    перезаписывается (это явная команда пользователя установить модель).
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"источник не найден: {source}")

    tmp_dir = tempfile.mkdtemp(prefix="vosk-setup-")
    try:
        if os.path.isfile(source) and zipfile.is_zipfile(source):
            staged = extract_zip_to(source, os.path.join(tmp_dir, "unzipped"))
            model_root = find_model_root(staged)
        elif os.path.isdir(source):
            model_root = find_model_root(source)
        else:
            raise ValueError(
                f"источник не является каталогом модели или zip-архивом: {source}"
            )

        if model_root is None:
            ok, reason = looks_like_vosk_model(source if os.path.isdir(source) else tmp_dir)
            raise ValueError(
                f"не похоже на модель Vosk: {reason}. "
                f"Ожидаются am/final.mdl и conf/."
            )

        dest_abs = os.path.abspath(dest)
        if os.path.exists(dest_abs):
            shutil.rmtree(dest_abs)
        os.makedirs(os.path.dirname(dest_abs) or ".", exist_ok=True)
        shutil.copytree(model_root, dest_abs)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    ok, reason = looks_like_vosk_model(dest_abs)
    if not ok:
        shutil.rmtree(dest_abs, ignore_errors=True)
        raise ValueError(f"установка не прошла проверку: {reason}")
    return dest_abs


def _print_instructions(dest: str) -> None:
    """Напечатать инструкции: где скачать модель и как затем её установить."""
    print("Настройка офлайн-движка Vosk (приватный голосовой ввод)")
    print("=" * 60)
    print()
    print("Vosk работает локально и ничего не отправляет в сеть, но требует")
    print("заранее скачанной модели. Этот скрипт сам модель НЕ скачивает.")
    print()
    print(f"1) Откройте страницу моделей: {MODELS_PAGE}")
    print("2) Скачайте модель для вашего языка, например:")
    for lang, name, note in SUGGESTED_MODELS:
        print(f"     [{lang}] {name}  — {note}")
    print("3) Установите её (из zip или распакованного каталога):")
    print("     python -m tools.setup_vosk --source <путь-к-модели-или-zip>")
    print()
    print(f"Каталог назначения по умолчанию: {dest}")
    print("(переопределяется флагом --dest)")
    print()
    print("Если модель не установлена, голос через Vosk просто недоступен —")
    print("приложение НЕ переключается на онлайн-движок Google молча.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setup_vosk",
        description="Установка офлайн-модели Vosk для голосового ввода AirControl.",
    )
    parser.add_argument(
        "--source",
        help="Локальный каталог модели Vosk или zip-архив. Без него печатаются инструкции.",
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_VOSK_MODEL_PATH,
        help="Куда установить модель (по умолчанию voice.vosk_model_path).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.source:
        _print_instructions(args.dest)
        return 0
    try:
        installed = install_model(args.source, args.dest)
    except (FileNotFoundError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"Ошибка установки модели: {exc}", file=sys.stderr)
        return 1
    print(f"Модель Vosk установлена: {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
