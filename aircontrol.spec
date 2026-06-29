# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller-спека AirControl (кросс-платформенная).

Собирает standalone-приложение. Запускать НА КАЖДОЙ ОС отдельно (PyInstaller не
умеет кросс-компиляцию):
    pyinstaller aircontrol.spec

Особенности:
  * product-бандл исключает research/optional пакеты (torch, pandas, notebooks);
  * MediaPipe data/libs собираются без импорта всех optional submodules;
  * matplotlib заменяется runtime-stub'ом: MediaPipe импортирует pyplot, но
    AirControl не использует matplotlib в пользовательском GUI;
  * модель hand_landmarker.task кладётся в корень бандла (config находит её по _MEIPASS);
  * onedir-сборка (надёжнее для mediapipe); на macOS дополнительно .app.
"""

import os
import sys
import importlib.util
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

datas, binaries, hiddenimports = [], [], []

# MediaPipe Tasks needs package resources and native libraries, but collect_all()
# imports every optional submodule and can drag torch/jax/notebooks into the GUI app.
for pkg in ("mediapipe",):
    try:
        datas += collect_data_files(pkg)
        binaries += collect_dynamic_libs(pkg)
        datas += copy_metadata(pkg)
    except Exception as exc:  # пакет может отсутствовать — не критично
        print(f"[spec] collect пропущен для {pkg}: {exc}")

for pkg in ("pynput", "SpeechRecognition", "opencv-python", "numpy", "Pillow"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Модель руки — в корень бандла.
datas += [("hand_landmarker.task", ".")]
datas += [("packaging/USER_GUIDE_RU.txt", ".")]
datas += [("LICENSE", ".")]
datas += [("NOTICE", ".")]


def _native_helper_binary():
    name = "aircontrol-helper.exe" if sys.platform.startswith("win") else "aircontrol-helper"
    path = os.path.join("bin", name)
    if os.path.exists(path):
        return [(path, ".")]
    print("[spec] native helper not found; bundle will use Python diagnostics only")
    return []


binaries += _native_helper_binary()

# Бэкенды pynput и прочее, что PyInstaller не находит автоматически.
hiddenimports += [
    "PIL._tkinter_finder", "PIL.ImageTk",
    "tkinter.filedialog", "tkinter.messagebox",
    "aircontrol.ui.calibration",
]
if importlib.util.find_spec("pynput") is not None:
    hiddenimports += [
        "pynput.keyboard._darwin", "pynput.mouse._darwin",
        "pynput.keyboard._win32", "pynput.mouse._win32",
        "pynput.keyboard._xorg", "pynput.mouse._xorg",
    ]

product_excludes = [
    "pytest", "torch", "jax", "jaxlib", "sentencepiece", "tensorflow",
    "IPython", "notebook", "jupyter", "nbformat", "pandas", "pyarrow",
    "numba", "llvmlite", "boto3", "botocore", "dask",
    "matplotlib", "sklearn", "scipy", "joblib",
    "onnxruntime", "av", "PyQt6", "tokenizers", "hf_xet", "ctranslate2",
    "Cython", "pocketsphinx", "pydantic", "pydantic_core", "lxml", "zmq",
    "aiohttp", "fastapi", "uvicorn", "mcp", "starlette", "httpx",
    "faster_whisper", "whisper", "groq",
]


def _without_speech_recognition_flac(entries):
    """Drop non-target or problematic SpeechRecognition FLAC executables."""
    keep = _speech_recognition_flac_to_keep()
    filtered = []
    for entry in entries:
        dest = entry[0] if entry else ""
        src = entry[1] if len(entry) > 1 else ""
        name = os.path.basename(str(src)) or os.path.basename(str(dest))
        if (
            (str(dest).startswith("speech_recognition/flac-") or name.startswith("flac-"))
            and name != keep
        ):
            continue
        filtered.append(entry)
    return filtered


def _speech_recognition_flac_to_keep():
    if sys.platform == "darwin":
        return None
    if sys.platform.startswith("win"):
        return "flac-win32.exe"
    if sys.platform.startswith("linux"):
        import platform
        machine = platform.machine()
        if machine in {"i686", "i786", "x86"}:
            return "flac-linux-x86"
        if machine in {"x86_64", "AMD64"}:
            return "flac-linux-x86_64"
    return None

a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["packaging/pyinstaller_hooks"],
    runtime_hooks=["packaging/pyinstaller_runtime_hooks/matplotlib_stub.py"],
    excludes=product_excludes,
    noarchive=False,
)
a.binaries = _without_speech_recognition_flac(a.binaries)
a.datas = _without_speech_recognition_flac(a.datas)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AirControl",
    debug=False,
    strip=False,
    upx=False,
    console=False,      # GUI-приложение без окна терминала
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="AirControl",
)

# На macOS дополнительно собираем .app-бандл.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="AirControl.app",
        icon=None,
        bundle_identifier="dev.aircontrol.app",
        info_plist={
            "NSCameraUsageDescription": "AirControl использует камеру для распознавания жестов рук.",
            "NSMicrophoneUsageDescription": "AirControl использует микрофон для голосовых команд.",
            "NSHighResolutionCapable": True,
        },
    )
