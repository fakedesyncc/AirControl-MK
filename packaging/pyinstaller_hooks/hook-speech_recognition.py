"""PyInstaller hook for the product bundle.

SpeechRecognition ships platform-specific ``flac-*`` converter executables.
The bundled macOS converter is built with an obsolete SDK and blocks clean
codesigning/notarization, so macOS builds intentionally rely on a system FLAC
converter for optional Google voice recognition. Windows/Linux builds keep only
the converter that matches the target platform.
"""

import platform
import sys

from PyInstaller.utils.hooks import collect_data_files


def _excluded_flac_files() -> list[str]:
    if sys.platform == "darwin":
        return ["flac-*"]
    if sys.platform.startswith("win"):
        return ["flac-mac", "flac-linux-*"]
    if sys.platform.startswith("linux"):
        machine = platform.machine()
        if machine in {"i686", "i786", "x86"}:
            return ["flac-mac", "flac-win32.exe", "flac-linux-x86_64"]
        if machine in {"x86_64", "AMD64"}:
            return ["flac-mac", "flac-win32.exe", "flac-linux-x86"]
        return ["flac-*"]
    return ["flac-*"]


datas = collect_data_files("speech_recognition", excludes=_excluded_flac_files())
