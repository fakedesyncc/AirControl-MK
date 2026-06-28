"""Smoke-test a PyInstaller AirControl build.

This script intentionally runs the frozen executable, not source code. It keeps
CI honest: if the end-user bundle misses MediaPipe, Tk, the hand model, or basic
diagnostics, the release job fails before artifacts are uploaded.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = Path(os.environ.get("AIRCONTROL_DIST_DIR", ROOT / "dist")).resolve()


def _exe_path() -> Path:
    name = "AirControl.exe" if sys.platform.startswith("win") else "AirControl"
    return DIST_ROOT / "AirControl" / name


def _assert_expected_bundled_speech_flac() -> None:
    app_dir = DIST_ROOT / "AirControl"
    matches = sorted(
        path for path in app_dir.rglob("flac-*")
        if "speech_recognition" in path.parts
    )
    names = {path.name for path in matches}
    expected = _expected_speech_flac_names()
    if names != expected:
        rel = [str(path.relative_to(app_dir)) for path in matches]
        raise RuntimeError(
            f"unexpected bundled SpeechRecognition FLAC converters: {rel}; "
            f"expected names: {sorted(expected)}"
        )


def _expected_speech_flac_names() -> set[str]:
    if sys.platform == "darwin":
        return set()
    if sys.platform.startswith("win"):
        return {"flac-win32.exe"}
    if sys.platform.startswith("linux"):
        machine = platform.machine()
        if machine in {"i686", "i786", "x86"}:
            return {"flac-linux-x86"}
        if machine in {"x86_64", "AMD64"}:
            return {"flac-linux-x86_64"}
    return set()


def _native_helper_name() -> str:
    return "aircontrol-helper.exe" if sys.platform.startswith("win") else "aircontrol-helper"


def _require_native_helper() -> bool:
    return os.environ.get("AIRCONTROL_REQUIRE_NATIVE_HELPER") == "1"


def _assert_native_helper_bundled() -> None:
    if not _require_native_helper():
        return
    app_dir = DIST_ROOT / "AirControl"
    matches = sorted(path for path in app_dir.rglob(_native_helper_name()) if path.is_file())
    if not matches:
        raise RuntimeError(f"native helper is missing from frozen bundle: {_native_helper_name()}")


def main() -> int:
    exe = _exe_path()
    if not exe.exists():
        print(f"Missing executable: {exe}", file=sys.stderr)
        return 1

    _assert_expected_bundled_speech_flac()
    _assert_native_helper_bundled()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    subprocess.run([str(exe), "--help"], cwd=ROOT, env=env, check=True, timeout=45)
    subprocess.run([str(exe), "selftest"], cwd=ROOT, env=env, check=True, timeout=60)

    bundle = ROOT / "aircontrol-support-smoke.zip"
    if bundle.exists():
        bundle.unlink()
    subprocess.run(
        [str(exe), "support", "--no-camera", "-o", str(bundle)],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=45,
    )

    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
        missing = {
            "README.txt",
            "support-manifest.json",
            "doctor.txt",
            "doctor-summary.txt",
            "config.json",
        } - names
        if _require_native_helper():
            missing -= {"native-helper.json"}
            if "native-helper.json" not in names:
                raise RuntimeError("support bundle misses native-helper.json")
        if missing:
            raise RuntimeError(f"support bundle misses files: {sorted(missing)}")
        manifest = zf.read("support-manifest.json").decode("utf-8", errors="replace")
        doctor = zf.read("doctor.txt").decode("utf-8", errors="replace")
        doctor_summary = zf.read("doctor-summary.txt").decode("utf-8", errors="replace")

    required = [
        "OpenCV: OK",
        "MediaPipe: OK",
        "Tkinter: OK",
        "Hand model: OK",
        "Frozen bundle: True",
        "input probe:",
    ]
    if not (sys.platform.startswith("linux") and "Display server: headless" in doctor):
        required.append("Pynput: OK")
    failed = [item for item in required if item not in doctor]
    if failed:
        print(doctor, file=sys.stderr)
        raise RuntimeError(f"frozen doctor check failed: {failed}")
    if _require_native_helper() and "Native helper: OK" not in doctor:
        print(doctor, file=sys.stderr)
        raise RuntimeError("frozen doctor did not detect the bundled native helper")
    if "AirControl readiness summary" not in doctor_summary:
        raise RuntimeError("doctor-summary.txt does not contain readiness summary")
    if "Что сделать дальше" not in doctor_summary:
        raise RuntimeError("doctor-summary.txt does not contain user-facing next steps")
    if '"app": "AirControl"' not in manifest:
        raise RuntimeError("support-manifest.json does not identify AirControl")

    bundle.unlink(missing_ok=True)
    print(f"Frozen smoke-test passed: {exe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
