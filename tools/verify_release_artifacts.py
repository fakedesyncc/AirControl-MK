"""Verify release artifacts produced by the GitHub Actions workflow.

The frozen smoke-test checks ``dist/AirControl`` before packaging. This script
checks the files that users actually download: zip/tar archives, AppImage, and
the Windows installer output.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _require_file(path: Path, min_size: int = 1) -> None:
    if not path.exists():
        _fail(f"missing artifact: {path}")
    if path.stat().st_size < min_size:
        _fail(f"artifact is unexpectedly small: {path} ({path.stat().st_size} bytes)")


def _check_zip(path: Path, required: list[str]) -> None:
    _require_file(path, min_size=1024)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    missing = [name for name in required if name not in names]
    if missing:
        _fail(f"{path.name} misses entries: {missing}")
    print(f"zip OK: {path}")


def _check_zip_flac_policy(path: Path, os_name: str) -> None:
    with zipfile.ZipFile(path) as zf:
        entries = zf.namelist()
        found = _check_speech_flac_policy(entries, os_name, path.name)
        _check_native_helper(entries, os_name, path.name)
    print(f"SpeechRecognition FLAC policy OK: {path.name} ({sorted(found)})")


def _check_tar_gz(path: Path, required: list[str]) -> None:
    _require_file(path, min_size=1024)
    with tarfile.open(path, "r:gz") as tf:
        names = set(tf.getnames())
    missing = [name for name in required if name not in names]
    if missing:
        _fail(f"{path.name} misses entries: {missing}")
    print(f"tar.gz OK: {path}")


def _check_tar_flac_policy(path: Path, os_name: str) -> None:
    with tarfile.open(path, "r:gz") as tf:
        entries = tf.getnames()
        found = _check_speech_flac_policy(entries, os_name, path.name)
        _check_native_helper(entries, os_name, path.name)
    print(f"SpeechRecognition FLAC policy OK: {path.name} ({sorted(found)})")


def _check_deb(path: Path, required: list[str]) -> None:
    _require_file(path, min_size=1024 * 1024)
    if platform.system() != "Linux":
        print(f"deb exists: {path}")
        return
    result = subprocess.run(
        ["dpkg-deb", "--contents", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    missing = [name for name in required if name not in result.stdout]
    if missing:
        _fail(f"{path.name} misses entries: {missing}")
    entries = _deb_paths_from_contents(result.stdout)
    found = _check_speech_flac_policy(entries, "Linux", path.name)
    _check_native_helper(entries, "Linux", path.name)
    print(f"SpeechRecognition FLAC policy OK: {path.name} ({sorted(found)})")
    print(f"deb OK: {path}")


def _run_appimage(path: Path) -> None:
    _require_file(path, min_size=1024)
    env = os.environ.copy()
    env["APPIMAGE_EXTRACT_AND_RUN"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run([str(path), "--help"], cwd=ROOT, env=env, check=True, timeout=60)
    subprocess.run(
        [str(path), "support", "--no-camera", "-o", str(ROOT / "aircontrol-appimage-smoke.zip")],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=60,
    )
    (ROOT / "aircontrol-appimage-smoke.zip").unlink(missing_ok=True)
    print(f"AppImage OK: {path}")


def _deb_paths_from_contents(contents: str) -> list[str]:
    paths: list[str] = []
    for line in contents.splitlines():
        parts = line.split()
        if parts:
            paths.append(parts[-1])
    return paths


def _check_speech_flac_policy(entries: list[str], os_name: str, artifact_name: str) -> set[str]:
    found = _speech_flac_names(entries)
    expected = _expected_speech_flac_names(os_name)
    if found != expected:
        _fail(
            f"{artifact_name} has unexpected SpeechRecognition FLAC converters: "
            f"{sorted(found)}; expected: {sorted(expected)}"
        )
    return found


def _speech_flac_names(entries: list[str]) -> set[str]:
    names: set[str] = set()
    for entry in entries:
        normalized = entry.replace("\\", "/")
        if "/speech_recognition/" not in normalized:
            continue
        name = normalized.rsplit("/", 1)[-1]
        if re.fullmatch(r"flac-[A-Za-z0-9_.-]+", name):
            names.add(name)
    return names


def _expected_speech_flac_names(os_name: str) -> set[str]:
    if os_name == "macOS":
        return set()
    if os_name == "Windows":
        return {"flac-win32.exe"}
    if os_name == "Linux":
        machine = platform.machine()
        if machine in {"i686", "i786", "x86"}:
            return {"flac-linux-x86"}
        if machine in {"x86_64", "AMD64"}:
            return {"flac-linux-x86_64"}
    return set()


def _check_native_helper(entries: list[str], os_name: str, artifact_name: str) -> None:
    expected = "aircontrol-helper.exe" if os_name == "Windows" else "aircontrol-helper"
    for entry in entries:
        if entry.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] == expected:
            return
    _fail(f"{artifact_name} misses native helper: {expected}")


def _verify_macos(root: Path) -> None:
    path = root / "AirControl-macOS.zip"
    _check_zip(path, [
        "AirControl/AirControl",
        "AirControl/USER_GUIDE_RU.txt",
        "AirControl.app/Contents/Info.plist",
    ])
    _check_zip_flac_policy(path, "macOS")


def _verify_windows(root: Path) -> None:
    path = root / "AirControl-Windows.zip"
    _check_zip(path, [
        "AirControl/AirControl.exe",
        "AirControl/USER_GUIDE_RU.txt",
    ])
    _check_zip_flac_policy(path, "Windows")
    _require_file(root / "installer" / "AirControl-Setup.exe", min_size=1024 * 1024)
    print(f"installer OK: {root / 'installer' / 'AirControl-Setup.exe'}")


def _verify_linux(root: Path, run_appimage: bool) -> None:
    path = root / "AirControl-Linux.tar.gz"
    _check_tar_gz(path, [
        "AirControl/AirControl",
        "AirControl/USER_GUIDE_RU.txt",
        "AirControl/AirControl.desktop",
    ])
    _check_tar_flac_policy(path, "Linux")
    appimage = root / "AirControl-Linux-x86_64.AppImage"
    if run_appimage:
        _run_appimage(appimage)
    else:
        _require_file(appimage, min_size=1024 * 1024)
        print(f"AppImage exists: {appimage}")
    _check_deb(root / "AirControl-Linux-amd64.deb", [
        "/opt/aircontrol/AirControl/AirControl",
        "/opt/aircontrol/AirControl/USER_GUIDE_RU.txt",
        "/usr/share/applications/aircontrol.desktop",
        "/usr/share/icons/hicolor/scalable/apps/aircontrol.svg",
        "/usr/bin/aircontrol",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="artifact root directory")
    parser.add_argument("--os", choices=["macOS", "Windows", "Linux"], default=None)
    parser.add_argument("--skip-appimage-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    os_name = args.os or {
        "Darwin": "macOS",
        "Windows": "Windows",
        "Linux": "Linux",
    }.get(platform.system())
    if os_name == "macOS":
        _verify_macos(root)
    elif os_name == "Windows":
        _verify_windows(root)
    elif os_name == "Linux":
        _verify_linux(root, run_appimage=not args.skip_appimage_run)
    else:
        _fail(f"unsupported platform for artifact verification: {platform.system()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"artifact verification failed: {exc}", file=sys.stderr)
        raise
