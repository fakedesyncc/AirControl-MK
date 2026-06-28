"""Runtime diagnostics for tester machines."""

from __future__ import annotations

import importlib
import json
import os
import platform as py_platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import List

from . import __app_name__, __version__
from .config import AppConfig, DATA_DIR


LINUX_TOOLS = [
    "wpctl", "pactl", "amixer",
    "grim", "gnome-screenshot", "spectacle", "scrot", "import",
    "xdotool", "ydotool", "ydotoold", "wmctrl", "gtk-launch", "xdg-open", "v4l2-ctl",
]


def build_report(scan_camera: bool = True, camera_limit: int | None = None,
                 input_probe: bool = False) -> str:
    cfg = AppConfig.load()
    lines: List[str] = []
    recommendations: List[str] = []

    lines.append("=== AirControl doctor ===")
    lines.append(f"OS: {py_platform.platform()}")
    lines.append(f"Python: {sys.version.split()[0]} ({sys.executable})")
    lines.append(f"Frozen bundle: {bool(getattr(sys, 'frozen', False))}")
    _append_system_resources(lines)
    lines.append("")
    _append_native_helper(lines, recommendations)
    lines.append("")

    _append_module(lines, "OpenCV", "cv2")
    _append_module(lines, "MediaPipe", "mediapipe")
    _append_module(lines, "NumPy", "numpy")
    _append_module(lines, "Pillow", "PIL")
    _append_module(lines, "Pynput", "pynput")
    _append_voice(lines, recommendations)
    _append_tk(lines, recommendations)
    _append_model(lines, cfg, recommendations)
    _append_runtime_config(lines, cfg)
    lines.append(f"Configured dry-input: {'ON' if cfg.input.dry_run else 'OFF'}")
    _append_input(lines, recommendations, probe_mouse=input_probe)

    if sys.platform.startswith("linux"):
        _append_linux(lines, recommendations)
    elif sys.platform == "darwin":
        _append_macos(lines, recommendations)
    elif sys.platform.startswith("win"):
        _append_windows(lines, recommendations)

    if scan_camera:
        limit = camera_limit if camera_limit is not None else getattr(cfg.camera, "scan_indices", 4)
        found = _append_camera(lines, cfg, max(1, int(limit)))
        if not found:
            recommendations.append(
                "Камера не найдена: проверьте /dev/video*, закройте приложения, занявшие камеру, "
                "и добавьте пользователя в группу video: sudo usermod -a -G video $USER"
            )
    else:
        lines.append("")
        lines.append("Camera scan: skipped")

    if recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for item in dict.fromkeys(recommendations):
            lines.append(f"- {item}")
    else:
        lines.append("")
        lines.append("Recommendations: явных проблем не найдено.")

    return "\n".join(lines)


def save_support_bundle(path: str | None = None, scan_camera: bool = False,
                        runtime_info: dict | None = None,
                        input_probe: bool = False) -> str:
    cfg = AppConfig.load()
    os.makedirs(DATA_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(DATA_DIR, f"aircontrol-support-{time.strftime('%Y%m%d-%H%M%S')}.zip")

    report = build_report(scan_camera=scan_camera, input_probe=input_probe)
    native_report = native_helper_report()
    doctor_summary = summarize_doctor_report(report)
    manifest = build_support_manifest(report, runtime_info, native_report)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", build_support_readme(manifest))
        zf.writestr("support-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("doctor.txt", report)
        zf.writestr("doctor-summary.txt", "\n".join(doctor_summary) + "\n")
        zf.writestr("config.json", json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
        if native_report is not None:
            zf.writestr(
                "native-helper.json",
                json.dumps(_public_native_helper_report(native_report), ensure_ascii=False, indent=2),
            )
        if runtime_info is not None:
            runtime_payload = dict(runtime_info)
            runtime_payload["summary"] = summarize_runtime(runtime_info)
            zf.writestr("runtime.json", json.dumps(runtime_payload, ensure_ascii=False, indent=2))
            zf.writestr("runtime-summary.txt", "\n".join(runtime_payload["summary"]) + "\n")
        _add_recent_logs(zf)
    return path


def build_support_manifest(report: str, runtime_info: dict | None = None,
                           native_report: dict | None = None) -> dict:
    """Build a small machine-readable index for the support ZIP."""
    files = [
        {"path": "README.txt", "purpose": "human-readable guide for this support bundle"},
        {"path": "doctor-summary.txt", "purpose": "short system readiness summary"},
        {"path": "doctor.txt", "purpose": "full system diagnostics"},
        {"path": "config.json", "purpose": "AirControl runtime configuration"},
    ]
    if runtime_info is not None:
        files.extend([
            {"path": "runtime-summary.txt", "purpose": "short live camera/control status"},
            {"path": "runtime.json", "purpose": "full live runtime status"},
        ])
    if native_report is not None:
        files.append({
            "path": "native-helper.json",
            "purpose": "native OS/session probe from the bundled AirControl helper",
        })
    files.append({"path": "logs/", "purpose": "recent telemetry logs, if present"})
    return {
        "app": __app_name__,
        "version": __version__,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": py_platform.platform(),
        "python": sys.version.split()[0],
        "frozen_bundle": bool(getattr(sys, "frozen", False)),
        "camera_scan_included": "Camera scan: skipped" not in report,
        "runtime_included": runtime_info is not None,
        "native_helper_included": native_report is not None,
        "files": files,
    }


def build_support_readme(manifest: dict) -> str:
    if manifest.get("runtime_included"):
        runtime_note = (
            "2. Откройте runtime-summary.txt: он показывает состояние камеры, FPS, "
            "Control path, Safe/Input, dwell-click, последнюю команду и runtime-проблемы.\n"
        )
        details_note = "3. Если summary недостаточно, откройте doctor.txt и runtime.json.\n"
    else:
        runtime_note = "2. runtime-summary.txt отсутствует: отчёт сохранён без работающего окна камеры.\n"
        details_note = "3. Если summary недостаточно, откройте doctor.txt.\n"
    return (
        "AirControl support bundle\n"
        "=========================\n\n"
        f"Приложение: {manifest.get('app', 'AirControl')} {manifest.get('version', '')}\n"
        f"Создано: {manifest.get('created_at', 'unknown')}\n"
        f"Платформа: {manifest.get('platform', 'unknown')}\n"
        f"Frozen bundle: {manifest.get('frozen_bundle')}\n\n"
        "Как читать отчёт:\n"
        "1. Сначала откройте doctor-summary.txt: это короткий вывод о готовности системы "
        "и блок «Что сделать дальше».\n"
        f"{runtime_note}"
        f"{details_note}"
        "4. config.json помогает понять профиль, камеру, Safe/Dwell и настройки производительности.\n\n"
        "Обычные причины проблем:\n"
        "- DRY INPUT означает безопасный режим: клики и клавиши намеренно отключены.\n"
        "- INPUT OFF/RISK/ERROR означает, что ОС может блокировать управление мышью/клавиатурой.\n"
        "- input probe показывает, проверялось ли безопасное движение курсора без кликов.\n"
        "- Low FPS/Slow detection обычно связаны с производительностью, освещением или камерой.\n"
        "- Light mode ON означает, что AirControl уже снизил нагрузку и мог ограничить FPS детекции.\n"
    )


def summarize_doctor_report(report: str) -> List[str]:
    """Build a short readiness summary from the text doctor report."""
    issues: List[str] = []
    notes: List[str] = []

    critical_checks = [
        ("OpenCV", "camera capture and image processing"),
        ("MediaPipe", "hand detection"),
        ("NumPy", "gesture math"),
        ("Pillow", "camera preview UI"),
        ("Tkinter", "no-console windows"),
        ("Hand model", "hand landmark model"),
    ]
    for label, meaning in critical_checks:
        if f"{label}: FAIL" in report:
            issues.append(f"{label} is not ready: {meaning} may not work.")

    if "Pynput: FAIL" in report:
        issues.append("Pynput is missing: bundled mouse/keyboard control is not available.")
    if "input backend: FAIL" in report:
        issues.append("OS input backend is unavailable: gestures may be detected but cannot control the computer.")
    if "input backend: WARN" in report:
        issues.append("OS input backend needs attention: gestures may be detected while clicks/keys are blocked by the session.")
    if "input probe: FAIL" in report or "input mouse move probe: FAIL" in report:
        issues.append("OS input probe failed: AirControl could not confirm safe mouse movement.")
    elif "input probe: WARN" in report or "input mouse move probe: SKIPPED" in report:
        notes.append("OS input probe is inconclusive: backend exists, but real cursor movement was not fully verified.")
    if "Camera scan: skipped" in report:
        notes.append("Camera was not opened during this check.")
    elif "Camera scan:" in report and ": OK frame=" not in report:
        issues.append("No working camera was found during the scan.")

    if "Display server: wayland" in report:
        notes.append("Linux Wayland may require ydotool/ydotoold or an Xorg session for full control.")
    if "PyAudio microphone backend: missing" in report:
        notes.append("Voice commands are disabled, but gestures and dwell-click can still work.")
    if "SpeechRecognition FLAC converter: missing" in report:
        notes.append("Online Google voice commands are disabled until a FLAC converter is available.")
    if "Frozen bundle: True" not in report:
        notes.append("This check is running from source, not from the final packaged app.")

    lines = ["=== AirControl readiness summary ==="]
    if issues:
        lines.append("Status: needs attention before assistive control.")
        lines.append("Issues:")
        lines.extend(f"- {issue}" for issue in dict.fromkeys(issues))
    else:
        lines.append("Status: ready for safe training. Use Assistive Control after camera and hand detection look stable.")
    if notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in dict.fromkeys(notes))
    next_steps = _doctor_next_steps(report, has_issues=bool(issues))
    if next_steps:
        lines.append("Что сделать дальше:")
        lines.extend(f"- {step}" for step in dict.fromkeys(next_steps))
    return lines


def _doctor_next_steps(report: str, *, has_issues: bool) -> List[str]:
    """Return non-technical next actions for the launcher and support ZIP."""
    steps: List[str] = []
    if "Camera scan: skipped" in report:
        steps.append(
            "Нажмите «Безопасная тренировка (без кликов)»: камера проверится в окне, "
            "а AirControl не будет нажимать мышь или клавиатуру."
        )
    elif "Camera scan:" in report and ": OK frame=" not in report:
        steps.append(
            "Закройте Zoom/браузер/другие программы с камерой и повторите проверку; "
            "на Linux может понадобиться доступ пользователя к группе video."
        )

    if ("input backend: FAIL" in report or "input backend: WARN" in report
            or "input probe: FAIL" in report or "input mouse move probe: FAIL" in report):
        if "Wayland" in report or "Display server: wayland" in report:
            steps.append(
                "Для реального управления на Linux войдите в Xorg-сессию или попросите "
                "помощника настроить ydotoold с доступом к /dev/uinput."
            )
        else:
            steps.append(
                "Если рука распознаётся, но компьютер не реагирует, проверьте разрешения "
                "ОС на управление мышью/клавиатурой или сохраните ZIP-отчёт помощнику."
            )

    if "OpenCV: FAIL" in report or "MediaPipe: FAIL" in report or "Hand model: FAIL" in report:
        steps.append(
            "Используйте готовый установщик/бандл AirControl для вашей ОС или пересоберите "
            "релиз: сейчас не хватает компонентов камеры/распознавания."
        )
    if "Tkinter: FAIL" in report:
        steps.append(
            "GUI-окна недоступны: для обычного пользователя лучше взять готовый "
            "установщик AirControl, где оконная часть уже включена."
        )

    if "PyAudio microphone backend: missing" in report or "SpeechRecognition FLAC converter: missing" in report:
        steps.append(
            "Голосовые команды можно пропустить: жесты, курсор и dwell-click работают без микрофона/FLAC."
        )
    if "Frozen bundle: True" not in report:
        steps.append(
            "Для обычного пользователя отдавайте готовый AirControl-Setup.exe, AirControl.app, "
            ".deb или AppImage; Python нужен только разработчику."
        )
    if not has_issues:
        steps.append(
            "Если безопасная тренировка стабильна, переходите к «Начать ассистивное управление»."
        )
    return steps


def summarize_runtime(runtime_info: dict) -> List[str]:
    """Build a concise support-facing summary from live app status."""
    mode = str(runtime_info.get("mode", "unknown"))
    profile = str(runtime_info.get("profile", "unknown"))
    start_mode = runtime_info.get("start_mode")
    input_status = str(runtime_info.get("input_status", "unknown"))
    safe_input = _bool_or_none(runtime_info.get("safe_input"))
    dwell_enabled = _bool_or_none(runtime_info.get("dwell_enabled"))
    dwell_profile = str(runtime_info.get("dwell_profile", "unknown") or "unknown")
    last_action = str(runtime_info.get("last_action", "") or "")
    seconds_since_action = runtime_info.get("seconds_since_action")
    last_input_error = str(runtime_info.get("last_input_error", "") or "")
    seconds_since_input_error = runtime_info.get("seconds_since_input_error")
    input_error_count = runtime_info.get("input_error_count")
    seconds_in_mode = runtime_info.get("seconds_in_mode")
    fps = _float_or_none(runtime_info.get("fps"))
    detect_ms = _float_or_none(runtime_info.get("detect_ms"))
    hand_detected = runtime_info.get("hand_detected")
    seconds_since_hand = runtime_info.get("seconds_since_hand")
    seconds_since_frame = runtime_info.get("seconds_since_frame")
    health_lines = [str(line) for line in runtime_info.get("health_lines", []) if line]
    performance = runtime_info.get("performance") or {}

    lines = [
        "=== AirControl runtime summary ===",
        f"Profile: {profile}",
        f"Mode: {mode}",
        f"Control path: {_runtime_control_path(mode, input_status, safe_input)}",
        f"Safe input: {_on_off_unknown(safe_input)}",
        f"Dwell-click: {_on_off_unknown(dwell_enabled)}",
        f"Dwell profile: {dwell_profile}",
        f"Input: {input_status}",
        f"Hand detected: {hand_detected}",
        f"FPS: {fps if fps is not None else 'unknown'}",
        f"Detection: {detect_ms if detect_ms is not None else 'unknown'} ms",
    ]
    if start_mode:
        lines.append(f"Start mode: {start_mode}")
    if seconds_in_mode is not None:
        lines.append(f"Seconds in mode: {seconds_in_mode}")
    if last_action:
        action_age = _float_or_none(seconds_since_action)
        if action_age is None:
            lines.append(f"Last action: {last_action}")
        else:
            lines.append(f"Last action: {last_action} ({action_age}s ago)")
    else:
        lines.append("Last action: none")
    if last_input_error:
        error_age = _float_or_none(seconds_since_input_error)
        if error_age is None:
            lines.append(f"Last input error: {last_input_error}")
        else:
            lines.append(f"Last input error: {last_input_error} ({error_age}s ago)")
    if input_error_count is not None:
        lines.append(f"Input error count: {input_error_count}")
    if seconds_since_hand is not None:
        lines.append(f"Seconds since hand: {seconds_since_hand}")
    if seconds_since_frame is not None:
        lines.append(f"Seconds since frame: {seconds_since_frame}")
    if performance:
        lines.append(
            "Performance: "
            f"downscale={performance.get('detect_downscale', 'unknown')}, "
            f"max_detect_fps={performance.get('detect_max_fps', 'unknown')}, "
            f"landmarks={'ON' if performance.get('show_landmarks', True) else 'OFF'}, "
            f"effects={'ON' if (performance.get('show_particles', False) or performance.get('show_trail', False)) else 'OFF'}"
        )
    if runtime_info.get("auto_tuned"):
        lines.append("Auto tune: ON" + (" (deep)" if runtime_info.get("deep_auto_tuned") else ""))
        if runtime_info.get("low_perf_reason"):
            lines.append(f"Auto tune reason: {runtime_info.get('low_perf_reason')}")

    issues: List[str] = []
    if mode != "control":
        issues.append("View mode is active: gestures are shown for preview/training, but OS control is not sent.")
    if input_status == "DRY INPUT":
        issues.append("Safe input is ON: cursor, clicks and keys are intentionally disabled.")
    elif safe_input is True:
        issues.append("Safe input is ON: cursor, clicks and keys may be intentionally disabled.")
    elif input_status == "INPUT OFF":
        issues.append("OS input backend is unavailable: gestures may be detected but cannot control the computer.")
    elif input_status == "INPUT ERROR":
        issues.append("OS input execution failed recently: gestures may fire while clicks/keys are rejected by the OS.")
    elif input_status == "INPUT RISK":
        issues.append("OS input backend is risky in this session: gestures may be detected while clicks/keys are blocked.")
    if last_input_error:
        issues.append(f"Last OS input error: {last_input_error}")
    if profile == "assistive" and dwell_enabled is False:
        issues.append("Assistive profile has dwell-click OFF: users who cannot pinch may be unable to click reliably.")
    if seconds_since_frame is not None and _float_or_none(seconds_since_frame) is not None:
        if float(seconds_since_frame) > 2.0:
            issues.append("Camera frames stopped recently.")
    if mode == "control" and hand_detected is False:
        issues.append("Control mode is active, but no hand is currently detected.")
    if fps is not None and fps < 18.0:
        if runtime_info.get("auto_tuned") and _is_expected_capped_runtime_fps(fps, performance):
            issues.append("Light mode is active: detection FPS is intentionally capped to reduce CPU load.")
        else:
            issues.append("Low FPS: use the assistive/low preset, improve lighting, or reduce camera load.")
    if detect_ms is not None and detect_ms > 70.0:
        issues.append("Slow hand detection: lower resolution/downscale or improve lighting/background contrast.")
    issues.extend(line for line in health_lines if line not in issues)

    lines.append("")
    lines.append("Issues:")
    if issues:
        lines.extend(f"- {issue}" for issue in dict.fromkeys(issues))
    else:
        lines.append("- No obvious runtime issue reported.")
    return lines


def _is_expected_capped_runtime_fps(fps: float, performance: dict) -> bool:
    max_fps = _float_or_none(performance.get("detect_max_fps"))
    if max_fps is None or max_fps <= 0:
        return False
    return fps >= max_fps * 0.85


def _runtime_control_path(mode: str, input_status: str, safe_input: bool | None) -> str:
    if mode != "control":
        return "OFF (View mode: preview/training only)"
    if input_status == "DRY INPUT" or safe_input is True:
        return "OFF (Safe input)"
    if input_status == "INPUT OFF":
        return "OFF (input backend unavailable)"
    if input_status == "INPUT ERROR":
        return "ERROR (last OS input event failed)"
    if input_status == "INPUT RISK":
        return "RISK (OS session may block input)"
    if input_status.startswith("INPUT "):
        return "ON"
    return "unknown"


def _on_off_unknown(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "ON" if value else "OFF"


def _bool_or_none(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    return None


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_module(lines: List[str], label: str, module_name: str) -> None:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "installed")
        lines.append(f"{label}: OK ({version})")
    except Exception as exc:
        lines.append(f"{label}: FAIL ({exc})")


def _append_system_resources(lines: List[str]) -> None:
    lines.append("System resources:")
    lines.append(f"- CPU: {_cpu_summary()}")
    lines.append(f"- CPU cores: {os.cpu_count() or 'unknown'}")
    lines.append(f"- Memory: {_memory_summary()}")


def _cpu_summary() -> str:
    if sys.platform == "darwin":
        value = _run_short(["sysctl", "-n", "machdep.cpu.brand_string"])
        if value:
            return value
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith(("model name", "hardware")) and ":" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return py_platform.processor() or py_platform.machine() or "unknown"


def _memory_summary() -> str:
    if sys.platform == "darwin":
        value = _run_short(["sysctl", "-n", "hw.memsize"])
        if value and value.isdigit():
            return _format_bytes(int(value))
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages and page_size:
            return _format_bytes(int(pages) * int(page_size))
    except Exception:
        pass
    return "unknown"


def _format_bytes(value: int) -> str:
    gib = value / (1024 ** 3)
    if gib >= 1:
        return f"{gib:.1f} GiB"
    mib = value / (1024 ** 2)
    return f"{mib:.0f} MiB"


def _run_short(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, text=True, capture_output=True, timeout=0.5)
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _append_native_helper(lines: List[str], recommendations: List[str]) -> None:
    report = native_helper_report()
    if report is None:
        lines.append("Native helper: missing (optional)")
        return

    helper_path = str(report.get("_helper_path") or "unknown")
    lines.append(
        f"Native helper: OK ({helper_path}; version={report.get('helper_version', 'unknown')})"
    )
    lines.append(f"Native helper OS: {report.get('os', 'unknown')}/{report.get('arch', 'unknown')}")
    lines.append(f"Native helper display: {report.get('display_server', 'unknown')}")

    devices = [str(item) for item in report.get("video_devices") or []]
    if devices:
        lines.append(f"Native helper video devices: {', '.join(devices)}")

    tools = report.get("tools") or []
    if isinstance(tools, list) and tools:
        summary = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "unknown")
            status = "OK" if tool.get("found") else "missing"
            summary.append(f"{name}={status}")
        if summary:
            lines.append(f"Native helper tools: {', '.join(summary)}")

    for item in report.get("recommendations") or []:
        recommendations.append(f"Native helper: {item}")


def native_helper_report() -> dict | None:
    """Return the optional native helper report, if the helper is available."""
    helper = _native_helper_path()
    if helper is None:
        return None
    try:
        result = subprocess.run(
            [str(helper), "doctor", "--json"],
            text=True,
            capture_output=True,
            timeout=2.0,
            check=True,
        )
        data = json.loads(result.stdout)
    except Exception:
        return None

    if isinstance(data, dict):
        data["_helper_path"] = str(helper)
        return data
    return None


def _public_native_helper_report(report: dict) -> dict:
    return {key: value for key, value in report.items() if not str(key).startswith("_")}


def _native_helper_path() -> Path | None:
    exe_name = "aircontrol-helper.exe" if sys.platform.startswith("win") else "aircontrol-helper"
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        candidates.append(executable_dir / exe_name)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass).resolve() / exe_name)

    project_dir = Path(__file__).resolve().parents[1]
    candidates.extend([
        project_dir / "bin" / exe_name,
        project_dir / exe_name,
    ])

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    found = shutil.which(exe_name)
    return Path(found) if found else None


def _append_tk(lines: List[str], recommendations: List[str]) -> None:
    try:
        import tkinter  # noqa: F401
        lines.append("Tkinter: OK")
    except Exception as exc:
        lines.append(f"Tkinter: FAIL ({exc})")
        recommendations.append("Установите Tkinter: sudo apt install python3-tk")


def _append_voice(lines: List[str], recommendations: List[str]) -> None:
    sr_ok = importlib.util.find_spec("speech_recognition") is not None
    pa_ok = importlib.util.find_spec("pyaudio") is not None
    lines.append(f"SpeechRecognition: {'OK' if sr_ok else 'missing'}")
    lines.append(f"PyAudio microphone backend: {'OK' if pa_ok else 'missing'}")
    flac_path = _speech_flac_converter_path() if sr_ok else None
    lines.append(f"SpeechRecognition FLAC converter: {'OK (' + flac_path + ')' if flac_path else 'missing'}")
    if sr_ok and not pa_ok:
        recommendations.append(
            "Голосовые команды отключены: PyAudio не установлен. Это не мешает "
            "управлению жестами/dwell-click. Для микрофона установите "
            "requirements-optional.txt и системный PortAudio."
        )
    if sr_ok and pa_ok and not flac_path:
        recommendations.append(
            "Онлайн-голосовые команды Google отключены: SpeechRecognition не нашёл "
            "FLAC-конвертер. Жесты и dwell-click работают без него; для голоса "
            "установите системный flac или используйте офлайн Vosk."
        )


def _speech_flac_converter_path() -> str | None:
    try:
        from .voice.recognizer import flac_converter_path
        return flac_converter_path()
    except Exception:
        return None


def _append_model(lines: List[str], cfg: AppConfig, recommendations: List[str]) -> None:
    model = cfg.tracking.model_path
    ok = os.path.exists(model)
    lines.append(f"Hand model: {'OK' if ok else 'FAIL'} ({model})")
    if not ok:
        recommendations.append("Проверьте, что hand_landmarker.task лежит рядом с проектом или включён в бандл.")


def _append_runtime_config(lines: List[str], cfg: AppConfig) -> None:
    lines.append("")
    lines.append("Runtime config:")
    lines.append(f"profile: {cfg.profile_name}")
    lines.append(f"start_mode: {cfg.start_mode}")
    lines.append(f"camera: {cfg.camera.width}x{cfg.camera.height} @ {cfg.camera.target_fps} fps")
    lines.append(f"camera_backend: {cfg.camera.backend}")
    lines.append(f"detect_downscale: {cfg.performance.detect_downscale}")
    lines.append(f"detect_max_fps: {cfg.performance.detect_max_fps}")
    lines.append(f"dwell_enabled: {cfg.cursor.dwell_enabled}")
    lines.append(f"dwell_profile: {getattr(cfg.cursor, 'dwell_profile', 'custom')}")
    lines.append(f"dwell_time: {cfg.cursor.dwell_time}")
    lines.append(f"dwell_radius: {cfg.cursor.dwell_radius}")
    lines.append(f"dynamic_enabled: {cfg.gestures.dynamic_enabled}")
    lines.append(f"bimanual_enabled: {cfg.gestures.bimanual_enabled}")


def _append_input(lines: List[str], recommendations: List[str],
                  probe_mouse: bool = False) -> None:
    try:
        ib = importlib.import_module("aircontrol.control.input_backend")
        probe = ib.probe_input_backend(move_mouse=probe_mouse)
        err = ib.input_backend_error()
        if err:
            lines.append(f"input backend: FAIL ({err})")
            if "No module named" in err and "pynput" in err:
                recommendations.append("Установите зависимости проекта: pip install -r requirements.txt")
            elif sys.platform.startswith("linux"):
                recommendations.append(
                    "Глобальное управление мышью/клавиатурой недоступно. "
                    "На Linux чаще всего помогает вход в Xorg-сессию; на Wayland "
                    "можно настроить ydotool/ydotoold через uinput."
                )
            else:
                recommendations.append("Проверьте разрешения ОС для управления мышью и клавиатурой.")
        else:
            warning = getattr(ib, "input_backend_warning", lambda: None)()
            if warning:
                lines.append(f"input backend: WARN ({ib.input_backend_name()}; {warning})")
                if sys.platform.startswith("linux"):
                    recommendations.append(
                        "Сессия Linux может блокировать глобальный ввод. Для полноценного "
                        "управления используйте Xorg или настройте ydotool/ydotoold с доступом "
                        "к /dev/uinput."
                    )
                else:
                    recommendations.append("Проверьте разрешения ОС для управления мышью и клавиатурой.")
            else:
                lines.append(f"input backend: OK ({ib.input_backend_name()})")
        lines.append(_format_input_probe_line(probe))
        mouse_line = _format_input_mouse_probe_line(probe)
        if mouse_line:
            lines.append(mouse_line)
    except Exception as exc:
        lines.append(f"input backend: FAIL ({exc})")


def _format_input_probe_line(probe: dict) -> str:
    status = probe.get("status", "unknown")
    backend = probe.get("backend", "unknown")
    detail = probe.get("detail") or "no detail"
    requested = "requested" if probe.get("mouse_move_requested") else "not requested"
    return f"input probe: {status} (backend={backend}; mouse_move={requested}; {detail})"


def _format_input_mouse_probe_line(probe: dict) -> str | None:
    if not probe.get("mouse_move_requested"):
        return None
    value = probe.get("mouse_move")
    if value is True:
        status = "OK"
    elif value is False:
        status = "FAIL"
    else:
        status = "SKIPPED"
    return f"input mouse move probe: {status} ({probe.get('mouse_detail') or 'no detail'})"


def _append_linux(lines: List[str], recommendations: List[str]) -> None:
    from .platform.linux import LinuxBackend

    backend = LinuxBackend()
    display = backend.display_server()
    lines.append("")
    lines.append("Linux:")
    lines.append(f"Display server: {display}")
    lines.append(f"DISPLAY: {os.environ.get('DISPLAY', '-')}")
    lines.append(f"WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY', '-')}")
    lines.append(f"XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE', '-')}")
    for tool in LINUX_TOOLS:
        lines.append(f"tool {tool}: {'OK' if shutil.which(tool) else 'missing'}")
    for warning in backend.startup_warnings():
        recommendations.append(warning)


def _append_macos(lines: List[str], recommendations: List[str]) -> None:
    lines.append("")
    lines.append("macOS permissions:")
    lines.append("- Camera: required for hand tracking")
    lines.append("- Microphone: required only for voice commands")
    lines.append("- Accessibility: required for cursor/keyboard control")
    lines.append("- Screen Recording: required only for screenshots/recording on newer macOS")
    recommendations.append(
        "macOS: откройте System Settings -> Privacy & Security и разрешите AirControl "
        "доступ к Camera, Microphone и Accessibility. После изменения разрешений "
        "перезапустите AirControl."
    )


def _append_windows(lines: List[str], recommendations: List[str]) -> None:
    lines.append("")
    lines.append("Windows permissions:")
    lines.append("- Camera privacy access: required for hand tracking")
    lines.append("- Microphone privacy access: required only for voice commands")
    lines.append("- Input control: can be limited by antivirus/security software")
    recommendations.append(
        "Windows: проверьте Settings -> Privacy & security -> Camera/Microphone. "
        "Если управление не работает, проверьте SmartScreen/антивирус и запустите "
        "сначала безопасную тренировку."
    )


def _append_camera(lines: List[str], cfg: AppConfig, limit: int) -> bool:
    lines.append("")
    lines.append(f"Camera scan: 0..{limit - 1}")
    try:
        import cv2
    except Exception as exc:
        lines.append(f"OpenCV camera scan: FAIL ({exc})")
        return False

    api = _camera_api(cv2, getattr(cfg.camera, "backend", "auto"))
    found = False
    for idx in range(limit):
        cap = cv2.VideoCapture(idx, api) if api != cv2.CAP_ANY else cv2.VideoCapture(idx)
        try:
            if getattr(cfg.camera, "fourcc", "") and len(cfg.camera.fourcc) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.camera.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
            ok_open = cap.isOpened()
            ok_read, frame = cap.read() if ok_open else (False, None)
            shape = getattr(frame, "shape", None)
            if ok_open and ok_read and frame is not None and frame.size > 0:
                found = True
                lines.append(f"camera[{idx}]: OK frame={shape}")
            elif ok_open:
                lines.append(f"camera[{idx}]: opened, no frame")
            else:
                lines.append(f"camera[{idx}]: not available")
        finally:
            cap.release()
    return found


def _camera_api(cv2, backend: str) -> int:
    backend = (backend or "auto").lower()
    if backend == "any":
        return cv2.CAP_ANY
    if backend == "v4l2":
        return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    if backend == "auto" and sys.platform.startswith("linux"):
        return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    return cv2.CAP_ANY


def _add_recent_logs(zf: zipfile.ZipFile, limit: int = 5) -> None:
    log_dir = os.path.join(DATA_DIR, "logs")
    if not os.path.isdir(log_dir):
        return
    entries = []
    for name in os.listdir(log_dir):
        path = os.path.join(log_dir, name)
        if os.path.isfile(path):
            entries.append((os.path.getmtime(path), path, name))
    for _, path, name in sorted(entries, reverse=True)[:limit]:
        try:
            zf.write(path, f"logs/{name}")
        except Exception:
            pass
