"""Simple no-console launcher for end users.

The packaged app starts here: large buttons, safe first action, no terminal.
Developer and tester commands remain available through ``python -m aircontrol``.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import Callable

from .config import AppConfig, apply_assistive_profile


def prepare_launch_config(
    cfg: AppConfig,
    *,
    assistive: bool = False,
    assistive_preset: str = "balanced",
    dry_input: bool = False,
    start_mode: str | None = None,
) -> AppConfig:
    """Apply launcher choices to a config object."""
    if assistive:
        apply_assistive_profile(cfg, assistive_preset)
    if start_mode is not None:
        cfg.start_mode = start_mode
    elif assistive:
        cfg.start_mode = "control"
    cfg.input.dry_run = dry_input
    return cfg


def run_launcher() -> None:
    root = tk.Tk()
    root.title("AirControl")
    root.geometry("680x720")
    root.minsize(560, 640)

    bg = "#101418"
    panel = "#182028"
    text = "#f3f6f8"
    muted = "#a9b4bf"
    accent = "#42d392"
    warning = "#ffd166"
    root.configure(bg=bg)

    container = tk.Frame(root, bg=bg, padx=28, pady=24)
    container.pack(fill=tk.BOTH, expand=True)

    tk.Label(container, text="AirControl", bg=bg, fg=text,
             font=("TkDefaultFont", 28, "bold")).pack(anchor="w")
    tk.Label(container, text="Бесконтактное управление компьютером",
             bg=bg, fg=muted, font=("TkDefaultFont", 14)).pack(anchor="w", pady=(4, 22))

    status = tk.StringVar(value="Сначала используйте безопасную тренировку: клики и клавиши отключены.")
    tk.Label(container, textvariable=status, bg=panel, fg=warning, justify="left",
             wraplength=530, padx=14, pady=12, font=("TkDefaultFont", 12)).pack(fill=tk.X)

    buttons = tk.Frame(container, bg=bg)
    buttons.pack(fill=tk.BOTH, expand=True, pady=(22, 0))

    def launch_app(
        assistive: bool,
        dry_input: bool,
        start_mode: str | None = None,
        assistive_preset: str = "balanced",
    ) -> None:
        cfg = prepare_launch_config(
            AppConfig.load(),
            assistive=assistive,
            assistive_preset=assistive_preset,
            dry_input=dry_input,
            start_mode=start_mode,
        )
        if _requires_control_preflight(cfg):
            ok = _confirm_control_preflight(
                root,
                status_var=status,
                diagnostics_callback=show_diagnostics,
            )
            if not ok:
                return
        _launch_aircontrol_from_launcher(root, cfg)

    def run_calibration() -> None:
        _run_calibration_from_launcher(root, AppConfig.load())

    def save_report() -> None:
        from .diagnostics import save_support_bundle
        default_path = os.path.join(os.path.expanduser("~"), "aircontrol-support.zip")
        path = filedialog.asksaveasfilename(
            title="Сохранить отчёт поддержки",
            initialfile=os.path.basename(default_path),
            defaultextension=".zip",
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_support_bundle(path, scan_camera=False)
            messagebox.showinfo("AirControl", f"Отчёт поддержки сохранён:\n{path}")
        except Exception as exc:
            messagebox.showerror("AirControl", f"Не удалось сохранить отчёт поддержки:\n{exc}")

    def show_first_run_wizard() -> None:
        win = tk.Toplevel(root)
        win.title("AirControl - первый запуск")
        win.geometry("760x620")
        win.minsize(620, 500)
        win.configure(bg=bg)

        tk.Label(
            win,
            text="Мастер первого запуска",
            bg=bg,
            fg=text,
            font=("TkDefaultFont", 20, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 6))
        tk.Label(
            win,
            text="Проверьте камеру, ввод ОС и затем переходите к безопасной тренировке.",
            bg=bg,
            fg=muted,
            font=("TkDefaultFont", 12),
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        step_frame = tk.Frame(win, bg=bg, padx=18)
        step_frame.pack(fill=tk.X)
        step_vars: dict[str, tk.StringVar] = {}
        step_labels: dict[str, tk.Label] = {}
        for step_id, title in [
            ("camera", "Камера и модель"),
            ("input", "Ввод ОС"),
            ("performance", "Производительность"),
            ("next", "Следующий шаг"),
        ]:
            var = tk.StringVar(value=f"{title}: не проверено")
            label = tk.Label(
                step_frame,
                textvariable=var,
                bg=panel,
                fg=muted,
                anchor="w",
                justify="left",
                wraplength=690,
                padx=12,
                pady=9,
                font=("TkDefaultFont", 12),
            )
            label.pack(fill=tk.X, pady=4)
            step_vars[step_id] = var
            step_labels[step_id] = label

        details = scrolledtext.ScrolledText(
            win,
            wrap=tk.WORD,
            bg="#0b0f13",
            fg="#f3f6f8",
            insertbackground="#f3f6f8",
            relief=tk.FLAT,
            padx=12,
            pady=12,
            font=("TkFixedFont", 10),
            height=10,
        )
        details.pack(fill=tk.BOTH, expand=True, padx=18, pady=(12, 12))
        details.insert(tk.END, "Нажмите «Проверить», чтобы начать.\n")
        details.configure(state=tk.DISABLED)

        actions = tk.Frame(win, bg=bg, padx=18, pady=(0, 16))
        actions.pack(fill=tk.X)

        def set_details(value: str) -> None:
            details.configure(state=tk.NORMAL)
            details.delete("1.0", tk.END)
            details.insert(tk.END, value)
            details.configure(state=tk.DISABLED)

        def apply_statuses(statuses: list[dict], detail_text: str) -> None:
            for item in statuses:
                step_id = str(item.get("id", ""))
                if step_id not in step_vars:
                    continue
                step_vars[step_id].set(
                    f"{item.get('title', step_id)}: {item.get('message', '')}"
                )
                step_labels[step_id].configure(fg=_wizard_status_color(str(item.get("status")), text, warning))
            set_details(detail_text)
            check_button.configure(state=tk.NORMAL)

        def run_checks() -> None:
            check_button.configure(state=tk.DISABLED)
            set_details("Идёт проверка камеры и безопасного ввода...\n")
            for step_id, title in [
                ("camera", "Камера и модель"),
                ("input", "Ввод ОС"),
                ("performance", "Производительность"),
                ("next", "Следующий шаг"),
            ]:
                step_vars[step_id].set(f"{title}: проверяется...")
                step_labels[step_id].configure(fg=warning)

            def worker() -> None:
                try:
                    from .diagnostics import build_report, summarize_doctor_report
                    cfg = AppConfig.load()
                    report = build_report(
                        scan_camera=True,
                        camera_limit=max(1, min(2, int(getattr(cfg.camera, "scan_indices", 4)))),
                        input_probe=True,
                    )
                    summary = summarize_doctor_report(report)
                    statuses = build_first_run_status(report, summary)
                    detail_text = format_first_run_report(statuses, summary, report)
                except Exception as exc:
                    statuses = [{
                        "id": "next",
                        "title": "Следующий шаг",
                        "status": "fail",
                        "message": f"проверка не выполнена: {exc}",
                    }]
                    detail_text = f"Не удалось выполнить мастер первого запуска:\n{exc}"
                try:
                    win.after(0, lambda: apply_statuses(statuses, detail_text))
                except Exception:
                    pass

            threading.Thread(target=worker, daemon=True).start()

        check_button = tk.Button(
            actions,
            text="Проверить",
            command=run_checks,
            bg=accent,
            fg="#07100b",
            activebackground="#56e0a4",
            activeforeground="#07100b",
            relief=tk.FLAT,
            padx=14,
            pady=9,
            font=("TkDefaultFont", 12, "bold"),
        )
        check_button.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            actions,
            text="Безопасная тренировка",
            command=lambda: (win.destroy(), launch_app(True, True)),
            bg="#27323c",
            fg=text,
            activebackground="#33414d",
            activeforeground=text,
            relief=tk.FLAT,
            padx=14,
            pady=9,
            font=("TkDefaultFont", 12),
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            actions,
            text="Сохранить ZIP",
            command=save_report,
            bg="#27323c",
            fg=text,
            activebackground="#33414d",
            activeforeground=text,
            relief=tk.FLAT,
            padx=14,
            pady=9,
            font=("TkDefaultFont", 12),
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            actions,
            text="Закрыть",
            command=win.destroy,
            bg="#27323c",
            fg=text,
            activebackground="#33414d",
            activeforeground=text,
            relief=tk.FLAT,
            padx=14,
            pady=9,
            font=("TkDefaultFont", 12),
        ).pack(side=tk.RIGHT)

        win.after(100, run_checks)

    def show_diagnostics() -> None:
        win = tk.Toplevel(root)
        win.title("AirControl - проверка системы")
        win.geometry("820x620")
        win.minsize(640, 460)
        win.configure(bg=bg)

        tk.Label(
            win,
            text="Проверка системы",
            bg=bg,
            fg=text,
            font=("TkDefaultFont", 20, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 8))

        box = scrolledtext.ScrolledText(
            win,
            wrap=tk.WORD,
            bg="#0b0f13",
            fg="#f3f6f8",
            insertbackground="#f3f6f8",
            relief=tk.FLAT,
            padx=12,
            pady=12,
            font=("TkFixedFont", 11),
        )
        box.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 12))
        box.insert(tk.END, "Идёт проверка...\n")
        box.configure(state=tk.DISABLED)

        actions = tk.Frame(win, bg=bg, padx=18, pady=(0, 16))
        actions.pack(fill=tk.X)

        def set_text(value: str) -> None:
            box.configure(state=tk.NORMAL)
            box.delete("1.0", tk.END)
            box.insert(tk.END, value)
            box.configure(state=tk.DISABLED)

        def refresh(input_probe: bool = False) -> None:
            try:
                from .diagnostics import build_report, summarize_doctor_report
                report = build_report(scan_camera=False, input_probe=input_probe)
                summary = "\n".join(summarize_doctor_report(report))
                set_text(f"{summary}\n\n{report}")
            except Exception as exc:
                set_text(f"Не удалось выполнить проверку:\n{exc}")

        def save_from_diagnostics() -> None:
            save_report()

        tk.Button(
            actions,
            text="Обновить",
            command=refresh,
            bg="#27323c",
            fg=text,
            activebackground="#33414d",
            activeforeground=text,
            relief=tk.FLAT,
            padx=14,
            pady=9,
            font=("TkDefaultFont", 12),
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            actions,
            text="Проверить ввод мыши",
            command=lambda: refresh(input_probe=True),
            bg="#27323c",
            fg=text,
            activebackground="#33414d",
            activeforeground=text,
            relief=tk.FLAT,
            padx=14,
            pady=9,
            font=("TkDefaultFont", 12),
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            actions,
            text="Сохранить ZIP",
            command=save_from_diagnostics,
            bg=accent,
            fg="#07100b",
            activebackground="#56e0a4",
            activeforeground="#07100b",
            relief=tk.FLAT,
            padx=14,
            pady=9,
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            actions,
            text="Закрыть",
            command=win.destroy,
            bg="#27323c",
            fg=text,
            activebackground="#33414d",
            activeforeground=text,
            relief=tk.FLAT,
            padx=14,
            pady=9,
            font=("TkDefaultFont", 12),
        ).pack(side=tk.RIGHT)

        win.after(50, refresh)

    def add_button(label: str, command, primary: bool = False) -> None:
        tk.Button(
            buttons,
            text=label,
            command=command,
            bg=accent if primary else "#27323c",
            fg="#07100b" if primary else text,
            activebackground="#56e0a4" if primary else "#33414d",
            activeforeground="#07100b" if primary else text,
            relief=tk.FLAT,
            padx=18,
            pady=16,
            font=("TkDefaultFont", 14, "bold" if primary else "normal"),
            anchor="w",
        ).pack(fill=tk.X, pady=6)

    add_button("Мастер первого запуска", show_first_run_wizard, True)
    add_button("1. Безопасная тренировка (без кликов)", lambda: launch_app(True, True))
    add_button("2. Ассистивное управление: баланс", lambda: launch_app(True, False))
    add_button(
        "Ассистивное управление: тремор / дрожание",
        lambda: launch_app(True, False, assistive_preset="steady"),
    )
    add_button(
        "Ассистивное управление: мало движения рукой",
        lambda: launch_app(True, False, assistive_preset="low_motion"),
    )
    add_button("Калибровка под пользователя", run_calibration)
    add_button("Проверить систему", show_diagnostics)
    add_button("Сохранить отчёт диагностики", save_report)
    add_button("Просмотр камеры (без управления)", lambda: launch_app(False, True, "view"))

    tk.Button(container, text="Выход", command=root.destroy, bg=bg, fg=muted,
              activebackground=bg, activeforeground=text, relief=tk.FLAT,
              font=("TkDefaultFont", 12)).pack(anchor="e", pady=(12, 0))

    def update_readiness_status() -> None:
        def worker() -> None:
            try:
                from .diagnostics import build_report, summarize_doctor_report
                summary = summarize_doctor_report(build_report(scan_camera=False))
                message = _launcher_status_from_summary(summary)
            except Exception as exc:
                message = f"Проверка системы недоступна: {exc}"
            try:
                root.after(0, lambda: status.set(message))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    root.after(150, update_readiness_status)
    root.mainloop()


def _launcher_status_from_summary(summary: list[str]) -> str:
    text = "\n".join(summary)
    if "Status: needs attention" in text:
        return "Проверка нашла проблему. Нажмите «Проверить систему» или сохраните отчёт диагностики."
    if "Camera was not opened" in text:
        return "Базовая проверка пройдена. Камера проверится при безопасной тренировке."
    return "Базовая проверка пройдена. Начните с безопасной тренировки."


def build_first_run_status(report: str, summary: list[str]) -> list[dict]:
    """Build first-run wizard statuses from doctor output."""
    summary_text = "\n".join(summary)
    statuses = [
        _first_run_camera_status(report),
        _first_run_input_status(report),
        _first_run_performance_status(summary_text),
    ]
    blocking = any(item["status"] == "fail" for item in statuses[:2])
    risky = any(item["status"] in ("warn", "fail") for item in statuses)
    if blocking:
        next_status = {
            "id": "next",
            "title": "Следующий шаг",
            "status": "fail",
            "message": "откройте диагностику или сохраните ZIP-отчёт перед реальным управлением",
        }
    elif risky:
        next_status = {
            "id": "next",
            "title": "Следующий шаг",
            "status": "warn",
            "message": "начните с безопасной тренировки и проверьте стабильность",
        }
    else:
        next_status = {
            "id": "next",
            "title": "Следующий шаг",
            "status": "ok",
            "message": "начните с безопасной тренировки, затем включайте ассистивное управление",
        }
    statuses.append(next_status)
    return statuses


def format_first_run_report(statuses: list[dict], summary: list[str], report: str) -> str:
    lines = ["=== AirControl first-run wizard ==="]
    for item in statuses:
        lines.append(f"{item.get('title', item.get('id'))}: {item.get('status')} - {item.get('message')}")
    lines.append("")
    lines.extend(summary)
    lines.append("")
    lines.append(report)
    return "\n".join(lines)


def _first_run_camera_status(report: str) -> dict:
    required = ("OpenCV: OK", "MediaPipe: OK", "Hand model: OK", "Tkinter: OK")
    missing = [item.split(":")[0] for item in required if item not in report]
    if missing:
        return {
            "id": "camera",
            "title": "Камера и модель",
            "status": "fail",
            "message": "не готовы компоненты: " + ", ".join(missing),
        }
    if "Camera scan: skipped" in report:
        return {
            "id": "camera",
            "title": "Камера и модель",
            "status": "warn",
            "message": "камера ещё не открывалась",
        }
    if ": OK frame=" in report:
        return {
            "id": "camera",
            "title": "Камера и модель",
            "status": "ok",
            "message": "камера открылась, кадр получен",
        }
    return {
        "id": "camera",
        "title": "Камера и модель",
        "status": "fail",
        "message": "камера не дала рабочий кадр",
    }


def _first_run_input_status(report: str) -> dict:
    if (
        "input backend: FAIL" in report
        or "input probe: FAIL" in report
        or "input mouse move probe: FAIL" in report
    ):
        return {
            "id": "input",
            "title": "Ввод ОС",
            "status": "fail",
            "message": "ОС не подтвердила управление курсором/клавишами",
        }
    if (
        "input backend: WARN" in report
        or "input probe: WARN" in report
        or "input mouse move probe: SKIPPED" in report
    ):
        return {
            "id": "input",
            "title": "Ввод ОС",
            "status": "warn",
            "message": "ввод доступен не полностью, проверьте в безопасной тренировке",
        }
    return {
        "id": "input",
        "title": "Ввод ОС",
        "status": "ok",
        "message": "backend ввода готов",
    }


def _first_run_performance_status(summary_text: str) -> dict:
    if "Low FPS" in summary_text or "Slow detection" in summary_text:
        return {
            "id": "performance",
            "title": "Производительность",
            "status": "warn",
            "message": "обнаружены признаки низкого FPS, используйте ассистивный профиль",
        }
    return {
        "id": "performance",
        "title": "Производительность",
        "status": "pending",
        "message": "FPS измеряется в окне безопасной тренировки",
    }


def _wizard_status_color(status: str, ok_color: str, warn_color: str) -> str:
    if status == "ok":
        return "#42d392"
    if status == "fail":
        return "#ff6b6b"
    if status == "warn":
        return warn_color
    return ok_color


def _requires_control_preflight(cfg: AppConfig) -> bool:
    return cfg.start_mode == "control" and not cfg.input.dry_run


def _control_preflight_message(summary: list[str]) -> str | None:
    text = "\n".join(summary)
    if "Status: needs attention" not in text:
        return None
    issues = [
        line[2:] for line in summary
        if line.startswith("- ") and not line.startswith("- Нажмите")
    ][:4]
    issue_text = "\n".join(f"- {item}" for item in issues) if issues else "- Ввод ОС требует проверки."
    return (
        "AirControl пока не подтвердил, что ОС принимает управление мышью и клавиатурой.\n\n"
        f"{issue_text}\n\n"
        "Если продолжить, жесты могут распознаваться, но курсор, клики или клавиши "
        "могут не выполняться. Нажмите «Нет», чтобы открыть диагностику и сохранить ZIP-отчёт. "
        "Для тренировки без риска используйте «Безопасная тренировка»."
    )


def _confirm_control_preflight(
    root,
    *,
    status_var=None,
    diagnostics_callback: Callable[[], None] | None = None,
    report_builder: Callable[..., str] | None = None,
    summary_builder: Callable[[str], list[str]] | None = None,
    ask_yes_no: Callable[[str, str], bool] | None = None,
) -> bool:
    try:
        if report_builder is None:
            from .diagnostics import build_report as report_builder
        if summary_builder is None:
            from .diagnostics import summarize_doctor_report as summary_builder
        report = report_builder(scan_camera=False, input_probe=True)
        message = _control_preflight_message(summary_builder(report))
    except Exception as exc:
        message = (
            "AirControl не смог выполнить предварительную проверку ввода.\n\n"
            f"Ошибка: {exc}\n\n"
            "Продолжить реальное управление всё равно?"
        )

    if not message:
        return True

    if ask_yes_no is None:
        ask_yes_no = messagebox.askyesno
    if ask_yes_no("AirControl", message):
        return True

    if status_var is not None:
        try:
            status_var.set("Реальное управление не запущено. Откройте диагностику или безопасную тренировку.")
        except Exception:
            pass
    if diagnostics_callback is not None:
        _call_later(root, diagnostics_callback)
    return False


def _call_later(root, callback: Callable[[], None]) -> None:
    try:
        root.after(50, callback)
    except Exception:
        callback()


def _launch_aircontrol_from_launcher(root, cfg: AppConfig, app_factory=None) -> bool:
    """Start the camera app from a Tk callback and report startup failures."""
    _destroy_root(root)
    try:
        if app_factory is None:
            from .app import AirControlApp
            app_factory = AirControlApp
        app_factory(cfg).run()
        return True
    except Exception as exc:
        _show_launcher_startup_error(exc)
        return False


def _run_calibration_from_launcher(root, cfg: AppConfig, calibration_runner=None) -> bool:
    """Start calibration from a Tk callback and report startup failures."""
    _destroy_root(root)
    try:
        if calibration_runner is None:
            from .ui.calibration import run_calibration as calibration_runner
        calibration_runner(cfg)
        return True
    except Exception as exc:
        _show_launcher_startup_error(exc)
        return False


def _destroy_root(root) -> None:
    try:
        root.destroy()
    except Exception:
        pass


def _show_launcher_startup_error(exc: BaseException) -> tuple[str | None, str | None, str | None]:
    """Write crash/support files and show a no-console error dialog."""
    try:
        from .crash import build_crash_message, write_crash_log, write_startup_support_bundle

        log_path = write_crash_log(exc)
        support_path = write_startup_support_bundle()
        message = build_crash_message(log_path, exc, support_path)
    except Exception:
        log_path = None
        support_path = None
        message = f"AirControl не смог запуститься.\n\nОшибка: {exc}"
    try:
        messagebox.showerror("AirControl", message)
    except Exception:
        pass
    return log_path, support_path, message
