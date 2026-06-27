"""Simple no-console launcher for end users.

The packaged app starts here: large buttons, safe first action, no terminal.
Developer and tester commands remain available through ``python -m aircontrol``.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from .config import AppConfig, apply_assistive_profile


def prepare_launch_config(
    cfg: AppConfig,
    *,
    assistive: bool = False,
    dry_input: bool = False,
    start_mode: str | None = None,
) -> AppConfig:
    """Apply launcher choices to a config object."""
    if assistive:
        apply_assistive_profile(cfg)
    if start_mode is not None:
        cfg.start_mode = start_mode
    elif assistive:
        cfg.start_mode = "control"
    cfg.input.dry_run = dry_input
    return cfg


def run_launcher() -> None:
    root = tk.Tk()
    root.title("AirControl")
    root.geometry("620x520")
    root.minsize(540, 460)

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

    def launch_app(assistive: bool, dry_input: bool, start_mode: str | None = None) -> None:
        cfg = prepare_launch_config(
            AppConfig.load(),
            assistive=assistive,
            dry_input=dry_input,
            start_mode=start_mode,
        )
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

    add_button("1. Безопасная тренировка (без кликов)", lambda: launch_app(True, True), True)
    add_button("2. Начать ассистивное управление", lambda: launch_app(True, False))
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
