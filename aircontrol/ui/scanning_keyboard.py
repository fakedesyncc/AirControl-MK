"""Tk-представление сканирующей клавиатуры.

Рисует раскладку из чистого автомата ScanKeyboard, подсвечивает текущую
позицию, гонит tick() по таймеру (интервал из конфига) и отображает выбор как
событие «выбор». Сам ввод символов/клавиш в ОС делается существующим
бэкендом (ActionExecutor) — ничего своего для ввода здесь не изобретается.

Источники «выбора»:
  (a) клавиша-переключатель (по умолчанию Space) и Enter — работают везде и
      удобны для ручной проверки без жестов;
  (b) внешний хук trigger_select(): его может вызвать жестовый dwell, чтобы
      управлять клавиатурой тем же удержанием, что и dwell-click.

Модуль НЕ требует юнит-тестов: вся логика — в aircontrol/control/scanning.py.
"""

from __future__ import annotations

import tkinter as tk
from typing import Optional

from ..config import AppConfig, SCREENSHOTS_DIR
from ..control.actions import ActionExecutor
from ..control.input_backend import Key
from ..control.scanning import (
    KEY_BACKSPACE,
    KEY_CHAR,
    KEY_ENTER,
    KEY_EXIT,
    KEY_SPACE,
    ScanKeyboard,
    ScanMode,
    ScanOutput,
    default_layout,
)


class ScanKeyboardWindow:
    """Окно сканирующей клавиатуры поверх остальных окон."""

    def __init__(self, cfg: AppConfig, executor: Optional[ActionExecutor] = None):
        self.cfg = cfg
        scan_cfg = cfg.scan_keyboard
        self.interval_ms = max(150, int(scan_cfg.scan_interval * 1000))
        self.executor = executor or ActionExecutor(
            SCREENSHOTS_DIR, dry_run=cfg.input.dry_run
        )

        self.state = ScanKeyboard(
            layout=default_layout(),
            on_output=self._on_output,
            on_exit=self.close,
            max_loops=scan_cfg.max_loops,
        )

        self.root = tk.Tk()
        self.root.title("AirControl — экранная клавиатура")
        self.root.configure(bg="#101418")
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass

        self._after_id: Optional[str] = None
        self._buttons: list[list[tk.Label]] = []
        self._build_ui()

        select_key = (scan_cfg.select_key or "space").lower()
        self.root.bind_all(f"<{select_key}>", lambda _e: self.trigger_select())
        if select_key != "space":
            self.root.bind_all("<space>", lambda _e: self.trigger_select())
        self.root.bind_all("<Return>", lambda _e: self.trigger_select())
        self.root.bind_all("<Escape>", lambda _e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ---- построение интерфейса --------------------------------------------

    def _build_ui(self) -> None:
        text = "#f3f6f8"
        muted = "#a9b4bf"

        tk.Label(
            self.root,
            text="Экранная клавиатура",
            bg="#101418",
            fg=text,
            font=("TkDefaultFont", 18, "bold"),
        ).pack(anchor="w", padx=18, pady=(14, 2))
        tk.Label(
            self.root,
            text="Подсветка сама перебирает строки и клавиши. "
            "Сделайте «выбор» (Пробел/Enter), чтобы выбрать строку, затем клавишу.",
            bg="#101418",
            fg=muted,
            font=("TkDefaultFont", 11),
            wraplength=560,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        self.text_var = tk.StringVar(value="")
        tk.Entry(
            self.root,
            textvariable=self.text_var,
            bg="#0b0f13",
            fg=text,
            insertbackground=text,
            relief=tk.FLAT,
            font=("TkFixedFont", 16),
        ).pack(fill=tk.X, padx=18, pady=(0, 12), ipady=8)

        grid = tk.Frame(self.root, bg="#101418")
        grid.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 12))
        for r, row in enumerate(self.state.layout):
            row_widgets: list[tk.Label] = []
            row_frame = tk.Frame(grid, bg="#101418")
            row_frame.pack(fill=tk.X, pady=3)
            for key in row:
                label = tk.Label(
                    row_frame,
                    text=key.label,
                    bg="#27323c",
                    fg=text,
                    width=4,
                    height=2,
                    relief=tk.FLAT,
                    font=("TkDefaultFont", 15, "bold"),
                )
                label.pack(side=tk.LEFT, padx=3)
                row_widgets.append(label)
            self._buttons.append(row_widgets)

        self.status_var = tk.StringVar(value="")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#101418",
            fg=muted,
            font=("TkDefaultFont", 11),
        ).pack(anchor="w", padx=18, pady=(0, 14))

    # ---- цикл подсветки ----------------------------------------------------

    def _schedule(self) -> None:
        self._after_id = self.root.after(self.interval_ms, self._on_tick)

    def _on_tick(self) -> None:
        self.state.tick()
        self._render()
        if self.state.running:
            self._schedule()
        else:
            self.status_var.set("Сканирование остановлено.")

    def _render(self) -> None:
        idle_bg, idle_fg = "#27323c", "#f3f6f8"
        row_bg = "#2c3f4f"          # подсветка целой строки (режим ROW)
        active_bg, active_fg = "#42d392", "#07100b"  # активная позиция

        in_col = self.state.mode is ScanMode.COL
        active_row = self.state.row if in_col else self.state.index
        active_col = self.state.index if in_col else -1

        for r, row_widgets in enumerate(self._buttons):
            for c, widget in enumerate(row_widgets):
                if r == active_row and (not in_col or c == active_col):
                    widget.configure(bg=active_bg, fg=active_fg)
                elif (not in_col) and r == active_row:
                    widget.configure(bg=row_bg, fg=idle_fg)
                elif in_col and r == active_row:
                    widget.configure(bg=row_bg, fg=idle_fg)
                else:
                    widget.configure(bg=idle_bg, fg=idle_fg)

        mode_name = "выбор клавиши" if in_col else "выбор строки"
        self.status_var.set(f"Режим: {mode_name}")

    # ---- «выбор» -----------------------------------------------------------

    def trigger_select(self) -> None:
        """Единое событие «выбор» (клавиша-переключатель или внешний dwell-хук)."""
        if not self.state.running:
            return
        self.state.select()
        self._render()

    def _on_output(self, output: ScanOutput) -> None:
        """Отправить выбранную клавишу в ОС и обновить превью текста."""
        if output.kind == KEY_CHAR:
            self.executor.type_text(output.char)
            self.text_var.set(self.text_var.get() + output.char)
        elif output.kind == KEY_SPACE:
            self.executor.type_text(" ")
            self.text_var.set(self.text_var.get() + " ")
        elif output.kind == KEY_BACKSPACE:
            self.executor.key_tap(Key.backspace)
            self.text_var.set(self.text_var.get()[:-1])
        elif output.kind == KEY_ENTER:
            self.executor.key_tap(Key.enter)

    # ---- жизненный цикл ----------------------------------------------------

    def run(self) -> None:
        self._render()
        self._schedule()
        self.root.mainloop()

    def close(self) -> None:
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        try:
            self.executor.release_all()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def run_scanning_keyboard(cfg: Optional[AppConfig] = None) -> None:
    """Точка входа: открыть окно сканирующей клавиатуры."""
    ScanKeyboardWindow(cfg or AppConfig.load()).run()
