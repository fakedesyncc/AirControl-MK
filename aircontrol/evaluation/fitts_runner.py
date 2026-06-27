"""Интерактивный стенд теста Фиттса (полноэкранное окно Tkinter).

Рисует мишени по схеме ISO 9241-9, ловит клики (любым курсором — жестовым или
обычной мышью) и по завершении сохраняет результаты в CSV и показывает сводку.
Запускается отдельно (`python -m aircontrol.evaluation.fitts_runner`) или из
приложения. Метка method ("gesture"/"mouse") пишется в CSV для сравнения.
"""

import os
import time
import tkinter as tk

from .fitts import FittsTest
from ..config import EvaluationConfig


class FittsRunner:
    def __init__(self, eval_cfg: EvaluationConfig, method: str = "gesture"):
        self.cfg = eval_cfg
        self.method = method
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#1e1e1e")
        self.root.title("Fitts' Law Test — AirControl")
        self.w = self.root.winfo_screenwidth()
        self.h = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h,
                                bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.test = FittsTest(eval_cfg, self.w, self.h)
        self.started = False

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.draw()

    def on_click(self, event) -> None:
        if self.test.finished:
            return
        self.test.register_click(event.x, event.y, time.time())
        if self.test.finished:
            self.finish()
        else:
            self.draw()

    def draw(self) -> None:
        self.canvas.delete("all")
        cur = self.test.current_target
        if cur is None:
            return
        done, total = self.test.progress

        # Все мишени текущего условия — бледным фоном (ориентир «звезды»).
        for t in self.test.targets:
            if t["condition_id"] == cur["condition_id"]:
                x, y = t["pos"]
                r = t["width"] / 2
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        outline="#3a3a3a", width=1)

        # Текущая мишень — ярко.
        x, y = cur["pos"]
        r = cur["width"] / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="#ff5252", outline="#ffffff", width=2)

        self.canvas.create_text(
            self.w / 2, 40,
            text=f"Тест Фиттса [{self.method}]  —  {done}/{total}   "
                 f"(A={cur['amplitude']}px, W={cur['width']}px)   ESC — выход",
            fill="#cccccc", font=("Helvetica", 18))

    def finish(self) -> None:
        summary = self.test.summary()
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        csv_path = os.path.join(
            self.cfg.log_dir,
            f"fitts_{self.cfg.participant_id}_{self.method}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.test.save_csv(csv_path, self.cfg.participant_id, self.method)

        self.canvas.delete("all")
        lines = [
            "ТЕСТ ЗАВЕРШЁН",
            "",
            f"Метод ввода: {self.method}",
            f"Средняя пропускная способность: {summary['throughput_mean']:.2f} бит/с",
            f"Среднее время наведения: {summary['mt_mean']*1000:.0f} мс",
            f"Средняя доля промахов: {summary['error_rate_mean']*100:.1f} %",
            f"Условий протестировано: {summary['n_conditions']}",
            "",
            f"Результаты сохранены: {csv_path}",
            "",
            "Нажмите ESC для выхода",
        ]
        self.canvas.create_text(self.w / 2, self.h / 2,
                                text="\n".join(lines), fill="#ffffff",
                                font=("Helvetica", 22), justify="center")
        print(f"[fitts] Сводка ({self.method}): TP={summary['throughput_mean']:.2f} бит/с, "
              f"MT={summary['mt_mean']*1000:.0f} мс, ошибки={summary['error_rate_mean']*100:.1f}%")
        print(f"[fitts] CSV: {csv_path}")

    def run(self) -> None:
        self.root.mainloop()


def run_fitts_test(eval_cfg: EvaluationConfig, method: str = "gesture") -> None:
    FittsRunner(eval_cfg, method).run()


if __name__ == "__main__":
    import argparse

    from ..config import AppConfig

    parser = argparse.ArgumentParser(description="Тест Фиттса (ISO 9241-9)")
    parser.add_argument("--method", default="gesture",
                        help="метка метода ввода (gesture/mouse/...)")
    parser.add_argument("--participant", default=None, help="ID участника")
    args = parser.parse_args()

    cfg = AppConfig.load()
    if args.participant:
        cfg.evaluation.participant_id = args.participant
    run_fitts_test(cfg.evaluation, args.method)
