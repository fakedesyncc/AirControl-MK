"""Accessible personal calibration wizard.

The old calibration flow used an OpenCV window and required Space/Esc. That is
fine for lab work, but not for an assistive product. This module keeps the same
calibration math and presents it through Tk: large buttons, live camera preview,
optional keyboard shortcuts, and no console dependency.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageTk

from ..config import AppConfig
from ..gestures import features as F
from ..tracking.camera import Camera
from ..tracking.hand_tracker import HandTracker, INDEX_TIP
from ..ui.renderer import draw_landmarks


def compute_active_region(samples: Iterable[Sequence[float]]) -> float | None:
    """Return calibrated cursor active_region from collected palm centers."""
    arr = np.array(list(samples), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 10 or arr.shape[1] < 2:
        return None
    span = max(float(np.ptp(arr[:, 0])), float(np.ptp(arr[:, 1])))
    return round(min(1.0, max(0.3, span)), 2)


def compute_pinch_thresholds(
    open_values: Iterable[float],
    pinch_values: Iterable[float],
) -> tuple[float, float] | None:
    """Return trigger/release thresholds from open-hand and pinch samples."""
    open_arr = np.array(list(open_values), dtype=np.float32)
    pinch_arr = np.array(list(pinch_values), dtype=np.float32)
    if open_arr.size <= 5 or pinch_arr.size <= 5:
        return None
    open_med = float(np.median(open_arr))
    pinch_med = float(np.median(pinch_arr))
    gap = open_med - pinch_med
    if gap <= 0.05:
        return None
    trigger = round(pinch_med + 0.35 * gap, 3)
    release = round(pinch_med + 0.65 * gap, 3)
    return trigger, release


# --- Калибровка взгляда: чистые помощники (без Tk/камеры/MediaPipe) ----------
#
# Эти функции вынесены отдельно от тяжёлого мастера, чтобы их можно было
# тестировать без камеры и модели лица. Они описывают раскладку точек-целей,
# сборку собранных (raw, target) пар в формат GazeEstimator.calibrate() и
# проверку качества/достаточности данных.

# Минимальное число успешно собранных целей для устойчивой аффинной подгонки.
# Аффинная карта по каждой оси требует ≥2 различных точек; 3 даёт запас на шум.
GAZE_MIN_TARGETS = 3


def gaze_target_points(margin: float = 0.15) -> list[tuple[float, float]]:
    """Нормализованные точки-цели для калибровки взгляда (5 шт.: углы + центр).

    Возвращает координаты в [0..1] (доли ширины/высоты экрана): четыре угла на
    отступе margin от краёв плюс центр. Отступ нужен, чтобы пользователь не
    «закатывал» глаза в самые края (там оценка радужки особенно шумная) и чтобы
    точки оставались видимыми в окне. Порядок фиксирован (углы по кругу, затем
    центр) — мастер показывает цели именно в нём."""
    m = clamp_unit(margin)
    if m > 0.45:                      # защита от вырождения в одну точку
        m = 0.45
    lo, hi, mid = m, 1.0 - m, 0.5
    return [
        (lo, lo),    # верхний левый
        (hi, lo),    # верхний правый
        (hi, hi),    # нижний правый
        (lo, hi),    # нижний левый
        (mid, mid),  # центр
    ]


def clamp_unit(v: float) -> float:
    """Зажать значение в [0..1] (доля экрана не выходит за его пределы)."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def average_raw(raw_samples: Iterable[Sequence[float]]) -> tuple[float, float] | None:
    """Усреднить собранные за одну цель сырые векторы взгляда (raw_x, raw_y).

    raw_samples — последовательность пар (raw_x, raw_y), накопленных за время
    удержания (dwell) одной точки. Возвращает усреднённую пару или None, если
    валидных образцов не было (взгляд не распознан → цель пропущена)."""
    xs: list[float] = []
    ys: list[float] = []
    for s in raw_samples:
        if s is None or len(s) < 2:
            continue
        xs.append(float(s[0]))
        ys.append(float(s[1]))
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def build_gaze_calibration_samples(
    collected: Iterable[tuple[Sequence[float] | None, Sequence[float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Собрать пары (raw, target) для GazeEstimator.calibrate()/GazeCalibration.fit().

    collected — последовательность (raw, target), где raw — усреднённый сырой
    вектор взгляда для цели (или None, если цель не удалось снять), а target —
    нормализованная точка экрана из gaze_target_points(). Цели с raw=None
    отбрасываются. Результат — список ((raw_x, raw_y), (tgt_x, tgt_y)),
    пригодный для аффинной подгонки."""
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for raw, target in collected:
        if raw is None or len(raw) < 2 or target is None or len(target) < 2:
            continue
        pairs.append((
            (float(raw[0]), float(raw[1])),
            (clamp_unit(target[0]), clamp_unit(target[1])),
        ))
    return pairs


def gaze_samples_quality_ok(
    samples: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> bool:
    """Проверить, что собранных пар хватает и они не вырождены по обеим осям.

    Нужно ≥GAZE_MIN_TARGETS пар, и сырые значения должны различаться и по x, и
    по y (иначе аффинная подгонка по оси выродится в тождество и калибровка
    окажется бесполезной — пользователь смотрел в одну точку или взгляд не
    считывался). Эта проверка отделяет «успешную» калибровку от «повторите»."""
    if len(samples) < GAZE_MIN_TARGETS:
        return False
    raw_xs = [s[0][0] for s in samples]
    raw_ys = [s[0][1] for s in samples]
    span_x = max(raw_xs) - min(raw_xs)
    span_y = max(raw_ys) - min(raw_ys)
    return span_x > 1e-3 and span_y > 1e-3


class CalibrationWizard:
    def __init__(self, root: tk.Tk, cfg: AppConfig):
        self.root = root
        self.cfg = cfg
        self.cam = None
        self.tracker = None
        try:
            self.cam = Camera(cfg.camera)
            self.tracker = HandTracker(cfg.tracking)
        except Exception:
            if self.cam is not None:
                self.cam.release()
            raise
        self.after_id: str | None = None
        self.photo = None

        self.stage = "region"
        self.collecting = False
        self.collect_kind: str | None = None
        self.collect_until = 0.0
        self.samples: list = []
        self.open_values: list[float] = []
        self.region_done = False
        self.pinch_done = False
        self.closed = False

        self._build_ui()
        self._set_stage_region()
        self._schedule_frame()

    def _build_ui(self) -> None:
        self.root.title("AirControl - калибровка")
        self.root.geometry("920x760")
        self.root.minsize(760, 620)
        self.root.configure(bg="#101418")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind_all("<space>", lambda _e: self.primary_action())
        self.root.bind_all("<Escape>", lambda _e: self.close())

        shell = tk.Frame(self.root, bg="#101418", padx=18, pady=16)
        shell.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            shell,
            text="Калибровка AirControl",
            bg="#101418",
            fg="#f3f6f8",
            font=("TkDefaultFont", 24, "bold"),
        ).pack(anchor="w")

        self.instruction = tk.StringVar()
        tk.Label(
            shell,
            textvariable=self.instruction,
            bg="#182028",
            fg="#f3f6f8",
            justify="left",
            wraplength=850,
            padx=16,
            pady=12,
            font=("TkDefaultFont", 14),
        ).pack(fill=tk.X, pady=(12, 14))

        self.video = tk.Label(shell, bg="#050708")
        self.video.pack(fill=tk.BOTH, expand=True)

        status_row = tk.Frame(shell, bg="#101418")
        status_row.pack(fill=tk.X, pady=(12, 0))
        self.hand_status = tk.StringVar(value="Рука: ожидание камеры")
        self.result_status = tk.StringVar(value="")
        tk.Label(
            status_row,
            textvariable=self.hand_status,
            bg="#101418",
            fg="#ffd166",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            status_row,
            textvariable=self.result_status,
            bg="#101418",
            fg="#a9b4bf",
            font=("TkDefaultFont", 12),
        ).pack(side=tk.RIGHT)

        buttons = tk.Frame(shell, bg="#101418")
        buttons.pack(fill=tk.X, pady=(14, 0))

        self.primary = self._button(buttons, "Начать", self.primary_action, True)
        self.primary.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.skip = self._button(buttons, "Пропустить шаг", self.skip_step, False)
        self.skip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.close_btn = self._button(buttons, "Закрыть", self.close, False)
        self.close_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

    def _button(self, parent, text, command, primary: bool) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#42d392" if primary else "#27323c",
            fg="#07100b" if primary else "#f3f6f8",
            activebackground="#56e0a4" if primary else "#33414d",
            activeforeground="#07100b" if primary else "#f3f6f8",
            relief=tk.FLAT,
            padx=16,
            pady=14,
            font=("TkDefaultFont", 14, "bold" if primary else "normal"),
        )

    def _set_stage_region(self) -> None:
        self.stage = "region"
        self.collecting = False
        self.primary.config(text="Начать запись активной зоны", state=tk.NORMAL)
        self.skip.config(text="Пропустить шаг", state=tk.NORMAL)
        self.instruction.set(
            "Шаг 1/3: активная зона.\n"
            "Двигайте раскрытой рукой по той области, где человеку удобно двигаться. "
            "AirControl подстроит чувствительность, чтобы курсор доставал до краёв экрана."
        )

    def _set_stage_open(self) -> None:
        self.stage = "open"
        self.collecting = False
        self.primary.config(text="Записать раскрытую руку", state=tk.NORMAL)
        self.skip.config(text="Сохранить без щипка", state=tk.NORMAL)
        self.instruction.set(
            "Шаг 2/3: раскрытая рука.\n"
            "Держите руку раскрытой и неподвижной. Это нужно, чтобы отличать обычное "
            "положение руки от щипка."
        )

    def _set_stage_pinch(self) -> None:
        self.stage = "pinch"
        self.collecting = False
        self.primary.config(text="Записать щипок", state=tk.NORMAL)
        self.skip.config(text="Сохранить без щипка", state=tk.NORMAL)
        self.instruction.set(
            "Шаг 3/3: щипок.\n"
            "Соедините большой и указательный пальцы и удерживайте жест. "
            "Если человеку сложно делать щипок, пропустите шаг: dwell-click уже включён "
            "в ассистивном профиле."
        )

    def _set_done(self) -> None:
        self.cfg.save()
        self.stage = "done"
        self.collecting = False
        self.primary.config(text="Готово", state=tk.NORMAL)
        self.skip.config(text="Повторить калибровку", state=tk.NORMAL)
        parts = []
        if self.region_done:
            parts.append(f"активная зона {self.cfg.cursor.active_region}")
        if self.pinch_done:
            parts.append(
                f"щипок {self.cfg.gestures.pinch_trigger_ratio}/"
                f"{self.cfg.gestures.pinch_release_ratio}"
            )
        self.result_status.set("Сохранено: " + (", ".join(parts) if parts else "без изменений"))
        self.instruction.set(
            "Калибровка сохранена.\n"
            "Теперь можно закрыть окно и запустить безопасную тренировку или "
            "ассистивное управление."
        )

    def primary_action(self) -> None:
        if self.collecting:
            return
        if self.stage == "region":
            self._start_collection("region", 5.0)
        elif self.stage == "open":
            self._start_collection("open", 3.0)
        elif self.stage == "pinch":
            self._start_collection("pinch", 3.0)
        elif self.stage == "done":
            self.close()

    def skip_step(self) -> None:
        if self.collecting:
            return
        if self.stage == "region":
            self._set_stage_open()
        elif self.stage == "open":
            self._set_done()
        elif self.stage == "pinch":
            self._set_done()
        elif self.stage == "done":
            self.region_done = False
            self.pinch_done = False
            self.open_values = []
            self._set_stage_region()

    def _start_collection(self, kind: str, duration: float) -> None:
        self.collecting = True
        self.collect_kind = kind
        self.collect_until = time.monotonic() + duration
        self.samples = []
        self.primary.config(text="Идёт запись...", state=tk.DISABLED)
        self.skip.config(state=tk.DISABLED)

    def _finish_collection(self) -> None:
        kind = self.collect_kind
        samples = self.samples
        self.collecting = False
        self.collect_kind = None
        self.primary.config(state=tk.NORMAL)
        self.skip.config(state=tk.NORMAL)

        if kind == "region":
            value = compute_active_region(samples)
            if value is None:
                self._set_stage_region()
                self.result_status.set("Рука почти не была видна. Повторите шаг.")
                return
            self.cfg.cursor.active_region = value
            self.region_done = True
            self._set_stage_open()
            self.result_status.set(f"Активная зона: {value}")
            return

        if kind == "open":
            if len(samples) <= 5:
                self._set_stage_open()
                self.result_status.set("Раскрытая рука не распознана. Повторите шаг.")
                return
            self.open_values = [float(v) for v in samples]
            self._set_stage_pinch()
            self.result_status.set("Раскрытая рука записана")
            return

        if kind == "pinch":
            thresholds = compute_pinch_thresholds(self.open_values, samples)
            if thresholds is None:
                self._set_stage_pinch()
                self.result_status.set("Щипок слишком похож на раскрытую руку. Повторите или пропустите.")
                return
            trigger, release = thresholds
            self.cfg.gestures.pinch_trigger_ratio = trigger
            self.cfg.gestures.pinch_release_ratio = release
            self.pinch_done = True
            self._set_done()

    def _schedule_frame(self) -> None:
        if self.closed:
            return
        self._update_frame()
        self.after_id = self.root.after(16, self._schedule_frame)

    def _update_frame(self) -> None:
        ok, frame = self.cam.read()
        if not ok:
            self.hand_status.set("Камера: нет кадра")
            frame = self._placeholder_frame("Камера не отдаёт кадры")
            self._show_frame(frame)
            return

        hand = None
        detector_error = False
        try:
            hands = self.tracker.detect(frame)
            hand = hands[0] if hands else None
        except Exception as exc:
            detector_error = True
            self.hand_status.set(f"Детектор: {exc}")

        if hand is not None:
            draw_landmarks(frame, hand)
            self.hand_status.set("Рука: найдена")
            if self.collecting and self.collect_kind == "region":
                self.samples.append(F.palm_center(hand.landmarks))
            elif self.collecting and self.collect_kind in {"open", "pinch"}:
                self.samples.append(float(F.pinch_ratio(hand.landmarks, INDEX_TIP)))
        elif not detector_error:
            self.hand_status.set("Рука: покажите ладонь в камеру")

        if self.collecting:
            remaining = max(0.0, self.collect_until - time.monotonic())
            cv2.putText(
                frame,
                f"{remaining:0.1f}s",
                (frame.shape[1] - 100, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            if remaining <= 0:
                self._finish_collection()

        self._show_frame(frame)

    def _placeholder_frame(self, text: str):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, text, (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2)
        return frame

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(320, self.video.winfo_width())
        max_h = max(220, self.video.winfo_height())
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.video.configure(image=self.photo)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
        try:
            if self.cam is not None:
                self.cam.release()
        finally:
            if self.tracker is not None:
                self.tracker.close()
            self.root.destroy()


def run_calibration(cfg: AppConfig = None) -> None:
    cfg = cfg or AppConfig.load()
    root = tk.Tk()
    try:
        CalibrationWizard(root, cfg)
    except Exception as exc:
        try:
            messagebox.showerror("AirControl", f"Не удалось открыть калибровку:\n{exc}")
        finally:
            root.destroy()
        return
    root.mainloop()


class GazeCalibrationWizard:
    """Мастер калибровки взгляда: показывает точки-цели и снимает сырой взгляд.

    Тонкий слой над GazeEstimator и камерой. Для каждой нормализованной цели
    (углы + центр, см. gaze_target_points) показывает мишень, ждёт dwell и
    усредняет сырые векторы взгляда GazeEstimator.last_raw, затем подгоняет
    аффинную карту через GazeEstimator.calibrate() и сохраняет коэффициенты в
    конфиг (write_calibration_to_config + cfg.save).

    ДЕГРАДИРУЕТ МЯГКО: если модель лица/MediaPipe недоступны (estimator.ready=
    False) или камера не открылась, показывает понятное сообщение и закрывается
    без падения — gaze остаётся опциональным. Вся проверяемая логика (раскладка
    целей, сборка пар, оценка качества) живёт в чистых помощниках выше."""

    # Сколько секунд удерживать взгляд на каждой цели (dwell) и пауза перед
    # стартом сбора, чтобы пользователь успел перевести взгляд на новую мишень.
    DWELL_SECONDS = 1.6
    SETTLE_SECONDS = 0.7

    def __init__(self, root: tk.Tk, cfg: AppConfig):
        self.root = root
        self.cfg = cfg
        self.cam = None
        self.estimator = None
        self.after_id: str | None = None
        self.closed = False

        # Ленивые импорты держат тяжёлый MediaPipe вне модульного уровня и не
        # ломают импорт calibration.py там, где gaze недоступен.
        from ..tracking.camera import Camera
        from ..tracking.gaze import GazeEstimator, write_calibration_to_config

        self._write_calibration_to_config = write_calibration_to_config

        try:
            self.estimator = GazeEstimator(cfg.fusion.gaze)
        except Exception as exc:
            self.estimator = None
            self._init_error = f"gaze estimator: {exc}"
        else:
            self._init_error = self.estimator.init_error

        if self.estimator is None or not self.estimator.ready:
            # Модель лица/MediaPipe недоступны — выходим с понятным сообщением.
            self._build_unavailable_ui()
            return

        try:
            self.cam = Camera(cfg.camera)
        except Exception as exc:
            self._init_error = f"camera: {exc}"
            self._teardown_resources()
            self._build_unavailable_ui()
            return

        # Состояние сбора по целям.
        self.targets = gaze_target_points()
        self.target_index = 0
        self.collected: list[tuple[tuple[float, float] | None, tuple[float, float]]] = []
        self.raw_buffer: list[tuple[float, float]] = []
        self.phase = "idle"            # idle | settle | collect | done
        self.phase_until = 0.0

        self._build_ui()
        self._schedule_frame()

    # ---- UI ---------------------------------------------------------------

    def _build_unavailable_ui(self) -> None:
        """Экран «взгляд недоступен» — без камеры/модели, только пояснение."""
        self.root.title("AirControl - калибровка взгляда")
        self.root.geometry("640x360")
        self.root.configure(bg="#101418")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind_all("<Escape>", lambda _e: self.close())

        shell = tk.Frame(self.root, bg="#101418", padx=24, pady=24)
        shell.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            shell,
            text="Калибровка взгляда недоступна",
            bg="#101418",
            fg="#f3f6f8",
            font=("TkDefaultFont", 20, "bold"),
        ).pack(anchor="w")
        detail = self._init_error or "модель лица не найдена"
        tk.Label(
            shell,
            text=(
                "Взгляд по веб-камере — опциональная функция и требует модели лица "
                "MediaPipe (face_landmarker.task).\n\n"
                f"Причина: {detail}\n\n"
                "Положите файл модели в каталог проекта и повторите. Остальная "
                "калибровка (рука и щипок) работает без модели лица."
            ),
            bg="#182028",
            fg="#f3f6f8",
            justify="left",
            wraplength=560,
            padx=16,
            pady=14,
            font=("TkDefaultFont", 13),
        ).pack(fill=tk.X, pady=(14, 16))
        self._button(shell, "Закрыть", self.close, True).pack(fill=tk.X)

    def _build_ui(self) -> None:
        self.root.title("AirControl - калибровка взгляда")
        self.root.geometry("960x720")
        self.root.minsize(720, 540)
        self.root.configure(bg="#101418")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind_all("<space>", lambda _e: self.primary_action())
        self.root.bind_all("<Escape>", lambda _e: self.close())

        shell = tk.Frame(self.root, bg="#101418", padx=18, pady=16)
        shell.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            shell,
            text="Калибровка взгляда",
            bg="#101418",
            fg="#f3f6f8",
            font=("TkDefaultFont", 22, "bold"),
        ).pack(anchor="w")

        self.instruction = tk.StringVar(value=(
            "Взгляд по веб-камере — грубый сигнал; калибровка делает его точнее.\n"
            "Нажмите «Начать», смотрите на каждую мишень и не отводите взгляд, "
            "пока она не погаснет. Точек — пять (углы и центр)."
        ))
        tk.Label(
            shell,
            textvariable=self.instruction,
            bg="#182028",
            fg="#f3f6f8",
            justify="left",
            wraplength=900,
            padx=16,
            pady=12,
            font=("TkDefaultFont", 13),
        ).pack(fill=tk.X, pady=(12, 14))

        # Холст с мишенью — координаты целей нормализованы [0..1] к его размеру.
        self.canvas = tk.Canvas(shell, bg="#050708", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value="Готово к запуску")
        tk.Label(
            shell,
            textvariable=self.status,
            bg="#101418",
            fg="#ffd166",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w", pady=(10, 0))

        buttons = tk.Frame(shell, bg="#101418")
        buttons.pack(fill=tk.X, pady=(12, 0))
        self.primary = self._button(buttons, "Начать", self.primary_action, True)
        self.primary.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.close_btn = self._button(buttons, "Закрыть", self.close, False)
        self.close_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

    def _button(self, parent, text, command, primary: bool) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#42d392" if primary else "#27323c",
            fg="#07100b" if primary else "#f3f6f8",
            activebackground="#56e0a4" if primary else "#33414d",
            activeforeground="#07100b" if primary else "#f3f6f8",
            relief=tk.FLAT,
            padx=16,
            pady=14,
            font=("TkDefaultFont", 14, "bold" if primary else "normal"),
        )

    # ---- управление сбором -------------------------------------------------

    def primary_action(self) -> None:
        if self.closed or self.phase in {"settle", "collect"}:
            return
        if self.phase == "done":
            self.close()
            return
        # Старт новой калибровки.
        self.target_index = 0
        self.collected = []
        self._begin_target()

    def _begin_target(self) -> None:
        """Перейти к показу текущей цели и фазе «осмотреться» перед сбором."""
        self.raw_buffer = []
        self.phase = "settle"
        self.phase_until = time.monotonic() + self.SETTLE_SECONDS
        self.primary.config(text="Идёт калибровка...", state=tk.DISABLED)
        self.status.set(
            f"Цель {self.target_index + 1}/{len(self.targets)}: смотрите на мишень"
        )

    def _advance_after_collect(self) -> None:
        """Зафиксировать усреднённый взгляд за текущую цель и перейти к следующей."""
        self.collected.append((average_raw(self.raw_buffer), self.targets[self.target_index]))
        self.target_index += 1
        if self.target_index >= len(self.targets):
            self._finish()
        else:
            self._begin_target()

    def _finish(self) -> None:
        """Собрать пары, проверить качество, подогнать и сохранить калибровку."""
        self.phase = "done"
        self.primary.config(text="Готово", state=tk.NORMAL)
        samples = build_gaze_calibration_samples(self.collected)
        if not gaze_samples_quality_ok(samples):
            self.status.set(
                "Недостаточно данных: взгляд почти не считывался или не двигался. "
                "Сядьте ровнее при хорошем свете и повторите."
            )
            self.primary.config(text="Повторить")
            return
        ok = self.estimator.calibrate(samples)
        if not ok:
            self.status.set("Не удалось подогнать калибровку. Повторите.")
            self.primary.config(text="Повторить")
            return
        self._write_calibration_to_config(self.estimator.calibration, self.cfg.fusion.gaze)
        try:
            self.cfg.save()
        except Exception as exc:
            self.status.set(f"Калибровка готова, но не сохранилась: {exc}")
            return
        cal = self.estimator.calibration
        self.status.set(
            f"Сохранено: x={cal.ax:.2f}*raw+{cal.bx:.2f}, y={cal.ay:.2f}*raw+{cal.by:.2f}"
        )
        self.instruction.set(
            "Калибровка взгляда сохранена. Взгляд остаётся вспомогательным грубым "
            "наведением — в режиме assist рука всегда уточняет позицию."
        )

    # ---- цикл кадров -------------------------------------------------------

    def _schedule_frame(self) -> None:
        if self.closed:
            return
        self._tick()
        self.after_id = self.root.after(16, self._schedule_frame)

    def _tick(self) -> None:
        ok, frame = self.cam.read()
        if ok and frame is not None:
            try:
                self.estimator.estimate(frame)   # обновляет last_raw
            except Exception:
                pass

        now = time.monotonic()
        if self.phase == "settle" and now >= self.phase_until:
            self.phase = "collect"
            self.phase_until = now + self.DWELL_SECONDS
            self.status.set(
                f"Цель {self.target_index + 1}/{len(self.targets)}: удерживайте взгляд"
            )
        elif self.phase == "collect":
            raw = self.estimator.last_raw
            if raw is not None:
                self.raw_buffer.append((float(raw[0]), float(raw[1])))
            if now >= self.phase_until:
                self._advance_after_collect()

        self._draw()

    def _draw(self) -> None:
        self.canvas.delete("all")
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        if self.phase not in {"settle", "collect"}:
            return
        tx, ty = self.targets[self.target_index]
        cx, cy = tx * w, ty * h
        active = self.phase == "collect"
        outer = "#ffd166" if active else "#5a6470"
        inner = "#42d392" if active else "#27323c"
        r = 26
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=outer, width=4)
        self.canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill=inner, outline="")

    # ---- завершение --------------------------------------------------------

    def _teardown_resources(self) -> None:
        if self.cam is not None:
            try:
                self.cam.release()
            except Exception:
                pass
            self.cam = None
        if self.estimator is not None:
            try:
                self.estimator.close()
            except Exception:
                pass
            self.estimator = None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
        self._teardown_resources()
        self.root.destroy()


def run_gaze_calibration(cfg: AppConfig = None) -> None:
    """Запустить мастер калибровки взгляда (отдельное окно).

    Не требует gaze: если модель лица/MediaPipe недоступны, мастер покажет
    пояснение и закроется без падения. Существующая калибровка руки/щипка
    (run_calibration) не затрагивается."""
    cfg = cfg or AppConfig.load()
    root = tk.Tk()
    try:
        GazeCalibrationWizard(root, cfg)
    except Exception as exc:
        try:
            messagebox.showerror(
                "AirControl", f"Не удалось открыть калибровку взгляда:\n{exc}"
            )
        finally:
            root.destroy()
        return
    root.mainloop()
