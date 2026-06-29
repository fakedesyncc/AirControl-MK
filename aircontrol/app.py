"""Главный оркестратор AirControl.

Связывает все подсистемы и ведёт основной цикл (через Tkinter after-loop):

    камера → трекер руки → движок жестов → координатор слияния
                                          ↘ контроллер курсора (фильтр + dwell)
                                          ↘ исполнитель действий (мышь/клавиатура)
                                          ↘ голос (по жесту «кулак»)
    → отрисовка оверлеев → вывод в окно.

Горячие клавиши позволяют переключать фильтр стабилизации и распознаватель
жестов прямо во время работы — это удобно для живой демонстрации сравнений
на защите.
"""

import os
import threading
import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from . import __app_name__, __version__
from .config import AppConfig, apply_dwell_profile, next_dwell_profile
from .control import ActionExecutor, CursorController
from .evaluation.metrics import FPSMeter, TelemetryLogger
from .fusion import MultimodalCoordinator
from .gestures import features as F
from .gestures import GestureEngine
from .gestures.bimanual import BimanualController
from .gestures.engine import FrameGestures
from .gestures.heuristic import HeuristicPoseClassifier
from .gestures.ml import MLPoseClassifier
from .tracking import Camera, HandTracker
from .tracking.hand_tracker import INDEX_TIP, MIDDLE_TIP, PINKY_TIP, RING_TIP
from .ui import renderer as R
from .voice import CommandProcessor, VoiceRecognizer

FILTER_CYCLE = ["none", "ema", "one_euro", "kalman"]


def resolve_key_command(keysym: str = "", char: str = "") -> str | None:
    """Resolve Tk key events to app commands, including Cyrillic keyboard layout."""
    key = (keysym or "").lower()
    ch = (char or "").lower()
    if key in ("escape",) or ch == "\x1b":
        return "close"
    if key in ("f2",):
        return "fitts_gesture"
    if key in ("f3",):
        return "fitts_mouse"
    if key in ("kp_add", "plus", "equal") or ch in ("+", "="):
        return "sensitivity_up"
    if key in ("kp_subtract", "minus") or ch == "-":
        return "sensitivity_down"
    if ch == "1" or key in ("1", "exclam"):
        return "mode_view"
    if ch == "2" or key in ("2", "at"):
        return "mode_control"

    # Same physical QWERTY keys in English and Russian layouts:
    # f/а, g/п, d/в, o/щ, l/д, h/р.
    char_map = {
        "f": "cycle_filter",
        "а": "cycle_filter",
        "g": "toggle_recognizer",
        "п": "toggle_recognizer",
        "d": "toggle_dwell",
        "в": "toggle_dwell",
        "o": "toggle_one_gesture",
        "щ": "toggle_one_gesture",
        "l": "toggle_landmarks",
        "д": "toggle_landmarks",
        "h": "toggle_hud",
        "р": "toggle_hud",
    }
    if ch in char_map:
        return char_map[ch]

    keysym_map = {
        "f": "cycle_filter",
        "cyrillic_a": "cycle_filter",
        "g": "toggle_recognizer",
        "cyrillic_pe": "toggle_recognizer",
        "d": "toggle_dwell",
        "cyrillic_ve": "toggle_dwell",
        "o": "toggle_one_gesture",
        "cyrillic_shcha": "toggle_one_gesture",
        "l": "toggle_landmarks",
        "cyrillic_de": "toggle_landmarks",
        "h": "toggle_hud",
        "cyrillic_er": "toggle_hud",
    }
    return keysym_map.get(key)


class AirControlApp:
    def __init__(self, config: AppConfig | None = None):
        self.cfg = config or AppConfig.load()

        # --- Окно ---
        self.root = tk.Tk()
        self.root.title(f"{__app_name__} {__version__}")
        if self.cfg.ui.frameless:
            self.root.overrideredirect(True)
        self.root.geometry(f"{self.cfg.ui.window_width}x{self.cfg.ui.window_height}"
                            f"+{self.cfg.ui.window_x}+{self.cfg.ui.window_y}")
        self.root.attributes("-topmost", self.cfg.ui.always_on_top)
        self.shell = tk.Frame(self.root, bg="#0f1317")
        self.shell.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.shell, highlightthickness=0, bg="black", takefocus=True)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Один переиспользуемый image-элемент: обновляем его кадром каждый тик
        # вместо delete("all")+create_image (меньше churn в Tk на ~30 Гц).
        self._canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        # --- Подсистемы ---
        self.camera = Camera(self.cfg.camera)
        self.tracker = HandTracker(self.cfg.tracking)
        self.engine = GestureEngine(self.cfg.gestures)
        self.bimanual = (BimanualController(self.cfg.gestures)
                         if self.cfg.gestures.bimanual_enabled else None)
        self.actions = ActionExecutor(
            screenshot_dir=_screenshots_dir(),
            on_toggle_record=self.toggle_recording,
            dry_run=self.cfg.input.dry_run)
        self.cursor = CursorController(self.cfg.cursor, self.cfg.filter,
                                       self.actions.mouse, (self.screen_w, self.screen_h))
        self.commands = CommandProcessor(self.actions, window_callbacks={
            "minimize": lambda: self.root.after(0, self.root.iconify),
            "restore": lambda: self.root.after(0, lambda: (self.root.deiconify(), self.root.lift())),
        })
        self.voice = VoiceRecognizer(self.cfg.voice, self.actions, self.commands)
        self.coordinator = MultimodalCoordinator(self.cfg.fusion, self.actions,
                                                 self.cursor, self.voice)

        # Оценщик взгляда — опциональная ассистивная модальность. Создаём ТОЛЬКО
        # при gaze_enabled; при недоступности модели/MediaPipe он молча
        # отключается (gaze=None), и поведение «только рука» не меняется.
        self.gaze = self._init_gaze_estimator()
        self._last_gaze = None
        self._gaze_phase = 0

        # --- Метрики и визуал ---
        self.fps = FPSMeter()
        self.telemetry = TelemetryLogger(self.cfg.telemetry, self.cfg.filter.type,
                                         self.cfg.gestures.recognizer)
        self.trail = R.TrailRenderer()
        self.particles = R.ParticleSystem()

        # --- Состояние ---
        self.mode = "control" if self.cfg.start_mode == "control" else "view"
        self.running = True
        self.recording = False
        self.video_writer = None
        self.record_start = 0.0
        self._toast = ""
        self._toast_until = 0.0
        self._detect_ms = 0.0
        self._hand_detected = False
        self._auto_tuned = False
        self._deep_auto_tuned = False
        self._low_perf_reason = ""
        self._last_frame_at = 0.0
        self._last_hand_seen_at = 0.0
        self._mode_started_at = time.time()
        self._started_at = time.time()
        # Монитор здоровья камеры: ловит пропажу потока кадров посреди сессии,
        # просит бэкофф (без busy-loop) и поднимает статус для пользователя.
        self._camera_health = CameraHealthMonitor()

        self._setup_window_drag()
        self._setup_controls()
        self._setup_keys()
        self.root.after(100, self._force_keyboard_focus)

        # Потоки: детекция/обработка и движение мыши — отдельно от UI, чтобы
        # плавность курсора не зависела от FPS детекции и отрисовки.
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._mouse_thread = threading.Thread(target=self._mouse_loop, daemon=True)
        self._proc_thread.start()
        self._mouse_thread.start()
        self._tick()

    # ------------------------------------------------------------------ окно

    def _setup_window_drag(self):
        self._drag_x = self._drag_y = 0
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Button-2>", lambda e: self.close())

    def _setup_controls(self):
        self._control_buttons = {}
        if not self.cfg.ui.show_controls:
            return
        bar = tk.Frame(self.shell, bg="#111820", padx=6, pady=5)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        def add(name: str, text: str, command, primary: bool = False):
            def run_command():
                try:
                    command()
                finally:
                    self.root.after(10, self._force_keyboard_focus)

            btn = tk.Button(
                bar,
                text=text,
                command=run_command,
                bg="#43d17d" if primary else "#26313b",
                fg="#07100b" if primary else "#f1f5f8",
                activebackground="#55e28e" if primary else "#33414d",
                activeforeground="#07100b" if primary else "#ffffff",
                relief=tk.FLAT,
                padx=10,
                pady=6,
                font=("TkDefaultFont", 11, "bold" if primary else "normal"),
                takefocus=False,
            )
            btn.pack(side=tk.LEFT, padx=3)
            self._control_buttons[name] = btn

        add("mode", "Control", self.toggle_mode, primary=True)
        add("safe", "Safe", self.toggle_safe_input)
        add("dwell", "Dwell", self.toggle_dwell)
        add("one_gesture", "One", self.toggle_one_gesture_mode)
        add("dwell_profile", "Dwell profile", self.cycle_dwell_profile)
        add("minus", "-", lambda: self.change_sensitivity(-0.1))
        add("plus", "+", lambda: self.change_sensitivity(0.1))
        add("report", "Report", self.save_diagnostic_report)
        add("exit", "Exit", self.close)
        self._refresh_controls()

    def _force_keyboard_focus(self):
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.canvas.focus_set()
        except Exception:
            pass

    def _on_press(self, e):
        try:
            self.canvas.focus_set()
        except Exception:
            pass
        self._drag_x, self._drag_y = e.x, e.y

    def _on_drag(self, e):
        x = self.root.winfo_x() + (e.x - self._drag_x)
        y = self.root.winfo_y() + (e.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    def _setup_keys(self):
        self.root.bind_all("<KeyPress>", self._on_key_press)

    def _on_key_press(self, event):
        command = resolve_key_command(
            getattr(event, "keysym", ""),
            getattr(event, "char", ""),
        )
        if not command:
            return None
        self._dispatch_key_command(command)
        return "break"

    def _dispatch_key_command(self, command: str) -> None:
        handlers = {
            "mode_view": lambda: self.set_mode("view"),
            "mode_control": lambda: self.set_mode("control"),
            "sensitivity_up": lambda: self.change_sensitivity(0.1),
            "sensitivity_down": lambda: self.change_sensitivity(-0.1),
            "close": self.close,
            "cycle_filter": self.cycle_filter,
            "toggle_recognizer": self.toggle_recognizer,
            "toggle_dwell": self.toggle_dwell,
            "toggle_one_gesture": self.toggle_one_gesture_mode,
            "toggle_landmarks": lambda: self._toggle("show_landmarks"),
            "toggle_hud": lambda: self._toggle("show_hud"),
            "fitts_gesture": lambda: self.launch_fitts("gesture"),
            "fitts_mouse": lambda: self.launch_fitts("mouse"),
        }
        handler = handlers.get(command)
        if handler:
            handler()

    # ----------------------------------------------------------- управление

    def set_mode(self, mode: str):
        self.mode = mode
        self._mode_started_at = time.time()
        if mode == "view":
            self.coordinator.shutdown()
            self.trail.clear()
            self.particles.clear()
        self._refresh_controls()
        self._toast_msg(f"mode: {mode}")

    def toggle_mode(self):
        self.set_mode("view" if self.mode == "control" else "control")

    def toggle_safe_input(self):
        self.cfg.input.dry_run = not self.cfg.input.dry_run
        self.actions.set_dry_run(self.cfg.input.dry_run)
        self.cursor.set_mouse(self.actions.mouse)
        self._refresh_controls()
        self._toast_msg(f"safe input: {'ON' if self.cfg.input.dry_run else 'OFF'}")

    def change_sensitivity(self, delta: float):
        self.cfg.cursor.sensitivity = max(0.2, round(self.cfg.cursor.sensitivity + delta, 2))
        self._refresh_controls()
        self._toast_msg(f"sensitivity x{self.cfg.cursor.sensitivity:.2f}")

    def cycle_filter(self):
        i = (FILTER_CYCLE.index(self.cfg.filter.type) + 1) % len(FILTER_CYCLE) \
            if self.cfg.filter.type in FILTER_CYCLE else 0
        self.cfg.filter.type = FILTER_CYCLE[i]
        self.cursor.set_filter(self.cfg.filter)
        self.telemetry.filter_type = self.cfg.filter.type
        self._toast_msg(f"filter: {self.cfg.filter.type}")

    def toggle_recognizer(self):
        if self.engine.classifier.name == "heuristic":
            clf = MLPoseClassifier.load(self.cfg.gestures.ml_model_path)
            if clf is None:
                self._toast_msg("ML-модель не обучена (см. train)")
                return
            self.engine.set_classifier(clf)
            self.cfg.gestures.recognizer = "ml"
        else:
            self.engine.set_classifier(HeuristicPoseClassifier())
            self.cfg.gestures.recognizer = "heuristic"
        self.telemetry.recognizer = self.cfg.gestures.recognizer
        self._toast_msg(f"recognizer: {self.cfg.gestures.recognizer}")

    def toggle_dwell(self):
        self.cfg.cursor.dwell_enabled = not self.cfg.cursor.dwell_enabled
        if self.cfg.cursor.dwell_enabled and self.cfg.cursor.dwell_profile == "custom":
            apply_dwell_profile(self.cfg, "normal")
        self._refresh_controls()
        self._toast_msg(f"dwell-click: {'ON' if self.cfg.cursor.dwell_enabled else 'OFF'}")

    def toggle_one_gesture_mode(self):
        self.cfg.gestures.dwell_only_mode = not self.cfg.gestures.dwell_only_mode
        if self.cfg.gestures.dwell_only_mode and not self.cfg.cursor.dwell_enabled:
            apply_dwell_profile(self.cfg, "normal")
        bimanual = getattr(self, "bimanual", None)
        if bimanual is not None and self.cfg.gestures.dwell_only_mode:
            bimanual.reset()
        self._refresh_controls()
        self._toast_msg(
            f"one gesture: {'ON' if self.cfg.gestures.dwell_only_mode else 'OFF'}"
        )

    def cycle_dwell_profile(self):
        profile = next_dwell_profile(getattr(self.cfg.cursor, "dwell_profile", "custom"))
        apply_dwell_profile(self.cfg, profile)
        self._refresh_controls()
        self._toast_msg(
            f"dwell profile: {profile} "
            f"({self.cfg.cursor.dwell_time:.2f}s, {self.cfg.cursor.dwell_radius}px, "
            f"cooldown {self.cfg.cursor.dwell_cooldown:.2f}s)"
        )

    def _toggle(self, attr: str):
        setattr(self.cfg.ui, attr, not getattr(self.cfg.ui, attr))
        self._toast_msg(f"{attr}: {getattr(self.cfg.ui, attr)}")

    def launch_fitts(self, method: str):
        """Запуск теста Фиттса в отдельном процессе (не блокирует окно камеры)."""
        import subprocess
        import sys
        self._toast_msg(f"Fitts test ({method}) — отдельное окно")
        subprocess.Popen([sys.executable, "-m", "aircontrol.evaluation.fitts_runner",
                          "--method", method,
                          "--participant", self.cfg.evaluation.participant_id])

    def save_diagnostic_report(self):
        try:
            from .diagnostics import save_support_bundle
            path = save_support_bundle(scan_camera=False,
                                       runtime_info=self.runtime_status())
            self._toast_msg(f"support report saved: {path}")
        except Exception as exc:
            self._toast_msg(f"support report failed: {exc}")

    def runtime_status(self) -> dict:
        now = time.time()
        seconds_since_hand = None if not self._last_hand_seen_at else round(now - self._last_hand_seen_at, 2)
        seconds_since_frame = None if not self._last_frame_at else round(now - self._last_frame_at, 2)
        seconds_since_action = (
            None if not self.actions.last_action_time
            else round(now - self.actions.last_action_time, 2)
        )
        seconds_since_input_error = (
            None if not self.actions.last_input_error_time
            else round(now - self.actions.last_input_error_time, 2)
        )
        input_status = self.actions.input_status()
        health_lines = build_runtime_health_lines(
            mode=self.mode,
            input_status=input_status,
            last_input_error=self.actions.last_input_error,
            input_error_age=seconds_since_input_error,
            fps=self.fps.fps,
            detect_ms=self._detect_ms,
            auto_tuned=self._auto_tuned,
            detect_max_fps=self.cfg.performance.detect_max_fps,
            low_perf_reason=self._low_perf_reason,
            last_frame_age=seconds_since_frame,
            hand_detected=self._hand_detected,
            mode_age=round(now - self._mode_started_at, 2),
            camera_lost=self._camera_health.lost,
        )
        return {
            "mode": self.mode,
            "start_mode": self.cfg.start_mode,
            "profile": self.cfg.profile_name,
            "assistive_preset": self.cfg.assistive_preset,
            "safe_input": self.cfg.input.dry_run,
            "dwell_enabled": self.cfg.cursor.dwell_enabled,
            "dwell_profile": self.cfg.cursor.dwell_profile,
            "dwell_cooldown": self.cfg.cursor.dwell_cooldown,
            "dwell_only_mode": self.cfg.gestures.dwell_only_mode,
            "swipe_backend": self.cfg.gestures.swipe_backend,
            "voice_engine": self.cfg.voice.engine,
            "gaze_enabled": self.cfg.fusion.gaze_enabled,
            "gaze_mode": self.cfg.fusion.gaze_mode,
            "last_action": self.actions.last_action,
            "seconds_since_action": seconds_since_action,
            "last_input_error": self.actions.last_input_error,
            "seconds_since_input_error": seconds_since_input_error,
            "input_error_count": self.actions.input_error_count,
            "seconds_in_mode": round(now - self._mode_started_at, 2),
            "fps": round(self.fps.fps, 2),
            "detect_ms": round(self._detect_ms, 2),
            "hand_detected": self._hand_detected,
            "seconds_since_hand": seconds_since_hand,
            "seconds_since_frame": seconds_since_frame,
            "camera_lost": self._camera_health.lost,
            "camera_read_failures": self._camera_health.consecutive_failures,
            "input_status": input_status,
            "health_lines": health_lines,
            "auto_tuned": self._auto_tuned,
            "deep_auto_tuned": self._deep_auto_tuned,
            "low_perf_reason": self._low_perf_reason,
            "performance": {
                "detect_downscale": self.cfg.performance.detect_downscale,
                "detect_max_fps": self.cfg.performance.detect_max_fps,
                "show_landmarks": self.cfg.ui.show_landmarks,
                "show_particles": self.cfg.ui.show_particles,
                "show_trail": self.cfg.ui.show_trail,
            },
        }

    def _refresh_controls(self):
        buttons = getattr(self, "_control_buttons", None)
        if not buttons:
            return
        if "mode" in buttons:
            buttons["mode"].configure(
                text="Control" if self.mode == "control" else "View",
                bg="#43d17d" if self.mode == "control" else "#ffd166",
                fg="#07100b",
            )
        if "safe" in buttons:
            buttons["safe"].configure(
                text="Safe ON" if self.cfg.input.dry_run else "Safe OFF",
                bg="#ffd166" if self.cfg.input.dry_run else "#26313b",
                fg="#07100b" if self.cfg.input.dry_run else "#f1f5f8",
            )
        if "dwell" in buttons:
            buttons["dwell"].configure(
                text="Dwell ON" if self.cfg.cursor.dwell_enabled else "Dwell OFF",
                bg="#43d17d" if self.cfg.cursor.dwell_enabled else "#26313b",
                fg="#07100b" if self.cfg.cursor.dwell_enabled else "#f1f5f8",
            )
        if "one_gesture" in buttons:
            buttons["one_gesture"].configure(
                text="One ON" if self.cfg.gestures.dwell_only_mode else "One OFF",
                bg="#43d17d" if self.cfg.gestures.dwell_only_mode else "#26313b",
                fg="#07100b" if self.cfg.gestures.dwell_only_mode else "#f1f5f8",
            )
        if "dwell_profile" in buttons:
            profile = getattr(self.cfg.cursor, "dwell_profile", "custom")
            buttons["dwell_profile"].configure(
                text=f"Dwell {profile}",
                bg="#43d17d" if self.cfg.cursor.dwell_enabled else "#26313b",
                fg="#07100b" if self.cfg.cursor.dwell_enabled else "#f1f5f8",
            )
        if "minus" in buttons:
            buttons["minus"].configure(text=f"-  x{self.cfg.cursor.sensitivity:.1f}")
        if "plus" in buttons:
            buttons["plus"].configure(text="+")

    # ---------------------------------------------------------------- запись

    def toggle_recording(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        import os
        from .config import RECORDINGS_DIR
        rec_dir = RECORDINGS_DIR
        os.makedirs(rec_dir, exist_ok=True)
        path = os.path.join(rec_dir, f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(path, fourcc, 30.0,
                                            (self.cfg.camera.width, self.cfg.camera.height))
        self.recording = True
        self.record_start = time.time()
        self._toast_msg("🔴 запись начата")

    def _stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        self._toast_msg("⏹ запись остановлена")

    # ------------------------------------------------------------ потоки

    _last_pose = "none"

    def _mouse_loop(self):
        """Высокочастотное (~150 Гц) плавное ведение курсора к цели."""
        while self.running:
            try:
                if self.mode == "control":
                    self.cursor.step()
            except Exception:
                pass
            time.sleep(1.0 / 150.0)

    def _process_loop(self):
        """Захват кадра + детекция + обработка жестов. Работает в своём темпе,
        не привязан к отрисовке UI."""
        while self.running:
            t0 = time.time()
            try:
                ok, frame = self.camera.read()
                if not ok:
                    # Камера отдала пустой кадр: копим неудачу, спим запрошенный
                    # бэкофф (чтобы не крутить цикл вхолостую) и идём на новый круг.
                    # Решение «показывать ли статус» считает чистый монитор —
                    # при стойкой пропаже пользователь увидит «Камера потеряна».
                    time.sleep(self._camera_health.record_failure())
                    continue
                # Камера вернулась/жива: снимаем статус «потеряна», сбрасываем счётчик.
                self._camera_health.record_success()
                self._last_frame_at = time.time()
                if self.mode == "control":
                    frame = self._process_control(frame)
                else:
                    frame = self._process_preview(frame)
                    self.cursor.target = None
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(cv2.resize(
                        frame, (self.cfg.camera.width, self.cfg.camera.height)))
                with self._frame_lock:
                    self._latest_frame = frame
                self.fps.tick(t0)
                self.telemetry.maybe_log(
                    self.fps.fps, (time.time() - t0) * 1000.0,
                    self._detect_ms, self._last_pose, self.mode)
                self._maybe_auto_tune()
                # Троттлинг детекции (экономия CPU на слабых устройствах).
                max_fps = self.cfg.performance.detect_max_fps
                if max_fps > 0:
                    rest = (1.0 / max_fps) - (time.time() - t0)
                    if rest > 0:
                        time.sleep(rest)
            except Exception as exc:
                import traceback
                print(f"[app] Ошибка в кадре: {exc}")
                traceback.print_exc()
                time.sleep(0.01)

    def _init_gaze_estimator(self):
        """Создать оценщик взгляда, если фича включена и доступна.

        Любой сбой (нет модели, нет MediaPipe) → None и тёплое сообщение, без
        падения: gaze-режим просто не активируется."""
        if not getattr(self.cfg.fusion, "gaze_enabled", False):
            return None
        try:
            from .tracking.gaze import GazeEstimator
            estimator = GazeEstimator(self.cfg.fusion.gaze)
        except Exception as exc:
            print(f"[gaze] Оценщик взгляда не создан: {exc}")
            return None
        if not estimator.ready:
            print(f"[gaze] Взгляд отключён: {estimator.init_error}")
            return None
        print(f"[gaze] Оценщик взгляда готов (режим '{self.cfg.fusion.gaze_mode}')")
        return estimator

    def _estimate_gaze(self, frame):
        """Оценка взгляда на текущем кадре с троттлингом (взгляд — грубый сигнал,
        нет смысла гонять модель лица на каждом кадре). Возвращает последнюю
        валидную оценку между прогонами; None — когда фича выключена."""
        if self.gaze is None:
            return None
        # Запускаем модель лица реже детекции руки (каждый 2-й кадр) — экономия CPU.
        self._gaze_phase = (self._gaze_phase + 1) % 2
        if self._gaze_phase == 0 or self._last_gaze is None:
            self._last_gaze = self.gaze.estimate(frame, time.time())
        return self._last_gaze

    def _detect_hands_for_current_frame(self, frame):
        t0 = time.time()
        ds = self.cfg.performance.detect_downscale
        det_frame = frame
        if ds and ds < 0.999:
            det_frame = cv2.resize(frame, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
        try:
            hands = self.tracker.detect(det_frame)
        except Exception as exc:
            # Разовый сбой детекции (битый кадр, икота MediaPipe) не должен ронять
            # цикл — считаем кадр «без рук» и едем дальше.
            print(f"[app] Сбой детекции, кадр пропущен: {exc}")
            hands = []
        self._detect_ms = (time.time() - t0) * 1000.0
        self._hand_detected = bool(hands)
        if hands:
            self._last_hand_seen_at = time.time()
        return hands

    def _tick(self):
        """Только отрисовка в окне (~30 Гц), на главном потоке Tkinter."""
        if not self.running:
            return
        try:
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is not None:
                self._render_chrome(frame)
                self._show(frame)
        except Exception as exc:
            print(f"[app] Ошибка отрисовки: {exc}")
        finally:
            self.root.after(33, self._tick)

    def _process_control(self, frame):
        hands = self._detect_hands_for_current_frame(frame)
        # Двуручный pinch-to-zoom обрабатываем первым: если он активен, одноручные
        # жесты подавляются (иначе щипок ведущей руки начал бы перетаскивание).
        one_gesture = self.cfg.gestures.dwell_only_mode
        if self.bimanual and one_gesture:
            self.bimanual.reset()
        zoom_actions = (
            self.bimanual.process(hands)
            if self.bimanual and not one_gesture else []
        )
        bimanual_active = self.bimanual.engaged if self.bimanual else False
        if one_gesture:
            bimanual_active = False

        hand = None if bimanual_active else self._pick_primary(hands)
        fg = self.engine.process(hand)
        self._last_pose = fg.pose
        gaze = None if bimanual_active else self._estimate_gaze(frame)
        self.coordinator.process(fg, time.time(), gaze=gaze)
        for act in zoom_actions:
            self.actions.execute(act)

        # Визуализация.
        h, w = frame.shape[:2]
        if self.cfg.ui.show_landmarks:
            for hnd in hands:
                R.draw_landmarks(frame, hnd)
        if self.bimanual is not None and self.bimanual.engaged and self.bimanual.points:
            (c1, c2) = self.bimanual.points
            import cv2 as _cv2
            p1 = (int(c1[0] * w), int(c1[1] * h)); p2 = (int(c2[0] * w), int(c2[1] * h))
            _cv2.line(frame, p1, p2, (255, 0, 255), 2)
            _cv2.putText(frame, "ZOOM", ((p1[0] + p2[0]) // 2 - 25, (p1[1] + p2[1]) // 2),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if hand is not None and fg.cursor_norm and not fg.frozen:
            self.trail.add(int(fg.cursor_norm[0] * w), int(fg.cursor_norm[1] * h))
        if self.cfg.ui.show_trail:
            self.trail.draw(frame)
        if self.cfg.ui.show_particles:
            self.particles.draw(frame)
        # Эффект-частицы на скриншот/запись.
        for ev in fg.events:
            if ev.action in ("screenshot", "toggle_record") and fg.cursor_norm:
                self.particles.burst(int(fg.cursor_norm[0] * w), int(fg.cursor_norm[1] * h),
                                     (0, 255, 255) if ev.action == "screenshot" else (255, 0, 0))
        R.draw_pose_badge(frame, fg)
        R.draw_dwell_ring(frame, fg, self.cursor.dwell_progress)
        return frame

    def _process_preview(self, frame):
        """Безопасный view-режим: показываем трекинг/позу, но не исполняем ввод."""
        hands = self._detect_hands_for_current_frame(frame)
        hand = self._pick_primary(hands)
        fg = build_preview_gestures(self.engine.classifier, hand)
        self._last_pose = fg.pose if fg.hand_detected else "view"
        h, w = frame.shape[:2]
        if self.cfg.ui.show_landmarks:
            for hnd in hands:
                R.draw_landmarks(frame, hnd)
        if fg.cursor_norm:
            self.trail.add(int(fg.cursor_norm[0] * w), int(fg.cursor_norm[1] * h))
        else:
            self.trail.clear()
        if self.cfg.ui.show_trail:
            self.trail.draw(frame)
        self.particles.clear()
        R.draw_pose_badge(frame, fg)
        R.draw_dwell_ring(frame, fg, 0.0)
        return frame

    def _maybe_auto_tune(self):
        elapsed = time.time() - self._started_at
        if elapsed < 6.0:
            return
        if self.fps.fps <= 0:
            return
        reasons = []
        if self.fps.fps < 18.0:
            reasons.append(f"FPS {self.fps.fps:.1f}")
        if self._detect_ms > 70.0:
            reasons.append(f"detect {self._detect_ms:.0f}ms")
        if not reasons:
            return
        if self._auto_tuned and not (elapsed > 18.0 and self.fps.fps < 14.0):
            return
        if self._auto_tuned and self._deep_auto_tuned:
            return
        self._low_perf_reason = ", ".join(reasons)
        if self._auto_tuned:
            self._deep_auto_tuned = True
            apply_runtime_performance_tune(self.cfg, deep=True)
            self._toast_msg(f"very low performance mode: ON ({self._low_perf_reason})")
            return
        self._auto_tuned = True
        apply_runtime_performance_tune(self.cfg, deep=False)
        self._toast_msg(f"low performance mode: ON ({self._low_perf_reason})")

    def _pick_primary(self, hands):
        """Выбирает ведущую руку (самую правую на экране) — стабильнее для курсора."""
        if not hands:
            return None
        return max(hands, key=lambda h: float(h.landmarks[0][0]))

    def _render_chrome(self, frame):
        if self.recording:
            R.draw_recording(frame, self.record_start)
        if self.cfg.ui.show_metrics:
            R.draw_metrics(frame, self.fps.fps, self.cfg.filter.type,
                           self.cfg.gestures.recognizer)
        R.draw_mode_status(frame, self.mode, self._hand_detected, self._detect_ms)
        # input_status() и время берём ОДИН раз за кадр и переиспользуем
        # (раньше input_status() считался дважды, а на Linux он дёргает shutil.which).
        now = time.time()
        input_status = self.actions.input_status()
        health_lines = self._runtime_health_lines(now=now, input_status=input_status)
        R.draw_runtime_health(frame, health_lines)
        if self.cfg.ui.show_hud:
            R.draw_hud(frame, self.mode)
        R.draw_assistive_status(
            frame,
            profile=self.cfg.profile_name,
            assistive_preset=self.cfg.assistive_preset,
            input_status=input_status,
            dwell_enabled=self.cfg.cursor.dwell_enabled,
            dwell_only_mode=self.cfg.gestures.dwell_only_mode,
            last_action=self.actions.last_action,
            last_action_age=now - self.actions.last_action_time
            if self.actions.last_action_time else None,
            last_input_error=self.actions.last_input_error,
            last_input_error_age=now - self.actions.last_input_error_time
            if self.actions.last_input_error_time else None,
        )
        if now < self._toast_until:
            toast_y = 176 if health_lines else 96
            cv2.putText(frame, self._toast, (15, toast_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    def _runtime_health_lines(self, now=None, input_status=None) -> list[str]:
        if now is None:
            now = time.time()
        if input_status is None:
            input_status = self.actions.input_status()
        last_frame_age = None if not self._last_frame_at else now - self._last_frame_at
        return build_runtime_health_lines(
            mode=self.mode,
            input_status=input_status,
            last_input_error=self.actions.last_input_error,
            input_error_age=now - self.actions.last_input_error_time
            if self.actions.last_input_error_time else None,
            fps=self.fps.fps,
            detect_ms=self._detect_ms,
            auto_tuned=self._auto_tuned,
            detect_max_fps=self.cfg.performance.detect_max_fps,
            low_perf_reason=self._low_perf_reason,
            last_frame_age=last_frame_age,
            hand_detected=self._hand_detected,
            mode_age=now - self._mode_started_at,
            camera_lost=self._camera_health.lost,
        )

    def _show(self, frame):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 1 and ch > 1:
            disp = cv2.resize(frame, (cw, ch))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.canvas.itemconfig(self._canvas_img_id, image=imgtk)
            self.canvas.imgtk = imgtk

    def _toast_msg(self, msg: str):
        self._toast = msg
        self._toast_until = time.time() + 2.0
        print(f"[app] {msg}")

    # --------------------------------------------------------------- выход

    def close(self):
        self.running = False
        # Дать рабочим потокам завершиться, прежде чем освобождать камеру.
        for th in (getattr(self, "_proc_thread", None), getattr(self, "_mouse_thread", None)):
            if th is not None and th.is_alive():
                th.join(timeout=1.0)
        self.coordinator.shutdown()
        if self.recording:
            self._stop_recording()
        self.telemetry.close()
        self.camera.release()
        self.tracker.close()
        if getattr(self, "gaze", None) is not None:
            self.gaze.close()
        try:
            self._save_window_geometry()
            self.cfg.save()
        except Exception as exc:
            print(f"[app] Не удалось сохранить конфиг при выходе: {exc}")
        try:
            self.root.destroy()
        except Exception:
            pass

    def _save_window_geometry(self):
        self.cfg.ui.window_width = self.root.winfo_width()
        self.cfg.ui.window_height = self.root.winfo_height()
        self.cfg.ui.window_x = self.root.winfo_x()
        self.cfg.ui.window_y = self.root.winfo_y()

    def run(self):
        self.root.mainloop()


def _screenshots_dir() -> str:
    from .config import SCREENSHOTS_DIR
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    return SCREENSHOTS_DIR


def build_preview_gestures(classifier, hand) -> FrameGestures:
    """Classify one hand for view-mode overlays without changing input state."""
    if hand is None:
        return FrameGestures()
    lm = hand.landmarks
    pose, confidence = classifier.predict(lm)
    return FrameGestures(
        hand_detected=True,
        cursor_norm=F.palm_center(lm),
        pose=pose,
        pose_confidence=confidence,
        frozen=pose == "open_palm",
        pinch_ratios={
            "index": F.pinch_ratio(lm, INDEX_TIP),
            "middle": F.pinch_ratio(lm, MIDDLE_TIP),
            "ring": F.pinch_ratio(lm, RING_TIP),
            "pinky": F.pinch_ratio(lm, PINKY_TIP),
        },
    )


# Сообщение пользователю, когда поток с камеры пропал посреди сессии.
CAMERA_LOST_STATUS = "Камера потеряна — переподключите"


def decide_camera_health(
    *,
    consecutive_failures: int,
    elapsed_since_last_ok: float | None,
    fail_threshold: int = 5,
    fail_seconds: float = 1.0,
) -> tuple[bool, float]:
    """Чистая, тестируемая логика деградации захвата кадра.

    На вход — счётчик подряд идущих неудачных camera.read() и время с момента
    последнего успешного кадра. На выход — кортеж:
      * show_lost  — показывать ли пользователю статус «камера потеряна»;
      * backoff    — сколько секунд поспать перед следующей попыткой (0 — не спать).

    Никаких побочных эффектов, никакой камеры и Tk: можно гонять в юнит-тестах.
    Статус «потеряна» включается только когда камера действительно подвисла —
    либо подряд накопилось fail_threshold неудач, либо с последнего кадра прошло
    больше fail_seconds. Бэкофф запрашивается на любой неудаче, чтобы не крутить
    busy-loop, и слегка растёт (но ограничен), пока камера не вернётся."""
    if consecutive_failures <= 0:
        # Камера в норме: статуса нет, спать не нужно.
        return False, 0.0
    over_count = consecutive_failures >= fail_threshold
    over_time = elapsed_since_last_ok is not None and elapsed_since_last_ok >= fail_seconds
    show_lost = over_count or over_time
    # Мягкий бэкофф: 5 мс пока сбоев мало, до 50 мс когда камера явно пропала.
    backoff = 0.05 if show_lost else 0.005
    return show_lost, backoff


class CameraHealthMonitor:
    """Крошечный накопитель состояния поверх decide_camera_health().

    Держит счётчик подряд идущих неудач и время последнего успешного кадра,
    решает — показывать ли статус и сколько спать. Не знает ни про камеру, ни про
    Tk, поэтому полностью юнит-тестируется без железа."""

    def __init__(self, *, fail_threshold: int = 5, fail_seconds: float = 1.0):
        self.fail_threshold = fail_threshold
        self.fail_seconds = fail_seconds
        self.consecutive_failures = 0
        self.last_ok_at: float | None = None
        self.lost = False

    def record_success(self, now: float | None = None) -> None:
        """Успешный кадр: сбрасываем счётчик и снимаем статус «потеряна»."""
        if now is None:
            now = time.time()
        self.consecutive_failures = 0
        self.last_ok_at = now
        self.lost = False

    def record_failure(self, now: float | None = None) -> float:
        """Неудачный camera.read(): копим счётчик, обновляем статус и возвращаем
        рекомендованное время сна (бэкофф), чтобы цикл не крутился вхолостую."""
        if now is None:
            now = time.time()
        self.consecutive_failures += 1
        elapsed = None if self.last_ok_at is None else now - self.last_ok_at
        self.lost, backoff = decide_camera_health(
            consecutive_failures=self.consecutive_failures,
            elapsed_since_last_ok=elapsed,
            fail_threshold=self.fail_threshold,
            fail_seconds=self.fail_seconds,
        )
        return backoff

    @property
    def status_line(self) -> str | None:
        """Готовая строка для оверлея, либо None когда всё хорошо."""
        return CAMERA_LOST_STATUS if self.lost else None


def build_runtime_health_lines(
    *,
    mode: str,
    input_status: str,
    fps: float,
    detect_ms: float,
    auto_tuned: bool,
    last_frame_age: float | None,
    hand_detected: bool,
    mode_age: float,
    detect_max_fps: int = 0,
    low_perf_reason: str = "",
    last_input_error: str = "",
    input_error_age: float | None = None,
    camera_lost: bool = False,
) -> list[str]:
    """Human-readable runtime warnings for both control and safe preview modes."""
    lines: list[str] = []
    if camera_lost:
        # Самое важное сообщение — наверх, чтобы пользователь сразу понял причину
        # «замороженного» кадра вместо немого зависания.
        lines.append(CAMERA_LOST_STATUS)
    if input_status == "INPUT OFF":
        lines.append("Input OFF: gestures are detected, but OS control is unavailable")
    elif input_status == "INPUT ERROR":
        detail = f": {last_input_error}" if last_input_error else ""
        lines.append(f"Input ERROR{detail}")
    elif input_status == "INPUT RISK":
        lines.append("Input RISK: this Linux session may block clicks and keys")
    if last_input_error and input_status != "INPUT ERROR":
        if input_error_age is None or input_error_age <= 30.0:
            lines.append(f"Input error: {last_input_error}")
    if last_frame_age is not None and last_frame_age > 2.0:
        lines.append("Camera: no recent frames")
    if mode != "view" and not hand_detected and mode_age > 3.0:
        lines.append("Hand not found: show the full palm inside the camera frame")
    if fps and fps < 18.0:
        if _is_expected_capped_detection_fps(fps, auto_tuned, detect_max_fps):
            lines.append(f"Light mode ON: detection capped at {detect_max_fps} FPS")
        else:
            lines.append(f"Low FPS {fps:.1f}: automatic light mode {'ON' if auto_tuned else 'pending'}")
    if detect_ms > 70.0:
        lines.append(f"Slow detection {detect_ms:.0f}ms: reduce resolution or lighting load")
    if auto_tuned and not any(line.startswith(("Low FPS", "Slow detection")) for line in lines):
        reason = f" ({low_perf_reason})" if low_perf_reason else ""
        lines.append(f"Low performance mode is ON{reason}")
    return lines


def _is_expected_capped_detection_fps(fps: float, auto_tuned: bool, detect_max_fps: int) -> bool:
    if not auto_tuned or detect_max_fps <= 0:
        return False
    return fps >= detect_max_fps * 0.85


def apply_runtime_performance_tune(cfg: AppConfig, *, deep: bool = False) -> AppConfig:
    """Apply runtime auto-tune values after low FPS or slow detection is observed."""
    assistive = cfg.profile_name == "assistive"
    if deep or assistive:
        cfg.performance.detect_downscale = min(cfg.performance.detect_downscale, 0.4)
        cfg.performance.detect_max_fps = 16
        cfg.ui.show_landmarks = False
    else:
        cfg.performance.detect_downscale = min(cfg.performance.detect_downscale, 0.5)
        cfg.performance.detect_max_fps = 20
    cfg.ui.show_particles = False
    cfg.ui.show_trail = False
    return cfg
