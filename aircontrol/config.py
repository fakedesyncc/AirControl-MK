"""Центральная конфигурация AirControl.

Все настраиваемые параметры собраны в одном месте в виде вложенных dataclass-ов
и сериализуются в JSON. Это позволяет:
  * хранить пользовательские профили и результаты калибровки;
  * менять поведение системы без правки кода (важно для экспериментов —
    например, переключать фильтр стабилизации между прогонами теста Фиттса);
  * воспроизводить условия эксперимента (config = часть методики).
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict

_FROZEN = getattr(sys, "frozen", False)   # True внутри PyInstaller-бандла
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_PKG_DIR)


def _user_base_dir() -> str:
    """Пользовательский каталог для записи (конфиг, логи, модели, скриншоты).

    В упакованном приложении пакет лежит в read-only бандле, поэтому пишем в
    ~/.aircontrol. В режиме разработки — рядом с пакетом."""
    if _FROZEN:
        d = os.path.join(os.path.expanduser("~"), ".aircontrol")
    else:
        d = _PKG_DIR
    return d


def _bundled_model_path() -> str:
    """Путь к модели hand_landmarker.task: из бандла (_MEIPASS) или из проекта."""
    if _FROZEN and hasattr(sys, "_MEIPASS"):
        p = os.path.join(sys._MEIPASS, "hand_landmarker.task")
        if os.path.exists(p):
            return p
    return _asset_path("hand_landmarker.task")


def _bundled_face_model_path() -> str:
    """Путь к модели face_landmarker.task: из бандла (_MEIPASS) или из проекта.

    Модель опциональна (нужна только для gaze-режима), поэтому отсутствие файла
    не является ошибкой — оценщик взгляда просто не активируется."""
    if _FROZEN and hasattr(sys, "_MEIPASS"):
        p = os.path.join(sys._MEIPASS, "face_landmarker.task")
        if os.path.exists(p):
            return p
    return _asset_path("face_landmarker.task")


def _asset_path(name: str) -> str:
    candidates = [
        os.path.join(_PROJECT_DIR, name),
        os.path.join(sys.prefix, "share", "aircontrol", name),
        os.path.join(sys.base_prefix, "share", "aircontrol", name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


# Каталоги данных (профили, датасеты жестов, логи, скриншоты, записи).
DATA_DIR = os.path.join(_user_base_dir(), "data")
SCREENSHOTS_DIR = os.path.join(_user_base_dir(), "screenshots") if _FROZEN \
    else os.path.join(_PROJECT_DIR, "screenshots")
RECORDINGS_DIR = os.path.join(_user_base_dir(), "recordings") if _FROZEN \
    else os.path.join(_PROJECT_DIR, "recordings")
DEFAULT_CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
DEFAULT_MODEL_PATH = _bundled_model_path()
DEFAULT_FACE_MODEL_PATH = _bundled_face_model_path()
DEFAULT_ML_MODEL_PATH = os.path.join(DATA_DIR, "gesture_model.npz")
DEFAULT_ML_DATASET_PATH = os.path.join(DATA_DIR, "gesture_dataset.npz")
DEFAULT_TEMPORAL_SWIPE_MODEL_PATH = os.path.join(DATA_DIR, "swipe_temporal_model.npz")
DEFAULT_LOG_DIR = os.path.join(DATA_DIR, "logs")
DEFAULT_VOSK_MODEL_PATH = os.path.join(DATA_DIR, "vosk-model")


@dataclass
class CameraConfig:
    index: int | None = None          # None = автопоиск встроенной камеры
    prefer_builtin: bool = True       # игнорировать телефон/USB-камеры
    backend: str = "auto"             # "auto" | "v4l2" | "any" (Linux: v4l2 стабильнее)
    scan_indices: int = 4             # сколько индексов проверять при автопоиске
    # 640x480 — детекция MediaPipe заметно быстрее, чем на 720p (меньше лага).
    width: int = 640
    height: int = 480
    flip_horizontal: bool = True      # зеркалить кадр (естественнее для пользователя)
    target_fps: int = 60
    buffer_size: int = 1              # короткий буфер уменьшает лаг на Linux/V4L2
    fourcc: str = "MJPG"              # часто разгружает USB-камеры на Linux
    reopen_delay: float = 0.7         # сек между попытками переоткрыть камеру


@dataclass
class TrackingConfig:
    model_path: str = DEFAULT_MODEL_PATH
    # 1 рука — быстрее и стабильнее для управления курсором. Для двуручного
    # zoom поставьте 2 (это нагружает CPU сильнее).
    num_hands: int = 1
    running_mode: str = "video"       # "video" (трекинг между кадрами) | "image"
    # Пониженные presence/tracking держат руку в трекинге увереннее (меньше
    # «мигания» детекции то есть/то нет).
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    delegate: str = "CPU"             # "CPU" | "GPU"


@dataclass
class FilterConfig:
    """Параметры стабилизации курсора — ключевой объект для сравнения в работе.

    type:
        "none"     — без фильтрации (сырые координаты, базовая линия);
        "ema"      — экспоненциальное сглаживание (исходная реализация проекта);
        "one_euro" — One-Euro фильтр (Casiez et al., CHI 2012) — рекомендуется;
        "kalman"   — фильтр Калмана с моделью постоянной скорости.
    """

    type: str = "one_euro"
    # EMA
    ema_alpha: float = 0.15
    # One-Euro
    one_euro_min_cutoff: float = 1.0
    # beta=1.0 подобран эмпирически (filter_benchmark) для нормализованных
    # координат [0..1]: даёт меньший jitter, чем EMA, и при этом меньшую задержку.
    one_euro_beta: float = 1.0
    one_euro_d_cutoff: float = 1.0
    # Kalman
    kalman_process_noise: float = 1e-3
    kalman_measurement_noise: float = 1e-1


@dataclass
class CursorConfig:
    sensitivity: float = 1.0          # множитель амплитуды движения руки
    edge_margin: int = 2              # отступ от краёв экрана, px
    invert_x: bool = False
    invert_y: bool = False
    # Зона активного движения руки в кадре (нормализованная). Меньше значение →
    # легче дотянуться до краёв экрана меньшим движением руки.
    active_region: float = 0.55
    # Сглаживание высокочастотного воркера мыши: доля пути к цели за шаг.
    # Больше → отзывчивее (меньше лага), меньше → плавнее. 0.4 — баланс.
    worker_easing: float = 0.4
    # Dwell-click — клик по наведению (ассистивный режим, без щипка).
    dwell_enabled: bool = False
    dwell_profile: str = "custom"     # "fast" | "normal" | "steady" | "custom"
    dwell_time: float = 1.0           # сек удержания для срабатывания
    dwell_radius: int = 35            # допустимый дрейф курсора, px
    dwell_cooldown: float = 0.8       # пауза после клика, чтобы не было повторов


@dataclass
class GestureConfig:
    """Распознавание жестов.

    recognizer:
        "heuristic" — пороговые правила по геометрии руки (быстро, без обучения);
        "ml"        — обучаемый классификатор по признакам лендмарков.
    mapping связывает имя жеста с именем действия (см. cursor/actions).
    """

    recognizer: str = "heuristic"
    pinch_trigger_ratio: float = 0.22
    pinch_release_ratio: float = 0.30
    double_click_interval: float = 0.4
    scroll_threshold: float = 0.015
    scroll_speed: float = 3.0
    # Ассистивный режим одного жеста: курсор + dwell-click без щипков/голоса/скролла.
    dwell_only_mode: bool = False
    # Динамические жесты (свайпы открытой ладонью).
    dynamic_enabled: bool = True
    swipe_backend: str = "heuristic"  # "heuristic" | "template" | "lstm" | "tcn"
    swipe_model_path: str = DEFAULT_TEMPORAL_SWIPE_MODEL_PATH
    swipe_min_confidence: float = 0.65
    swipe_sequence_length: int = 16
    swipe_min_dist: float = 0.18      # мин. смещение (норм. координаты)
    swipe_max_time: float = 0.5       # макс. длительность взмаха, с
    swipe_cooldown: float = 0.8
    # Двуручный жест: pinch-to-zoom (обе руки щипком, разводим/сводим).
    bimanual_enabled: bool = True
    zoom_step_dist: float = 0.04      # изменение расстояния между руками на 1 «щелчок» зума
    ml_model_path: str = DEFAULT_ML_MODEL_PATH
    ml_dataset_path: str = DEFAULT_ML_DATASET_PATH
    ml_min_confidence: float = 0.6
    # Темпоральная стабилизация позы: взвешенное по уверенности голосование за
    # последние N кадров — гасит одиночные ошибки распознавания (дрожание позы).
    pose_smoothing_window: int = 3
    mapping: Dict[str, str] = field(default_factory=lambda: {
        "pinch_index": "left_click",
        "pinch_middle": "right_click",
        "pinch_ring": "backspace",
        "pinch_pinky": "enter",
        "peace": "scroll",
        "open_palm": "freeze",
        "fist": "voice",
    })


@dataclass
class VoiceConfig:
    enabled: bool = True
    language: str = "ru-RU"
    engine: str = "google"            # "google" | "vosk" (офлайн, если установлен)
    vosk_model_path: str = DEFAULT_VOSK_MODEL_PATH
    listen_timeout: float = 8.0
    phrase_time_limit: float = 15.0
    ambient_duration: float = 0.3


@dataclass
class InputConfig:
    """Runtime input safety options.

    dry_run: камера, трекинг и распознавание работают, но реальные события
        мыши/клавиатуры не отправляются в ОС. Это безопасный режим тренировки,
        особенно важный для ассистивного сценария и первого запуска.
    """

    dry_run: bool = False


@dataclass
class GazeConfig:
    """Параметры оценщика взгляда (MediaPipe Face Landmarker, iris).

    Оценщик опционален и по умолчанию выключен (см. FusionConfig.gaze_enabled).
    Калибровка хранится как аффинное отображение сырого вектора глаз → экран:
    cal_ax/cal_bx задают x = ax*raw_x + bx, аналогично по y. Значения по
    умолчанию близки к тождественному преобразованию, поэтому без калибровки
    оценка работает, хоть и грубо."""

    model_path: str = DEFAULT_FACE_MODEL_PATH
    delegate: str = "CPU"             # "CPU" | "GPU"
    running_mode: str = "video"       # как у hand_tracker — трекинг между кадрами
    min_face_confidence: float = 0.4
    # EMA-сглаживание сырой оценки взгляда (взгляд шумнее руки → сильнее гасим).
    smoothing_alpha: float = 0.35
    # Аффинная калибровка raw → экран [0..1] (по умолчанию ≈ тождество).
    cal_ax: float = 1.0
    cal_bx: float = 0.0
    cal_ay: float = 1.0
    cal_by: float = 0.0


@dataclass
class FusionConfig:
    """Слияние модальностей.

    Стратегия разрешения конфликтов между жестами и голосом и приоритеты команд.
    """

    enabled: bool = True
    # Жест активирует прослушивание голоса; пока слушаем — курсор не дёргаем.
    suppress_cursor_while_listening: bool = True
    # Окно (сек), в течение которого голос может уточнить жест и наоборот.
    fusion_window: float = 1.5
    voice_priority: float = 0.6       # вес голоса при конфликте [0..1]
    # --- Взгляд (gaze) как дополнительная модальность наведения ---
    # По умолчанию ВЫКЛ: фича ассистивная и опциональная, без модели не работает.
    gaze_enabled: bool = False
    # "assist" — взгляд лишь грубо подсказывает зону, рука всегда уточняет/перебивает;
    # "cursor" — взгляд ведёт курсор сам, когда руки нет в кадре.
    gaze_mode: str = "assist"
    gaze_min_confidence: float = 0.5  # ниже — оценка взгляда игнорируется
    gaze_weight: float = 0.5          # вес взгляда в смеси с рукой [0..1] (assist)
    gaze_max_age: float = 0.3         # сек: устаревшая оценка взгляда не применяется
    gaze: GazeConfig = field(default_factory=GazeConfig)


@dataclass
class ScanKeyboardConfig:
    """Параметры экранной сканирующей клавиатуры (row–column scanning).

    Ассистивный ввод текста одним действием «выбор» для пользователей, которым
    недоступны щипки и точное наведение. Подсветка сама перебирает строки, затем
    клавиши; пользователь фиксирует выбор удержанием/клавишей.

    scan_interval — пауза между сдвигами подсветки (с). Больше → проще успеть
        «выбрать», но медленнее набор; подбирается под пользователя.
    max_loops     — сколько полных проходов подсветки сделать без выбора, прежде
        чем остановить сканирование (0 = не останавливаться).
    select_key    — клавиша-переключатель «выбор» в Tk-представлении (работает
        везде и удобна для ручной проверки без жестов).
    """

    scan_interval: float = 1.2
    max_loops: int = 0
    select_key: str = "space"


@dataclass
class UIConfig:
    window_width: int = 800
    window_height: int = 600
    window_x: int = 100
    window_y: int = 100
    always_on_top: bool = True
    frameless: bool = True
    show_trail: bool = True
    show_particles: bool = True
    show_hud: bool = True
    show_landmarks: bool = True
    show_metrics: bool = True
    show_controls: bool = True
    theme: str = "dark"


@dataclass
class EvaluationConfig:
    enabled: bool = False
    # Параметры теста Фиттса (ISO 9241-9, multidirectional tapping).
    num_targets: int = 13             # точек по кругу (нечётное по стандарту)
    target_widths: list = field(default_factory=lambda: [40, 70, 110])
    ring_amplitudes: list = field(default_factory=lambda: [200, 350, 500])
    repetitions: int = 1
    log_dir: str = DEFAULT_LOG_DIR
    participant_id: str = "P01"


@dataclass
class PerformanceConfig:
    """Параметры под слабые устройства.

    detect_downscale: во сколько уменьшать кадр ПЕРЕД детекцией. Лендмарки
        нормализованы [0..1], поэтому уменьшение НЕ ломает координаты, но сильно
        ускоряет MediaPipe на слабом CPU (0.5 ≈ вдвое меньше пикселей).
    detect_max_fps: ограничение частоты детекции (0 = без лимита). Экономит CPU —
        курсор остаётся плавным за счёт высокочастотного воркера мыши.
    """

    detect_downscale: float = 1.0
    detect_max_fps: int = 0


@dataclass
class TelemetryConfig:
    enabled: bool = True
    log_dir: str = DEFAULT_LOG_DIR
    sample_interval: float = 1.0      # сек между записями метрик
    log_to_csv: bool = True


@dataclass
class AppConfig:
    """Корневой объект конфигурации."""

    profile_name: str = "default"
    assistive_preset: str = "balanced"
    start_mode: str = "view"          # "view" | "control"
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    input: InputConfig = field(default_factory=InputConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    scan_keyboard: ScanKeyboardConfig = field(default_factory=ScanKeyboardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # ---- (де)сериализация --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH) -> "AppConfig":
        """Загружает конфиг из JSON, дополняя недостающие поля значениями по
        умолчанию (forward-compatible: новый параметр в коде не ломает старый файл)."""
        if not os.path.exists(path):
            cfg = cls()
            cfg.save(path)
            return cfg
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _repair_runtime_paths(_validate(_from_dict(cls, data)))


def _from_dict(cls, data: Dict[str, Any]):
    """Рекурсивно собирает dataclass из словаря, игнорируя лишние ключи и
    подставляя дефолты для отсутствующих (устойчиво к эволюции схемы)."""
    if not is_dataclass(cls):
        return data
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


# Допустимые значения для строковых полей-перечислений. Конфиг — это
# персистентный JSON, переживающий смену версий приложения и ручную правку,
# поэтому «чужое» значение (опечатка, старый/будущий вариант, мусор) не должно
# ни уронить приложение, ни тихо включить сломанное поведение. При недопустимом
# значении возвращаемся к безопасному дефолту с печатью заметки в stderr.
_ENUM_FIELDS = {
    "tracking.running_mode": ("video", "image"),
    "filter.type": ("none", "ema", "one_euro", "kalman"),
    "cursor.dwell_profile": ("fast", "normal", "steady", "custom"),
    "gestures.recognizer": ("heuristic", "ml"),
    "gestures.swipe_backend": ("heuristic", "template", "lstm", "tcn"),
    "voice.engine": ("google", "vosk"),
    "fusion.gaze_mode": ("assist", "cursor"),
    "gaze.running_mode": ("video", "image"),
    "start_mode": ("view", "control"),
}


def _validate(cfg: AppConfig) -> AppConfig:
    """Нормализует строковые поля-перечисления, возвращая недопустимые значения
    к безопасному дефолту.

    Вызывается из AppConfig.load после сборки dataclass-а. Корректный конфиг
    проходит проверку без изменений (поведение байт-в-байт идентично). Каждое
    исправление сопровождается заметкой в stderr, чтобы порча файла не оставалась
    незамеченной."""
    defaults = AppConfig()
    _clamp_enum(cfg.tracking, "running_mode", "tracking.running_mode", defaults.tracking)
    _clamp_enum(cfg.filter, "type", "filter.type", defaults.filter)
    _clamp_enum(cfg.cursor, "dwell_profile", "cursor.dwell_profile", defaults.cursor)
    _clamp_enum(cfg.gestures, "recognizer", "gestures.recognizer", defaults.gestures)
    _clamp_enum(cfg.gestures, "swipe_backend", "gestures.swipe_backend", defaults.gestures)
    _clamp_enum(cfg.voice, "engine", "voice.engine", defaults.voice)
    _clamp_enum(cfg.fusion, "gaze_mode", "fusion.gaze_mode", defaults.fusion)
    _clamp_enum(cfg.fusion.gaze, "running_mode", "gaze.running_mode", defaults.fusion.gaze)
    _clamp_enum(cfg, "start_mode", "start_mode", defaults)
    return cfg


def _clamp_enum(section: Any, attr: str, key: str, default_section: Any) -> None:
    """Если section.attr не входит в допустимый набор — заменить на дефолт.

    section          — dataclass-секция (или сам AppConfig) с проверяемым полем;
    attr             — имя атрибута внутри секции;
    key              — ключ в _ENUM_FIELDS (для набора допустимых значений);
    default_section  — соответствующая секция эталонного AppConfig (источник
                       безопасного значения по умолчанию)."""
    allowed = _ENUM_FIELDS[key]
    value = getattr(section, attr)
    if value in allowed:
        return
    fallback = getattr(default_section, attr)
    print(
        f"[config] недопустимое значение {key}={value!r}; "
        f"возвращено к безопасному значению {fallback!r}",
        file=sys.stderr,
    )
    setattr(section, attr, fallback)


def _repair_runtime_paths(cfg: AppConfig) -> AppConfig:
    """Fix paths that become stale after moving the project or unpacking a bundle."""
    cfg.tracking.model_path = _repair_file_path(cfg.tracking.model_path, DEFAULT_MODEL_PATH)
    cfg.fusion.gaze.model_path = _repair_file_path(cfg.fusion.gaze.model_path, DEFAULT_FACE_MODEL_PATH)
    cfg.gestures.ml_model_path = _repair_file_path(cfg.gestures.ml_model_path, DEFAULT_ML_MODEL_PATH)
    cfg.gestures.ml_dataset_path = _repair_file_path(cfg.gestures.ml_dataset_path, DEFAULT_ML_DATASET_PATH)
    cfg.gestures.swipe_model_path = _repair_file_path(
        cfg.gestures.swipe_model_path, DEFAULT_TEMPORAL_SWIPE_MODEL_PATH
    )
    cfg.evaluation.log_dir = _repair_dir_path(cfg.evaluation.log_dir, DEFAULT_LOG_DIR)
    cfg.telemetry.log_dir = _repair_dir_path(cfg.telemetry.log_dir, DEFAULT_LOG_DIR)
    return cfg


def _repair_file_path(path: str, default: str) -> str:
    if os.path.exists(path):
        return path
    if os.path.exists(default) or _looks_like_stale_aircontrol_path(path):
        return default
    return path


def _repair_dir_path(path: str, default: str) -> str:
    if _looks_like_stale_aircontrol_path(path):
        return default
    if os.path.isdir(path):
        return path
    if os.path.isdir(default):
        return default
    return path


def _looks_like_stale_aircontrol_path(path: str) -> bool:
    """Detect absolute paths from an older checkout/bundle without touching custom dirs."""
    if not path:
        return False
    try:
        p = os.path.abspath(os.path.expanduser(path))
        current_project = os.path.abspath(_PROJECT_DIR)
        current_data = os.path.abspath(DATA_DIR)
        if p.startswith(current_project + os.sep) or p == current_project:
            return False
        if p.startswith(current_data + os.sep) or p == current_data:
            return False
        parts = set(p.split(os.sep))
        if "Hand Mouse Controller" in p and "aircontrol" in parts:
            return True
        return f"{os.sep}aircontrol{os.sep}data{os.sep}" in p or p.endswith(
            f"{os.sep}hand_landmarker.task"
        )
    except Exception:
        return False


DWELL_PROFILES = {
    "fast": {"time": 0.75, "radius": 32, "cooldown": 0.55, "label": "fast"},
    "normal": {"time": 1.15, "radius": 48, "cooldown": 0.85, "label": "normal"},
    "steady": {"time": 1.70, "radius": 68, "cooldown": 1.20, "label": "steady"},
}
DWELL_PROFILE_ORDER = ("fast", "normal", "steady")

ASSISTIVE_PRESETS = {
    "balanced": {
        "label": "balanced",
        "dwell_profile": "normal",
        "active_region": 0.45,
        "sensitivity": 1.05,
        "worker_easing": 0.30,
        "pose_window": 5,
        "scroll_speed": 2.2,
        "detect_downscale": 0.5,
        "detect_max_fps": 24,
        "filter_min_cutoff": 0.8,
        "filter_beta": 0.8,
    },
    "steady": {
        "label": "steady",
        "dwell_profile": "steady",
        "active_region": 0.52,
        "sensitivity": 0.90,
        "worker_easing": 0.22,
        "pose_window": 7,
        "scroll_speed": 1.6,
        "detect_downscale": 0.5,
        "detect_max_fps": 20,
        "filter_min_cutoff": 0.65,
        "filter_beta": 0.55,
    },
    "low_motion": {
        "label": "low motion",
        "dwell_profile": "normal",
        "active_region": 0.34,
        "sensitivity": 1.20,
        "worker_easing": 0.28,
        "pose_window": 5,
        "scroll_speed": 1.9,
        "detect_downscale": 0.5,
        "detect_max_fps": 24,
        "filter_min_cutoff": 0.75,
        "filter_beta": 0.7,
    },
}
ASSISTIVE_PRESET_ORDER = ("balanced", "steady", "low_motion")


def apply_dwell_profile(cfg: AppConfig, profile: str) -> AppConfig:
    """Apply one named dwell-click profile and enable dwell-click."""
    if profile not in DWELL_PROFILES:
        profile = "normal"
    values = DWELL_PROFILES[profile]
    cfg.cursor.dwell_enabled = True
    cfg.cursor.dwell_profile = profile
    cfg.cursor.dwell_time = float(values["time"])
    cfg.cursor.dwell_radius = int(values["radius"])
    cfg.cursor.dwell_cooldown = float(values["cooldown"])
    return cfg


def next_dwell_profile(current: str) -> str:
    """Return the next named dwell profile for UI cycling."""
    if current not in DWELL_PROFILE_ORDER:
        return DWELL_PROFILE_ORDER[0]
    idx = DWELL_PROFILE_ORDER.index(current)
    return DWELL_PROFILE_ORDER[(idx + 1) % len(DWELL_PROFILE_ORDER)]


def apply_assistive_profile(cfg: AppConfig, preset: str = "balanced") -> AppConfig:
    """Tune defaults for low-effort, safer hands-free control."""
    if preset not in ASSISTIVE_PRESETS:
        preset = "balanced"
    values = ASSISTIVE_PRESETS[preset]

    cfg.profile_name = "assistive"
    cfg.assistive_preset = preset
    cfg.start_mode = "control"

    # Reliable baseline for low-power laptops and integrated GPUs.
    cfg.camera.width = 480
    cfg.camera.height = 360
    cfg.camera.target_fps = 30

    # Less physical travel, smoother motion, and click by dwell instead of pinch.
    cfg.cursor.active_region = float(values["active_region"])
    cfg.cursor.sensitivity = float(values["sensitivity"])
    cfg.cursor.worker_easing = float(values["worker_easing"])
    apply_dwell_profile(cfg, str(values["dwell_profile"]))

    # Reduce accidental high-energy gestures and CPU load.
    cfg.gestures.dwell_only_mode = True
    cfg.gestures.dynamic_enabled = False
    cfg.gestures.bimanual_enabled = False
    cfg.gestures.pose_smoothing_window = max(
        cfg.gestures.pose_smoothing_window, int(values["pose_window"])
    )
    cfg.gestures.scroll_speed = min(cfg.gestures.scroll_speed, float(values["scroll_speed"]))

    cfg.filter.type = "one_euro"
    cfg.filter.one_euro_min_cutoff = float(values["filter_min_cutoff"])
    cfg.filter.one_euro_beta = float(values["filter_beta"])

    cfg.performance.detect_downscale = float(values["detect_downscale"])
    cfg.performance.detect_max_fps = int(values["detect_max_fps"])

    cfg.ui.show_hud = True
    cfg.ui.show_metrics = True
    cfg.ui.show_particles = False
    cfg.ui.show_trail = True
    cfg.voice.enabled = True
    return cfg
