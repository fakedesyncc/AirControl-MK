"""Точка входа AirControl.

Подкоманды:
    run      — запустить приложение (по умолчанию).
    collect  — собрать датасет жестов для обучения ML.
    train    — обучить ML-распознаватель и вывести метрики.
    fitts    — запустить тест Фиттса (ISO 9241-9) для оценки ввода.

Примеры:
    python -m aircontrol
    python -m aircontrol collect
    python -m aircontrol train --backend rf
    python -m aircontrol fitts --method gesture --participant P01
"""

import argparse
import json
import os
import sys

from .config import ASSISTIVE_PRESET_ORDER, AppConfig, apply_assistive_profile


PRESETS = {
    # (camera_w, camera_h, detect_downscale, num_hands)
    "low":    (480, 360, 0.6, 1),
    "medium": (640, 480, 1.0, 1),
    "high":   (1280, 720, 1.0, 2),
}


def _configure_stdio() -> None:
    """Keep CLI diagnostics printable on Windows legacy code pages."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def cmd_run(args):
    from .app import AirControlApp
    cfg = AppConfig.load()
    if getattr(args, "assistive", False):
        apply_assistive_profile(
            cfg,
            getattr(args, "assistive_preset", None) or "balanced",
        )
    if getattr(args, "preset", None):
        w, h, ds, nh = PRESETS[args.preset]
        cfg.camera.width, cfg.camera.height = w, h
        cfg.performance.detect_downscale = ds
        cfg.tracking.num_hands = nh
        print(f"[run] Пресет '{args.preset}': {w}x{h}, downscale={ds}, hands={nh}")
    if args.recognizer:
        cfg.gestures.recognizer = args.recognizer
    if getattr(args, "swipe_backend", None):
        cfg.gestures.swipe_backend = args.swipe_backend
    if args.filter:
        cfg.filter.type = args.filter
    if getattr(args, "voice_engine", None):
        cfg.voice.engine = args.voice_engine
    if getattr(args, "gaze_mode", None):
        cfg.fusion.gaze_enabled = args.gaze_mode != "off"
        if cfg.fusion.gaze_enabled:
            cfg.fusion.gaze_mode = args.gaze_mode
    if args.mode:
        cfg.start_mode = args.mode
    if getattr(args, "dry_input", False):
        cfg.input.dry_run = True
    try:
        AirControlApp(cfg).run()
    except RuntimeError as exc:
        # Понятное сообщение вместо сырого трейсбека (камера/модель недоступны и т.п.).
        print(f"\n[run] AirControl не запустился:\n{exc}\n", file=sys.stderr)
        print("Проверьте систему: python -m aircontrol doctor", file=sys.stderr)
        raise SystemExit(1)


def cmd_assistive(args):
    args.assistive = True
    args.assistive_preset = getattr(args, "assistive_preset", None) or "balanced"
    args.preset = getattr(args, "preset", None)
    args.recognizer = getattr(args, "recognizer", None)
    args.swipe_backend = getattr(args, "swipe_backend", None)
    args.filter = getattr(args, "filter", None)
    args.voice_engine = getattr(args, "voice_engine", None)
    args.gaze_mode = getattr(args, "gaze_mode", None)
    args.mode = getattr(args, "mode", None) or "control"
    cmd_run(args)


def cmd_launcher(args):
    from .launcher import run_launcher
    run_launcher()


def cmd_keyboard(args):
    from .ui.scanning_keyboard import run_scanning_keyboard
    cfg = AppConfig.load()
    if getattr(args, "scan_interval", None):
        cfg.scan_keyboard.scan_interval = args.scan_interval
    if getattr(args, "dry_input", False):
        cfg.input.dry_run = True
    run_scanning_keyboard(cfg)


def cmd_collect(args):
    from .gestures.collector import collect_dataset
    collect_dataset(AppConfig.load())


def cmd_train(args):
    from .gestures.ml import train_from_dataset
    cfg = AppConfig.load()
    try:
        metrics = train_from_dataset(cfg.gestures.ml_dataset_path,
                                     cfg.gestures.ml_model_path, backend=args.backend)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Нечего обучать: {exc}")
        print("Сначала соберите датасет: python -m aircontrol collect")
        return
    print("=== Результаты обучения ===")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"},
                     ensure_ascii=False, indent=2))
    if metrics.get("confusion_matrix"):
        print("Confusion matrix (labels:", metrics.get("labels"), "):")
        for row in metrics["confusion_matrix"]:
            print("  ", row)


def cmd_fitts(args):
    from .evaluation.fitts_runner import run_fitts_test
    cfg = AppConfig.load()
    if args.participant:
        cfg.evaluation.participant_id = args.participant
    run_fitts_test(cfg.evaluation, args.method)


def cmd_bench(args):
    from .evaluation.filter_benchmark import main as bench_main
    bench_main()


def cmd_selftest(args):
    """Самопроверка: грузит модель руки и прогоняет детекцию на пустом кадре.
    Полезно для проверки собранного бандла (модель + mediapipe на месте)."""
    import numpy as np
    from .ui.calibration import compute_active_region, compute_pinch_thresholds
    from .tracking.hand_tracker import HandTracker
    from .config import DEFAULT_MODEL_PATH
    print(f"Модель: {DEFAULT_MODEL_PATH}")
    print(f"Существует: {os.path.exists(DEFAULT_MODEL_PATH)}")
    try:
        assert compute_active_region([(0.3 + i * 0.01, 0.4) for i in range(12)]) is not None
        assert compute_pinch_thresholds([0.8] * 8, [0.25] * 8) is not None
        tr = HandTracker(AppConfig.load().tracking)
        n = len(tr.detect(np.zeros((480, 640, 3), dtype=np.uint8)))
        tr.close()
        print("✓ Модуль калибровки доступен")
        print(f"✓ HandTracker инициализирован, детекция работает (рук на пустом кадре: {n})")
        print("✓ Самопроверка пройдена — приложение готово к работе")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"✗ Самопроверка провалена: {exc}")
        raise SystemExit(1)


def cmd_mictest(args):
    """Диагностика голосового ввода: список микрофонов + одно распознавание."""
    try:
        import speech_recognition as sr
    except ImportError:
        print("SpeechRecognition не установлен"); return
    try:
        import pyaudio  # noqa: F401
    except Exception as exc:
        print(f"PyAudio недоступен: {exc}")
        print("Голосовые команды отключены. Для микрофона установите optional-зависимости.")
        return
    print("Доступные микрофоны:")
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{i}] {name}")
    r = sr.Recognizer()
    cfg = AppConfig.load()
    print(f"\nГоворите что-нибудь ({cfg.voice.language}) после 'Слушаю...' (нужен интернет)")
    try:
        with sr.Microphone() as src:
            r.adjust_for_ambient_noise(src, duration=0.5)
            print("Слушаю...")
            audio = r.listen(src, timeout=6, phrase_time_limit=6)
        text = r.recognize_google(audio, language=cfg.voice.language)
        print(f"✓ Распознано: {text!r}")
    except sr.WaitTimeoutError:
        print("✗ Тишина — микрофон не слышит (проверьте разрешение на микрофон)")
    except sr.UnknownValueError:
        print("✗ Речь не распознана (попробуйте громче/чётче)")
    except sr.RequestError as e:
        print(f"✗ Нет связи с сервисом распознавания (нужен интернет): {e}")
    except Exception as e:
        print(f"✗ Ошибка микрофона: {e} (дайте разрешение на микрофон в Системных настройках)")


def cmd_doctor(args):
    from .diagnostics import build_report
    print(build_report(scan_camera=not args.no_camera,
                       camera_limit=args.camera_limit,
                       input_probe=args.input_probe))


def cmd_support(args):
    from .diagnostics import save_support_bundle
    path = save_support_bundle(args.output,
                               scan_camera=not args.no_camera,
                               input_probe=args.input_probe)
    print(f"Support bundle saved: {path}")


def cmd_report(args):
    from .evaluation.analysis import generate_report
    generate_report(AppConfig.load())


def cmd_usability(args):
    from .evaluation.usability import (
        append_usability_result,
        parse_sus_csv,
        parse_tlx_pairs,
    )
    cfg = AppConfig.load()
    output = args.output or os.path.join(cfg.evaluation.log_dir, "usability.csv")
    weights = parse_tlx_pairs(args.tlx_weights) if args.tlx_weights else None
    result = append_usability_result(
        output,
        participant_id=args.participant,
        condition=args.condition,
        sus_responses=parse_sus_csv(args.sus),
        tlx_ratings=parse_tlx_pairs(args.tlx),
        tlx_weights=weights,
    )
    print(f"Usability saved: {output}")
    print(f"SUS: {result.sus_score:.2f}/100")
    print(f"NASA-TLX raw: {result.nasa_tlx_raw:.2f}/100")
    if result.nasa_tlx_weighted is not None:
        print(f"NASA-TLX weighted: {result.nasa_tlx_weighted:.2f}/100")


def cmd_calibrate(args):
    from .ui.calibration import run_calibration
    run_calibration(AppConfig.load())


def cmd_synth(args):
    from .gestures.synthetic import generate_synthetic_dataset
    from .gestures.ml import GestureDataset
    cfg = AppConfig.load()
    ds = generate_synthetic_dataset(per_pose=args.per_pose, seed=args.seed)
    if args.merge:
        existing = GestureDataset.load(cfg.gestures.ml_dataset_path)
        existing.X.extend(ds.X); existing.y.extend(ds.y)
        ds = existing
    ds.save(cfg.gestures.ml_dataset_path)
    print(f"Синтетический датасет сохранён: {len(ds)} примеров {ds.counts()}")
    print(f"→ {cfg.gestures.ml_dataset_path}")
    print("Обучить модель: python -m aircontrol train --backend rf")


def main(argv=None):
    _configure_stdio()
    parser = argparse.ArgumentParser(prog="aircontrol",
                                     description="AirControl — мультимодальное бесконтактное управление ПК")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="запустить приложение")
    p_run.add_argument("--recognizer", choices=["heuristic", "ml"])
    p_run.add_argument("--swipe-backend", choices=["heuristic", "template", "lstm", "tcn"],
                       help="backend динамических свайпов")
    p_run.add_argument("--filter", choices=["none", "ema", "one_euro", "kalman"])
    p_run.add_argument("--voice-engine", choices=["google", "vosk"],
                       help="google или приватный офлайн Vosk")
    p_run.add_argument("--gaze-mode", choices=["off", "assist", "cursor"],
                       help="экспериментальное слияние взгляда: off, assist, cursor")
    p_run.add_argument("--mode", choices=["view", "control"])
    p_run.add_argument("--preset", choices=["low", "medium", "high"],
                       help="low — для слабых ПК, high — для мощных (две руки)")
    p_run.add_argument("--assistive", action="store_true",
                       help="ассистивный профиль: dwell-click, мягкий курсор, меньше случайных жестов")
    p_run.add_argument("--assistive-preset", choices=ASSISTIVE_PRESET_ORDER, default="balanced",
                       help="balanced, steady для тремора, low_motion для малого движения руки")
    p_run.add_argument("--dry-input", action="store_true",
                       help="безопасная тренировка: не отправлять события мыши/клавиатуры в ОС")
    p_run.set_defaults(func=cmd_run)

    p_assistive = sub.add_parser("assistive", help="запустить ассистивный режим")
    p_assistive.add_argument("--dry-input", action="store_true",
                             help="сначала тренироваться без реальных кликов/клавиш")
    p_assistive.add_argument("--preset", choices=["low", "medium", "high"], default="low")
    p_assistive.add_argument("--assistive-preset", choices=ASSISTIVE_PRESET_ORDER,
                             default="balanced",
                             help="balanced, steady для тремора, low_motion для малого движения руки")
    p_assistive.add_argument("--recognizer", choices=["heuristic", "ml"], default=None)
    p_assistive.add_argument("--swipe-backend", choices=["heuristic", "template", "lstm", "tcn"],
                             default=None)
    p_assistive.add_argument("--filter", choices=["none", "ema", "one_euro", "kalman"], default=None)
    p_assistive.add_argument("--voice-engine", choices=["google", "vosk"], default=None)
    p_assistive.add_argument("--gaze-mode", choices=["off", "assist", "cursor"], default=None)
    p_assistive.add_argument("--mode", choices=["view", "control"], default="control")
    p_assistive.set_defaults(func=cmd_assistive)

    sub.add_parser("launcher", help="показать стартовое окно без консоли").set_defaults(func=cmd_launcher)

    p_keyboard = sub.add_parser("keyboard",
                                help="экранная сканирующая клавиатура (ввод одним «выбором»)")
    p_keyboard.add_argument("--scan-interval", type=float, default=None, dest="scan_interval",
                            help="пауза между сдвигами подсветки, сек (больше — проще успеть выбрать)")
    p_keyboard.add_argument("--dry-input", action="store_true",
                            help="безопасная проверка: не отправлять клавиши в ОС")
    p_keyboard.set_defaults(func=cmd_keyboard)

    sub.add_parser("collect", help="собрать датасет жестов").set_defaults(func=cmd_collect)

    sub.add_parser("calibrate", help="мастер персональной калибровки").set_defaults(func=cmd_calibrate)

    p_train = sub.add_parser("train", help="обучить ML-распознаватель")
    p_train.add_argument("--backend", default="knn", choices=["knn", "mlp", "rf"])
    p_train.set_defaults(func=cmd_train)

    p_fitts = sub.add_parser("fitts", help="тест Фиттса (ISO 9241-9)")
    p_fitts.add_argument("--method", default="gesture")
    p_fitts.add_argument("--participant", default=None)
    p_fitts.set_defaults(func=cmd_fitts)

    sub.add_parser("bench", help="бенчмарк фильтров стабилизации").set_defaults(func=cmd_bench)

    sub.add_parser("report", help="сгенерировать исследовательские графики").set_defaults(func=cmd_report)

    p_usability = sub.add_parser("usability", help="записать SUS и NASA-TLX для исследования")
    p_usability.add_argument("--participant", default="P01")
    p_usability.add_argument("--condition", default="assistive")
    p_usability.add_argument("--sus", required=True,
                             help="10 ответов SUS через запятую, значения 1..5")
    p_usability.add_argument("--tlx", required=True,
                             help="mental=40,physical=50,temporal=30,performance=20,effort=60,frustration=10")
    p_usability.add_argument("--tlx-weights", default=None,
                             help="опциональные веса NASA-TLX в том же формате")
    p_usability.add_argument("-o", "--output", default=None)
    p_usability.set_defaults(func=cmd_usability)

    sub.add_parser("mictest", help="диагностика голосового ввода/микрофона").set_defaults(func=cmd_mictest)

    sub.add_parser("selftest", help="самопроверка (модель + детекция)").set_defaults(func=cmd_selftest)

    p_doctor = sub.add_parser("doctor", help="диагностика окружения для тестеров")
    p_doctor.add_argument("--no-camera", action="store_true",
                          help="не сканировать индексы камер")
    p_doctor.add_argument("--camera-limit", type=int, default=None,
                          help="сколько индексов камер проверить")
    p_doctor.add_argument("--input-probe", action="store_true",
                          help="без кликов проверить, может ли backend сдвинуть курсор на 1 px и вернуть назад")
    p_doctor.set_defaults(func=cmd_doctor)

    p_support = sub.add_parser("support", help="сохранить ZIP-отчёт поддержки")
    p_support.add_argument("-o", "--output", default=None,
                           help="путь к zip-файлу отчёта")
    p_support.add_argument("--no-camera", action="store_true",
                           help="не сканировать индексы камер")
    p_support.add_argument("--input-probe", action="store_true",
                           help="добавить безопасную проверку движения курсора без кликов")
    p_support.set_defaults(func=cmd_support)

    p_synth = sub.add_parser("synth", help="сгенерировать синтетический датасет жестов")
    p_synth.add_argument("--per-pose", type=int, default=300, dest="per_pose")
    p_synth.add_argument("--seed", type=int, default=42)
    p_synth.add_argument("--merge", action="store_true",
                         help="добавить к существующему датасету, а не перезаписать")
    p_synth.set_defaults(func=cmd_synth)

    args = parser.parse_args(argv)
    if not args.command:
        args.func = cmd_launcher
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
