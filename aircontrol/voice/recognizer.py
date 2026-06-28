"""Распознавание речи в фоновом потоке.

Триггерится жестом «кулак» (через координатор слияния). По завершении фразы
текст передаётся в CommandProcessor: если это команда — выполняется, иначе
печатается в активное поле. Поддерживает онлайн-движок Google и опциональный
офлайн Vosk (если установлен)."""

import importlib.util
import os
import threading
import time
from typing import Optional

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    sr = None

MICROPHONE_BACKEND_AVAILABLE = importlib.util.find_spec("pyaudio") is not None

from ..config import DEFAULT_VOSK_MODEL_PATH, VoiceConfig


class VoiceRecognizer:
    def __init__(self, cfg: VoiceConfig, action_executor, command_processor):
        self.cfg = cfg
        self.act = action_executor
        self.commands = command_processor
        self.is_listening = False
        self.last_text = ""
        self.last_status = _initial_status(cfg)
        self._recognizer = sr.Recognizer() if SPEECH_AVAILABLE else None
        self._vosk_model = None
        self._vosk_error = ""
        if SPEECH_AVAILABLE and cfg.engine == "vosk":
            self._try_load_vosk()
            if self._vosk_model is None and self._vosk_error:
                self.last_status = self._vosk_error

    @property
    def available(self) -> bool:
        if not (SPEECH_AVAILABLE and MICROPHONE_BACKEND_AVAILABLE and self.cfg.enabled):
            return False
        if self.cfg.engine == "vosk":
            return self._vosk_model is not None
        if self.cfg.engine == "google" and flac_converter_path() is None:
            return False
        return True

    def _try_load_vosk(self) -> None:
        model_dir = getattr(self.cfg, "vosk_model_path", "") or DEFAULT_VOSK_MODEL_PATH
        try:
            import vosk  # type: ignore
        except Exception as exc:
            self._vosk_model = None
            self._vosk_error = f"vosk package unavailable: {exc}"
            return
        if not os.path.isdir(model_dir):
            self._vosk_model = None
            self._vosk_error = f"vosk model unavailable: {model_dir}"
            return
        try:
            self._vosk_model = vosk.Model(model_dir)
            self._vosk_error = ""
        except Exception as exc:
            self._vosk_model = None
            self._vosk_error = f"vosk model load failed: {exc}"

    def start_listening(self) -> None:
        if not self.available or self.is_listening:
            return
        self.is_listening = True
        self.last_status = "listening"
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def _listen_worker(self) -> None:
        try:
            with sr.Microphone() as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=self.cfg.ambient_duration)
                audio = self._recognizer.listen(
                    source, timeout=self.cfg.listen_timeout,
                    phrase_time_limit=self.cfg.phrase_time_limit)
            text = self._recognize(audio)
            if text:
                self.last_text = text
                self.last_status = f"recognized: {text}"
                if not self.commands.process(text):
                    self.act.type_text(text + " ")
            else:
                self.last_status = "not recognized"
        except sr.WaitTimeoutError:
            self.last_status = "timeout"
        except Exception as exc:
            self.last_status = f"error: {exc}"
        finally:
            self.is_listening = False

    def _recognize(self, audio) -> Optional[str]:
        if self.cfg.engine == "vosk":
            if self._vosk_model is None:
                return None
            try:
                import json
                import vosk  # type: ignore
                rec = vosk.KaldiRecognizer(self._vosk_model, 16000)
                rec.AcceptWaveform(audio.get_raw_data(convert_rate=16000, convert_width=2))
                return json.loads(rec.FinalResult()).get("text", "") or None
            except Exception:
                return None
        try:
            return self._recognizer.recognize_google(audio, language=self.cfg.language)
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None


def _initial_status(cfg: VoiceConfig) -> str:
    if not cfg.enabled:
        return "disabled"
    if not SPEECH_AVAILABLE:
        return "speech_recognition unavailable"
    if not MICROPHONE_BACKEND_AVAILABLE:
        return "microphone backend unavailable"
    if cfg.engine == "vosk":
        if importlib.util.find_spec("vosk") is None:
            return "vosk package unavailable"
        model_dir = getattr(cfg, "vosk_model_path", "") or DEFAULT_VOSK_MODEL_PATH
        if not os.path.isdir(model_dir):
            return "vosk model unavailable"
    if cfg.engine == "google" and flac_converter_path() is None:
        return "flac converter unavailable"
    return "ready"


def flac_converter_path(getter=None) -> str | None:
    """Return a usable FLAC converter path for SpeechRecognition, if present."""
    if not SPEECH_AVAILABLE:
        return None
    getter = getter or sr.get_flac_converter
    try:
        path = getter()
    except Exception:
        return None
    if path and os.path.exists(path) and os.access(path, os.X_OK):
        return path
    return None
