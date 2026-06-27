"""Голосовая подсистема: распознавание речи и команды."""

from .commands import CommandProcessor
from .recognizer import VoiceRecognizer, SPEECH_AVAILABLE

__all__ = ["VoiceRecognizer", "CommandProcessor", "SPEECH_AVAILABLE"]
