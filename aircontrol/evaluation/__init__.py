"""Подсистема оценки: тест Фиттса и телеметрия производительности."""

from .fitts import ConditionResult, FittsTest
from .metrics import FPSMeter, TelemetryLogger

__all__ = ["FittsTest", "ConditionResult", "FPSMeter", "TelemetryLogger"]
