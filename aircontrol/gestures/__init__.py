"""Подсистема распознавания жестов."""

from .engine import FrameGestures, GestureEngine, GestureEvent
from .heuristic import HeuristicPoseClassifier
from .ml import GestureDataset, MLPoseClassifier, train_from_dataset

__all__ = [
    "GestureEngine", "FrameGestures", "GestureEvent",
    "HeuristicPoseClassifier", "MLPoseClassifier",
    "GestureDataset", "train_from_dataset",
]
