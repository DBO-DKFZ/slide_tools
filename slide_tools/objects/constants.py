from enum import Enum

__all__ = ["SizeUnit", "SlideType", "LabelInterpolation", "LabelField", "BalanceMode"]


class SizeUnit(Enum):
    PIXEL = "pixel"
    MICRON = "micron"


class SlideType(Enum):
    APERIO = "aperio"


class LabelInterpolation(Enum):
    LINEAR = "linear"
    NEAREST = "nearest"


class LabelField(Enum):
    POINTS = "points"
    LABELS = "labels"


class BalanceMode(Enum):
    MIN = "min"
    MEAN = "mean"
    MAX = "max"
    MEDIAN = "median"
