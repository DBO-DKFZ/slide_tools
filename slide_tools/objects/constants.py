from enum import Enum

__all__ = ["SizeUnit", "SlideType", "LabelInterpolation", "LabelField", "BalanceMode"]


class SizeUnit(Enum):
    PIXEL = "pixel"
    MICRON = "micron"


class SlideType(Enum):
    APERIO = "aperio"
    TIFF = "tiff"


class LabelInterpolation(Enum):
    LINEAR = "linear"
    NEAREST = "nearest"


class LabelField(Enum):
    POINTS = "points"
    VALUES = "values"


class BalanceMode(Enum):
    MIN = "min"
    MEAN = "mean"
    MAX = "max"
    MEDIAN = "median"
