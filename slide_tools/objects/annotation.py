from dataclasses import dataclass

from shapely.geometry import GeometryCollection

__all__ = ["Annotation"]


@dataclass
class Annotation:
    """
    An object to collect attributes of an annotation.
    """

    geometry: GeometryCollection
    properties: dict
