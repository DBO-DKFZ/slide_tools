import json
import random
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Union

import cucim
import numpy as np
import xmltodict
from numpy.typing import ArrayLike
from rasterio.features import rasterize as rio_rasterize
from scipy import interpolate as scipy_interpolate
from shapely import affinity as shapely_affinity
from shapely import geometry as shapely_geometry

from .annotation import Annotation
from .constants import LabelField, LabelInterpolation, SizeUnit, SlideType

__all__ = ["Slide"]


@dataclass
class Slide:
    """
    An object to collect attributes of a single slide, including annotation and label.

    Attributes:
        annotations (Annotation): Annotation object with geometry and properties
        image (CuImage): a single loaded WSI
        label_func (callable): function that maps coordinates to labels: (x, y) -> label
        properties (dict): any properties concerning the slide
        microns_per_pixel (float): native/highest WSI resolution
        native_sizes (sequence): native tile sizes for all levels (level=0 is highest resolution)
        regions (np.array): array of coordinates and size per region ((x, y, w, h), ...)
        labels (dict): Dictionary of labels for all regions e.g. {"tumor": [0,0,1,...], "normal": [1,1,0,...]}
    """

    annotations: Optional[Sequence[Annotation]] = None
    image: Optional[cucim.CuImage] = None
    local_label_func: Optional[Callable] = None
    global_label_func: Optional[Callable] = None
    properties: Optional[dict] = None
    microns_per_pixel: Optional[float] = None
    native_sizes: Optional[Sequence[int]] = None
    regions: Optional[Sequence] = None
    labels: Optional[Sequence] = None

    def load_annotations_from_geojson(self, path: str, simplify_tolerance: int = 0):
        """
        Load GeoJSON saved by QuPath.

        Args:
            path (str): path
            simplify_tolerance (int): simplify annotation shape (see `shapely.geometry.simplify`) (default: 0)
        """
        with open(path) as stream:
            dictionary = json.load(stream)
            assert dictionary["type"] == "FeatureCollection"

        self.annotations = []

        for i in dictionary["features"]:
            geometry = shapely_geometry.shape(i["geometry"]).buffer(0)
            if simplify_tolerance > 0:
                geometry = geometry.simplify(simplify_tolerance)
            self.annotations.append(
                Annotation(geometry=geometry, properties=i["properties"])
            )
        if len(self.annotations) == 0:
            raise RuntimeError(f"No annotation found in: {path}")

    def load_wsi(self, path: str):
        """
        Load WSI with cuCIM and populate slide.image.

        Args:
            path (str): path

        Supports population of slide.microns_per_pixel for SlideType.{APERIO, TIFF}.
        """
        self.image = cucim.CuImage(path)
        self.image_shape = self.image.shape
        self.native_sizes = np.array(self.image.resolutions["level_tile_sizes"]) \
            * np.array(self.image.resolutions["level_downsamples"], dtype=int)[..., None]
        self.is_loaded = True

        if SlideType.APERIO.value in self.image.metadata:
            self.microns_per_pixel = float(self.image.metadata["aperio"]["MPP"])
        elif SlideType.TIFF.value in self.image.metadata:
            props = xmltodict.parse(self.image.raw_metadata)["OME"]["Image"]["Pixels"]
            is_mu = (props["@PhysicalSizeXUnit"] == "µm") and (
                props["@PhysicalSizeYUnit"] == "µm"
            )
            is_equal = props["@PhysicalSizeX"] == props["@PhysicalSizeY"]
            if is_mu and is_equal:
                self.microns_per_pixel = float(props["@PhysicalSizeX"])
            else:
                raise RuntimeError(
                    f"Could not extract microns_per_pixel from:\n{props}"
                )
        else:
            raise NotImplementedError(
                "Unknown WSI: Please add a way to extract microns_per_pixel for your WSI!"
            )

    def unload_wsi(self):
        if self.is_loaded:
            self.image = self.image.path
            self.is_loaded = False

    def set_global_label(self, labels: dict):
        """
        Set a global label for all regions and populate slide.label_func: coord=(x, y) -> label.

        Args:
            labels (dict): e.g. {"age": 42, "": 0, ...}
        """
        assert isinstance(labels, dict)
        labels = {k: np.asarray(labels[k]) for k in labels}

        def global_label_func(x, y):
            return {k: labels[k][None, ...].repeat(len(x), axis=0) for k in labels}

        self.global_label_func = global_label_func

    def load_label_from_json(
        self,
        path: str,
        interpolation: LabelInterpolation = LabelInterpolation.NEAREST,
        load_keys: Optional[Sequence[str]] = None,
        linear_fill_value: float = np.nan,
    ):
        """
        Load labels from json and populate slide.label_func: coord=(x, y) -> label.

        Args:
            path (str): path
            interpolation (LabelInterpolation): interpolation for unknown (x, y) coordinates
                                                supports LabelInterpolation.{NEAREST, LINEAR}
                                                (default: LabelInterpolation.NEAREST)
            load_keys (sequence of str): which label keys to load (default: all)
            centroid_in_annotation (bool): whether region centroids have to overlap with the annotation (default: False)
            linear_fill_value (float): fill value for linear interpolation outside of convex hull (default: np.nan)
        """
        if interpolation == LabelInterpolation.NEAREST:
            func = scipy_interpolate.NearestNDInterpolator
        elif interpolation == LabelInterpolation.LINEAR:
            func = partial(
                scipy_interpolate.LinearNDInterpolator, fill_value=linear_fill_value
            )
        else:
            raise NotImplementedError(f"Unsupported interpolation: {interpolation}")

        with open(path) as stream:
            labels = json.load(stream)

        assert isinstance(labels, dict)
        load_keys = (
            load_keys or labels.keys()
        )  # Load everything if no load_keys were given

        label_funcs = {}

        for key in load_keys:
            points = np.asarray(labels[key][LabelField.POINTS.value])
            values = np.asarray(labels[key][LabelField.VALUES.value])

            assert len(points) == len(values)
            label_funcs[key] = func(points, values)

        def local_label_func(x, y):
            return {k: label_funcs[k](x, y) for k in label_funcs}

        self.local_label_func = local_label_func

    def get_labels(self, x: int, y: int):
        """
        Get global and local labels at (x, y).

        Args:
            x (int): x-coordinate in pixels at level 0
            y (int): y-coordinate in pixels at level 0

        Returns:
            labels (dict): e.g. {"tumor": [1,1,0, ...], ...}
        """
        if (self.local_label_func is None) and (self.global_label_func is None):
            raise RuntimeError("No labels loaded!")
        labels = {}
        if self.local_label_func is not None:
            labels.update(self.local_label_func(x, y))
        if self.global_label_func is not None:
            labels.update(self.global_label_func(x, y))
        return labels

    def setup_regions(
        self,
        size: Optional[Sequence[int]] = None,
        unit: SizeUnit = SizeUnit.PIXEL,
        level: int = 0,
        centroid_in_annotation: bool = False,
        annotation_align: bool = False,
        region_overlap: float = 0.0,
        with_labels: bool = False,
        filter_by_label_func: Optional[Callable] = None,
    ):
        """
        Load all suitable regions (and corresponding labels) on the slide.

        This will populate slide.regions and slide.labels.

        Args:
            size (int or (int, int)): (width, height) in specified unit (default: native)
            unit (SizeUnit): pixels or microns (default: pixels)
            level (int): which hierachy level to use (default: 0)
            centroid_in_annotation (bool): whether region centroids have to overlap with the annotation (default: False)
            annotation_align (bool): if `True` will align tiles with annotation bbox else native alignment is used (default: False)
            region_overlap (float): overlap between neighboring regions in [0,1) (default: 0)
            with_labels (bool): whether to load corresponding labels (default: False)
            filter_by_label_func (callable): should convert slide.labels dictionary into mask (`True` == keep) (default: None)
        """
        assert self.is_loaded
        self.level = level

        # Use native resolution if size is unspecified
        if size is None:
            size = self.native_sizes[self.level]
            unit = SizeUnit.PIXEL
        elif isinstance(size, int):
            size = (size, size)

        if unit == SizeUnit.MICRON:
            assert size is not None
            assert self.microns_per_pixel is not None
            size = [int(s / self.microns_per_pixel) for s in size]

        size = np.asarray(size)
        x_min, y_min, x_max, y_max = 0, 0, *(np.array(self.image_shape[1::-1]) / size)

        if centroid_in_annotation:
            assert self.annotations is not None
            # Rescale pixel coordinates into tile coordinates
            annotation = shapely_geometry.GeometryCollection(
                [annotation.geometry for annotation in self.annotations]
            )
            scaled = shapely_affinity.scale(
                geom=annotation, xfact=1 / size[0], yfact=1 / size[1], origin=(0, 0)
            )
            if annotation_align:
                x_min, y_min, x_max, y_max = scaled.bounds

        step = 1.0 - region_overlap  # step is in tile coordinates
        x, y = np.meshgrid(
            np.arange(x_min, x_max, step),
            np.arange(y_min, y_max, step),
        )
        grid = np.stack([x.flatten(), y.flatten()])
        grid = grid.T

        if centroid_in_annotation:
            grid = grid + 0.5  # +0.5 for centroid of tiles
            # Round float coordinates to nearest grid idx
            x_idx, y_idx = grid.T.round().astype(int)

            # Create binary mask of tiles overlapping the annotation
            mask = rio_rasterize(
                shapes=[scaled], out_shape=(np.array([y_max, x_max]) + 2).astype(int)
            ).astype(bool)

            grid = grid[mask[y_idx, x_idx]]

        # Rescale to pixel coordinates
        regions = (size * grid).round().astype(int)

        labels = None
        if with_labels or (filter_by_label_func is not None):
            x, y = (regions + size / 2).T
            labels = self.get_labels(x, y)
            if filter_by_label_func is not None:
                filter_mask = filter_by_label_func(labels)
                regions = regions[filter_mask]
                if with_labels:
                    labels = {k: labels[k][filter_mask] for k in labels}
                else:
                    labels = None

        sizes = np.broadcast_to(size, (len(regions), 2))
        self.regions = np.concatenate([regions, sizes], axis=1)
        self.labels = labels

    def read_region(self, *args, **kwargs) -> ArrayLike:
        """Wrap the call to a CuImage to return an array for convenience."""
        if not self.is_loaded:
            self.image = cucim.CuImage(self.image)
            self.is_loaded = True
        return np.asarray(self.image.read_region(*args, **kwargs, level=self.level))
