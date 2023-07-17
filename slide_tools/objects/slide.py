import json
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Union

import cucim
import numpy as np
import openslide
import rasterio
import xmltodict
from numpy.typing import ArrayLike
from rasterio import features
from scipy import interpolate as scipy_interpolate
from shapely import geometry as shapely_geometry

from slide_tools.objects.annotation import Annotation
from slide_tools.objects.constants import (
    LabelField,
    LabelInterpolation,
    SizeUnit,
    SlideType,
)

__all__ = ["Slide"]


@dataclass
class Slide:
    """
    An object to collect attributes of a single slide, including annotation and label.

    Attributes:
        annotations (Annotation): Annotation object with geometry and properties
        image (CuImage, OpenSlide): a single loaded WSI
        label_func (callable): function that maps coordinates to labels: (x, y) -> label
        properties (dict): any properties concerning the slide
        microns_per_pixel (float): native/highest WSI resolution
        native_sizes (sequence): native tile sizes for all levels (level=0 is highest resolution)
        regions (np.array): array of coordinates and size per region ((x, y, w, h), ...)
        labels (dict): Dictionary of labels for all regions e.g. {"tumor": [0,0,1,...], "normal": [1,1,0,...]}
    """

    annotations: Optional[Sequence[Annotation]] = None
    image: Optional[Union[cucim.CuImage, openslide.OpenSlide]] = None
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
        assert os.path.exists(path), f"{path} does not exist"

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

    def load_wsi(
        self, path: str, raise_resolution: bool = True, backend: str = "cucim"
    ):
        """
        Load WSI with cuCIM or openslide and populate slide.image.

        Args:
            path (str): path
            raise_resolution (bool): Raise error on unknown pixel resolution (default: True)
            backend (str): 'cucim' or 'openslide' (default: 'cucim')

        Supports population of slide.microns_per_pixel for SlideType.{APERIO, TIFF}.
        """

        assert os.path.exists(path), f"{path} does not exist"

        self.backend = backend.lower()
        assert self.backend in (
            "cucim",
            "openslide",
        ), "Backend must be 'cucim' or 'openslide'"

        self.image_path = path
        if self.backend == "cucim":
            self._load_cucim(path)

        elif self.backend == "openslide":
            self._load_openslide(path)

        self.is_loaded = True

        if self.microns_per_pixel is None and raise_resolution:
            raise NotImplementedError(
                "Unknown WSI: Please add a way to extract microns_per_pixel for your WSI!"
            )

    def _load_cucim(self, path: str):
        self.loader = cucim.CuImage
        self.image = self.loader(path)
        self.image_shape = self.image.shape
        self.level_tile_sizes = np.array(self.image.resolutions["level_tile_sizes"])
        self.level_downsamples = np.array(
            self.image.resolutions["level_downsamples"], dtype=int
        )
        self.native_sizes = self.level_tile_sizes * self.level_downsamples[..., None]

        if SlideType.APERIO.value in self.image.metadata:
            self.microns_per_pixel = float(self.image.metadata["aperio"]["MPP"])
        elif SlideType.TIFF.value in self.image.metadata:
            if len(self.image.raw_metadata) > 0:
                props = xmltodict.parse(self.image.raw_metadata)["OME"]["Image"][
                    "Pixels"
                ]
                is_mu = (props["@PhysicalSizeXUnit"] == "µm") and (
                    props["@PhysicalSizeYUnit"] == "µm"
                )
                is_equal = props["@PhysicalSizeX"] == props["@PhysicalSizeY"]
                if is_mu and is_equal:
                    self.microns_per_pixel = float(props["@PhysicalSizeX"])

                if self.microns_per_pixel is None and raise_resolution:
                    raise RuntimeError(
                        f"Could not extract microns_per_pixel from:\n{props}"
                    )

    def _load_openslide(self, path: str):
        self.loader = openslide.open_slide
        self.image = self.loader(path)
        self.image_shape = [
            *self.image.dimensions[::-1],
            3,
        ]  # openslide does not provide channel number

        self.level_tile_sizes = np.array(
            [
                [
                    self.image.properties[f"openslide.level[{l}].tile-width"],
                    self.image.properties[f"openslide.level[{l}].tile-height"],
                ]
                for l in range(self.image.level_count)
            ],
            dtype=int,
        )

        self.level_downsamples = np.array(self.image.level_downsamples, dtype=int)
        self.native_sizes = self.level_tile_sizes * self.level_downsamples[..., None]

        mpp_x = float(self.image.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(self.image.properties[openslide.PROPERTY_NAME_MPP_X])
        assert mpp_x == mpp_y, f"Unequal resolutions {mpp_x=} != {mpp_y=}"
        assert (mpp_x is not None) and (mpp_x > 0), f"Unrecognized resolution {mpp_x=}"
        self.microns_per_pixel = mpp_x

    def unload_wsi(self):
        if self.is_loaded:
            self.image = None
            self.is_loaded = False

    def reload_wsi(self):
        if not self.is_loaded:
            self.image = self.loader(self.image_path)
            self.is_loaded = True

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
        assert os.path.exists(path), f"{path} does not exist"

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
        annotation_resolution_factor: int = 1,
        allow_out_of_bounds: bool = False,
        annotation_threshold: float = 0.5,
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
            annotation_resolution_factor (float): internal annotation resolution in units of region resolution (default: 1)
            allow_out_of_bounds (bool): Whether to include last row & column of regions to cover slide/annotation (default: False)
            annotation_threshold (float): Used for thresholding average pooled annotation (default: 0.5)
        """
        assert self.is_loaded
        assert isinstance(annotation_resolution_factor, int)
        self.level = level

        # Use native resolution if size is unspecified
        if size is None:
            size = self.native_sizes[self.level]
            unit = SizeUnit.PIXEL
        elif isinstance(size, int) and unit == SizeUnit.PIXEL:
            if size <= 10:
                native_size = self.native_sizes[self.level]
                size = (size * native_size).astype(int)
            else:
                size = (size, size)

        if unit == SizeUnit.MICRON:
            assert size is not None
            assert self.microns_per_pixel is not None
            if isinstance(size, int) or isinstance(size, float):
                size = (size, size)
            size = [np.round(s / self.microns_per_pixel) for s in size]

        region_dims = np.asarray(size).astype(int)

        boundary_func = np.ceil if allow_out_of_bounds else np.floor
        dims = {
            "region": {
                "w": region_dims[0],
                "h": region_dims[1],
            },
            "slide": {
                "w": self.image_shape[1],
                "h": self.image_shape[0],
            },
            # grid is aligned to slide boundaries
            "grid": {
                "x_min": 0,
                "y_min": 0,
                "x_max": boundary_func(self.image_shape[1] / region_dims[0]),
                "y_max": boundary_func(self.image_shape[0] / region_dims[1]),
                "w": int(boundary_func(self.image_shape[1] / region_dims[0])),
                "h": int(boundary_func(self.image_shape[0] / region_dims[1])),
            },
        }

        if centroid_in_annotation:
            assert self.annotations is not None
            # Rescale pixel coordinates into tile coordinates
            annotation = shapely_geometry.GeometryCollection(
                [annotation.geometry for annotation in self.annotations]
            )

            if annotation_align:
                # Recalculate grid with annotation in mind
                x_min, y_min, x_max, y_max = annotation.bounds
                x_min, x_max = np.array([x_min, x_max]) / dims["region"]["w"]
                y_min, y_max = np.array([y_min, y_max]) / dims["region"]["h"]
                x_bound = dims["slide"]["w"] / dims["region"]["w"]
                y_bound = dims["slide"]["h"] / dims["region"]["h"]
                assert (
                    (x_min >= 0)
                    and (y_min >= 0)
                    and (x_max <= x_bound)
                    and (y_max <= y_bound)
                )
                w = int(boundary_func(x_max - x_min))
                h = int(boundary_func(y_max - y_min))

                dims["grid"] = {
                    "x_min": x_min,
                    "x_max": x_min + w,
                    "y_min": y_min,
                    "y_max": y_min + h,
                    "w": w,
                    "h": h,
                }

        step = 1.0 - region_overlap  # step is in tile coordinates
        dims["grid"]["step"] = step
        # Top-Left corner of tiles
        x, y = np.meshgrid(
            np.arange(dims["grid"]["x_min"], dims["grid"]["x_max"] - 1 + 1e-6, step),
            np.arange(dims["grid"]["y_min"], dims["grid"]["y_max"] - 1 + 1e-6, step),
        )
        grid = np.stack([x.flatten(), y.flatten()])
        grid = grid.T

        if centroid_in_annotation:
            mask_shape = np.array([dims["grid"]["h"], dims["grid"]["w"]])
            mask_shape = annotation_resolution_factor * mask_shape
            transform = rasterio.transform.from_origin(
                dims["grid"]["x_min"] * dims["region"]["w"],
                dims["grid"]["y_max"] * dims["region"]["h"],
                dims["region"]["w"] / annotation_resolution_factor,
                dims["region"]["h"] / annotation_resolution_factor,
            )
            mask = features.geometry_mask(
                geometries=[annotation],
                out_shape=mask_shape,
                transform=transform,
                invert=True,
            )
            mask = mask[::-1, :]  # mask is always upside down
            if annotation_resolution_factor > 1:
                k = annotation_resolution_factor
                average_pooled = (
                    mask.reshape(
                        mask_shape[0] // k,
                        k,
                        mask_shape[1] // k,
                        k,
                    )
                    .swapaxes(1, 2)
                    .mean(axis=(2, 3))
                )
                mask = average_pooled >= annotation_threshold

            origin = np.array([dims["grid"]["x_min"], dims["grid"]["y_min"]])
            centroid = (grid + 0.5) - origin  # +0.5 for centroid of tiles
            x_idx, y_idx = np.floor(centroid.T).astype(int)
            grid = grid[mask[y_idx, x_idx]]

        # Rescale to pixel coordinates
        regions = (region_dims * grid).round().astype(int)

        labels = None
        if with_labels or (filter_by_label_func is not None):
            x, y = (regions + region_dims / 2).T
            labels = self.get_labels(x, y)
            if filter_by_label_func is not None:
                filter_mask = filter_by_label_func(labels)
                regions = regions[filter_mask]
                if with_labels:
                    labels = {k: labels[k][filter_mask] for k in labels}
                else:
                    labels = None

        sizes = np.broadcast_to(region_dims, (len(regions), 2))
        self.regions = np.concatenate([regions, sizes], axis=1)
        self.labels = labels
        self.dims = dims

    def read_region(self, *args, **kwargs) -> ArrayLike:
        """Wrap the call to a CuImage to return an array for convenience."""
        if not self.is_loaded:
            self.image = self.image_loader(self.image_path)
            self.is_loaded = True
        return np.asarray(self.image.read_region(*args, **kwargs, level=self.level))[
            ..., :3
        ]
