import random
from functools import partial
from typing import Callable, Optional, Sequence, Union

import numpy as np
from torch.utils.data import Dataset

from ..objects import BalanceMode, LabelInterpolation, SizeUnit, Slide

__all__ = ["TileLevelDataset"]


class TileLevelDataset(Dataset):
    def __init__(
        self,
        slide_paths: Sequence[str],
        annotation_paths: Optional[Sequence[str]] = None,
        label_paths: Optional[Sequence[str]] = None,
        global_labels: Optional[Sequence] = None,
        img_tfms: Optional[Callable] = None,
        label_tfms: Optional[Callable] = None,
        **kwargs
    ):
        """
        Map-style dataset for tile-level training of WSI.

        The idea is to call .setup_epoch(...) at the beginning of every epoch to sample
        new tiles from the pool of available WSI. This will populate .samples which
        are just indices ((slide_idx, region_idx), ...). Balancing by slide size and tile labels
        are supported. Shuffling is also handled inside this dataset (so you should disbale it
        in your DataLoader)and can be done in chunks such that a worker is more likely to read
        a few (neighboring) tiles from the same slide which should lead to some speed up.

        Regions are np.array with shape=[height, width, channels].

        Args:
            slide_paths (sequence of str): WSI paths (must be readable by cuCIM)
            annotation_paths (sequence of str, optional): annotation-GeoJSON paths
            label_paths (sequence of str, optional): label-JSON paths
            global_labels (dict, optional): global label dictionary
            img_tfms (callable, optional): image transform (np.array -> np.array or torch.tensor)
            label_tfms (callable, optional): label transform (dict of np.array -> dict of np.array)
            **kwargs: passed down to .setup_regions() and .setup_epoch()
        """
        if annotation_paths is not None:
            assert len(slide_paths) == len(annotation_paths)
        if label_paths is not None:
            assert len(slide_paths) == len(annotation_paths)
        if (label_paths is not None) and (global_labels is not None):
            raise RuntimeError("Can not have local and global labels at the same time!")

        self.img_tfms = img_tfms
        self.label_tfms = label_tfms

        self.slides = []
        for i in range(len(slide_paths)):
            slide = Slide()
            slide.load_wsi(slide_paths[i])
            if annotation_paths is not None:
                slide.load_annotations_from_geojson(
                    path=annotation_paths[i],
                    simplify_tolerance=kwargs.get("simplify_tolerance", 0),
                )
            if global_labels is not None:
                slide.set_global_label(labels[i])
            if label_paths is not None:
                slide.load_label_from_json(
                    path=label_paths[i],
                    interpolation=kwargs.get(
                        "interpolation", LabelInterpolation.NEAREST
                    ),
                    load_keys=kwargs.get("load_keys"),
                    linear_fill_value=kwargs.get("linear_fill_value", np.nan),
                )
            self.slides.append(slide)

        self.setup_regions(
            size=kwargs.get("size"),
            unit=kwargs.get("unit", SizeUnit.PIXEL),
            level=kwargs.get("level", 0),
            centroid_in_annotation=kwargs.get("centroid_in_annotation", False),
            annotation_align=kwargs.get("annotation_align", False),
            region_overlap=kwargs.get("region_overlap", 0.0),
            with_labels=kwargs.get("with_labels", False),
        )

        self.setup_epoch(
            balance_size_by=kwargs.get("balance_size_by"),
            balance_label_key=kwargs.get("balance_label_key"),
            balance_label_bins=kwargs.get("balance_label_bins", 10),
            shuffle=kwargs.get("shuffle", False),
            shuffle_chunk_size=kwargs.get("shuffle_chunk_size", 1),
        )

    def setup_regions(
        self,
        size: Optional[Sequence[int]] = None,
        unit: SizeUnit = SizeUnit.PIXEL,
        level: int = 0,
        centroid_in_annotation: bool = False,
        annotation_align: bool = False,
        region_overlap: float = 0.0,
        with_labels: bool = False,
    ):
        """
        Call .setup_regions(...) for all .slides
        See `slide_tools.objects.Slide`
        """
        for slide in self.slides:
            slide.setup_regions(
                size=size,
                unit=unit,
                level=level,
                centroid_in_annotation=centroid_in_annotation,
                annotation_align=annotation_align,
                region_overlap=region_overlap,
                with_labels=with_labels,
            )

    def setup_epoch(
        self,
        balance_size_by: Optional[Union[BalanceMode, int]] = None,
        balance_label_key: Optional[str] = None,
        balance_label_bins: int = 10,
        shuffle: bool = False,
        shuffle_chunk_size: int = 1,
    ):
        """
        Populate .samples with corresponding region from all .slides to iterate over.

        Args:
            balance_size_by (BalanceMode or int, optional): Determines N_samples = len(.slides) * balance_size_by
            balance_label_key (str, optional): label used for balancing (will be digitized, see np.digitize)
            balance_label_bins (int, optional): number of bins for label digitization (default: 10)
            shuffle (bool): shuffle samples or not
            shuffle_chunk_size (int): chunk samples before shuffling for faster loading (default: 1)

        Balancing by size and/or label will determine a regions weight for being sampled with replacement.
        Shuffling can be done in chunks so that a worker is more likely to read multiple (neighboring) tiles
        from one slide which will likely lead to a speedup. Keep your batch size in mind as you will likely get
        batch_size/shuffle_chunk_size different slides inside each batch.
        """
        sizes = np.array([len(slide.regions) for slide in self.slides])
        samples = np.concatenate(
            [
                np.stack([np.broadcast_to(i, size), np.arange(size)]).T
                for i, size in enumerate(sizes)
            ]
        )
        weight = None

        if balance_size_by is not None:
            assert isinstance(balance_size_by, BalanceMode) or isinstance(
                balance_size_by, int
            )
            weight = np.concatenate([np.broadcast_to(1 / size, size) for size in sizes])
            if isinstance(balance_size_by, int):
                num_samples = balance_size_by
            elif balance_size_by == BalanceMode.MIN:
                num_samples = np.min(sizes)
            elif balance_size_by == BalanceMode.MEDIAN:
                num_samples = np.median(sizes)
            elif balance_size_by == BalanceMode.MEAN:
                num_samples = np.mean(sizes)
            elif balance_size_by == BalanceMode.MAX:
                num_samples = np.max(sizes)

            num_samples = int(len(sizes) * num_samples)

        # Labels are digitized before balancing to also work with continous labels
        if balance_label_key is not None:
            labels = np.concatenate(
                [slide.labels[balance_label_key] for slide in self.slides]
            )
            assert len(labels) == sizes.sum()
            # 1e-6 because np.digitize does not include right most bin edge
            bins = np.linspace(0, labels.max() + 1e-6, balance_label_bins + 1)
            labels_bin_idx = np.digitize(labels, bins=bins)
            bin_values, bin_counts = np.unique(labels_bin_idx, return_counts=True)

            label_weight = np.zeros(len(labels))
            for value, count in zip(bin_values, bin_counts):
                label_weight[labels_bin_idx == value] = 1.0 / count

            if weight is not None:
                weight = weight * label_weight
            else:
                weight = label_weight
                num_samples = len(samples)

        if weight is not None:
            weight = weight / weight.sum()  # choice needs the sum == 1
            idx = np.random.choice(
                len(weight), size=num_samples, replace=True, p=weight
            )
            if shuffle and shuffle_chunk_size == 1:
                shuffle = False  # idx is already shuffled
            else:
                idx = np.sort(idx)  # Reverse shuffling from choice
            samples = samples[idx]

        if shuffle:
            if shuffle_chunk_size > 1:
                # Split into unique and duplicates
                # Explaination:
                # We want chunks to be made of neighboring tiles but we sample with replacement so
                # sorting will put the same exact tile multiple times in a chunk. To prevent this from
                # happening we will only sort the unique values and sprinkle the duplicates uniformly
                # back in and only then will shuffle with our chunk_size.
                unique, counts = np.unique(samples, return_counts=True, axis=0)
                have_duplicates = counts > 1
                duplicates = unique[have_duplicates]
                duplicate_counts = counts[have_duplicates] - 1
                rest = np.repeat(duplicates, duplicate_counts, axis=0)
                np.random.shuffle(rest)  # Shuffle before sprinkling back in

                # Sprinkle duplicates uniformly into unique
                samples_sorted = np.empty_like(samples)
                duplicate_idx = np.random.choice(
                    len(samples), size=len(rest), replace=False
                )
                duplicate_mask = np.zeros(len(samples), dtype=bool)
                duplicate_mask[duplicate_idx] = True
                samples_sorted[duplicate_mask] = rest
                samples_sorted[~duplicate_mask] = unique
                samples = samples_sorted

                # Chunk shuffle
                idx = np.arange(len(samples))
                chunks = np.array_split(idx, len(idx) // shuffle_chunk_size)
                random.shuffle(
                    chunks
                )  # Using random because numpy can not handle N % chunk_size != 0
                idx_shuffled = np.concatenate(chunks)
                samples = samples[idx_shuffled]
            else:
                np.random.shuffle(samples)

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slide_idx, region_idx = self.samples[idx]

        slide = self.slides[slide_idx]
        region = slide.regions[region_idx]
        img = slide.read_region(location=region[:2], size=region[2:])

        if self.img_tfms is not None:
            img = self.img_tfms(img)

        if slide.labels is not None:
            label = {k: slide.labels[k][region_id] for k in slide.labels}
            if self.label_tfms is not None:
                label = self.label_tfms(label)
            return img, label
        else:
            return img
