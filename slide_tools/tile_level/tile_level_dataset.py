import random
import warnings
from functools import partial
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
        return_labels: Optional[Union[Sequence[str], str]] = None,
        return_index: bool = False,
        verbose: bool = False,
        lazy_loading: bool = False,
        location_wiggle: Optional[float] = None,
        simple_epoch: bool = False,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """
        Map-style dataset for tile-level training of WSI.

        The idea is to call .setup_epoch(...) at the beginning of every epoch to sample
        new tiles from the pool of available WSI. This will populate .samples which
        are just indices ((slide_idx, region_idx), ...). Balancing by slide size and tile labels
        is supported. Shuffling is also handled inside this dataset (so you should disbale it
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

        self.img_tfms = img_tfms
        self.label_tfms = label_tfms
        self.return_labels = (
            [return_labels] if isinstance(return_labels, str) else return_labels
        )
        self.verbose = verbose
        self.return_index = return_index
        self.location_wiggle = location_wiggle
        
        self.rng = np.random.default_rng(random_state or torch.random.initial_seed())

        self.slides = []

        iterator = range(len(slide_paths))
        if self.verbose:
            iterator = tqdm(iterator, desc="Load slides")

        for i in iterator:
            slide = Slide()
            slide.load_wsi(slide_paths[i])
            if annotation_paths is not None:
                slide.load_annotations_from_geojson(
                    path=annotation_paths[i],
                    simplify_tolerance=kwargs.get("simplify_tolerance", 0),
                )
            if global_labels is not None:
                slide.set_global_label(global_labels[i])
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
            filter_by_label_func=kwargs.get("filter_by_label_func"),
        )

        # Remove empty slides
        empty = [i for i, slide in enumerate(self.slides) if len(slide.regions) == 0]
        for idx in empty[::-1]:
            warnings.warn(
                f"{slide_paths[idx]} has zero regions and is removed from the dataset!",
                stacklevel=2,
            )
            self.slides.pop(idx)
        
        self.simple_epoch = simple_epoch
        if simple_epoch:
            self.setup_epoch_no_sampling(
                balance_strict_size_by=kwargs.get("balance_size_by"),
                shuffle=kwargs.get("shuffle", False),
            )
        else:
            self.setup_epoch_with_sampling(
                balance_size_by=kwargs.get("balance_size_by"),
                balance_label_key=kwargs.get("balance_label_key"),
                balance_label_bins=kwargs.get("balance_label_bins", 10),
                shuffle=kwargs.get("shuffle", False),
                shuffle_chunk_size=kwargs.get("shuffle_chunk_size", 1),
                with_replacement=kwargs.get("with_replacement", True),
                strict_size_balance=kwargs.get("strict_size_balance", False),
            )

        if lazy_loading:
            self.unload_wsi()

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
        Call .setup_regions(...) for all .slides
        See `slide_tools.objects.Slide`
        """

        iterator = self.slides
        if self.verbose:
            iterator = tqdm(iterator, desc="Setup Regions")

        for slide in iterator:
            slide.setup_regions(
                size=size,
                unit=unit,
                level=level,
                centroid_in_annotation=centroid_in_annotation,
                annotation_align=annotation_align,
                region_overlap=region_overlap,
                with_labels=with_labels,
                filter_by_label_func=filter_by_label_func,
            )
            
    def reload(self):
        if self.simple_epoch:
            self.setup_epoch_no_sampling(**self._reload_kwargs)
        else:
            self.setup_epoch_with_sampling(**self._reload_kwargs)

    def setup_epoch_no_sampling(
        self,
        balance_strict_size_by: Optional[Union[BalanceMode, int]] = None,
        shuffle: bool = False,
    ):
        """
        Populate .samples with corresponding region from all .slides to iterate over.
        This is a simple version dedicated for very large datasets and pre-training.

        Args:
            balance_strict_size_by (BalanceMode or int, optional): Determines N_samples = len(.slides) * balance_size_by
            shuffle (bool): shuffle samples or not
            shuffle_chunk_size (int): chunk samples before shuffling for faster loading (default: 1)

        Shuffling can be done in chunks so that a worker is more likely to read multiple (neighboring) tiles
        from one slide which will likely lead to a speedup. Keep your batch size in mind as you will likely get
        batch_size/shuffle_chunk_size different slides inside each batch.
        """
        if self.verbose:
            print("Setting up simple epoch")

        sizes = np.array([len(slide.regions) for slide in self.slides])
        samples = np.concatenate(
            [
                np.stack([np.broadcast_to(i, size), np.arange(size)]).T
                for i, size in enumerate(sizes)
            ]
        )

        if balance_strict_size_by is not None:
            assert isinstance(balance_strict_size_by, BalanceMode) or isinstance(
                balance_strict_size_by, int
            )

            if isinstance(balance_strict_size_by, int):
                num_samples = balance_strict_size_by
            elif balance_size_by == BalanceMode.MIN:
                num_samples = np.min(sizes)
            elif balance_size_by == BalanceMode.MEDIAN:
                num_samples = np.median(sizes)
            elif balance_size_by == BalanceMode.MEAN:
                num_samples = np.mean(sizes)
            elif balance_size_by == BalanceMode.MAX:
                num_samples = np.max(sizes)

            sample_idx = []
            offset = 0
            for size in sizes:
                n = int(min(size, num_samples))
                idx = offset + self.rng.choice(size, size=n, replace=False)
                sample_idx.append(idx)
                offset += size

            sample_idx = np.concatenate(sample_idx)
            samples = samples[sample_idx]

        if shuffle:
            self.rng.shuffle(samples)

        self.samples = samples

        self._reload_kwargs = {
            "balance_strict_size_by": balance_strict_size_by,
            "shuffle": shuffle,
        }

    def setup_epoch_with_sampling(
        self,
        balance_size_by: Optional[Union[BalanceMode, int]] = None,
        balance_label_key: Optional[str] = None,
        balance_label_bins: int = 10,
        shuffle: bool = False,
        shuffle_chunk_size: int = 1,
        with_replacement: bool = True,
        strict_size_balance: bool = False,
    ):
        """
        Populate .samples with corresponding region from all .slides to iterate over.

        Args:
            balance_size_by (BalanceMode or int, optional): Determines N_samples = len(.slides) * balance_size_by
            balance_label_key (str, optional): label used for balancing (will be digitized, see np.digitize)
            balance_label_bins (int, optional): number of bins for label digitization (default: 10)
            shuffle (bool): shuffle samples or not
            shuffle_chunk_size (int): chunk samples before shuffling for faster loading (default: 1)
            with_replacement (bool): Whether to sample with or without replacement (default: True)
            strict_size_balance (bool): Will choose random min(#tiles, balance_size_by) per slide (default: False)

        Balancing by size and/or label will determine a regions weight for being sampled with or without replacement.
        Shuffling can be done in chunks so that a worker is more likely to read multiple (neighboring) tiles
        from one slide which will likely lead to a speedup. Keep your batch size in mind as you will likely get
        batch_size/shuffle_chunk_size different slides inside each batch.
        """
        if self.verbose:
            print("Setting up epoch")

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

            if strict_size_balance:
                weight = np.zeros(len(samples))
                offset = 0
                for size in sizes:
                    n = min(size, num_samples)
                    idx = default_rng(n).choice(size, size=n, replace=False)
                    weight[idx + offset] = 1 / n
                    offset += size
            else:
                weight = np.concatenate(
                    [np.broadcast_to(1 / size, size) for size in sizes]
                )

            num_samples = int(len(sizes) * num_samples)
            if not with_replacement:
                num_samples = min(num_samples, sizes.sum())

        # Labels are digitized before balancing to also work with continous labels
        if balance_label_key is not None:
            labels = np.concatenate(
                [slide.labels[balance_label_key] for slide in self.slides]
            )
            if weight is not None:
                if (weight == 0).sum() > 0:
                    labels = labels[weight != 0]

            assert len(labels) == sizes.sum()
            # 1e-6 because np.digitize does not include right most bin edge
            bins = np.linspace(0, labels.max() + 1e-6, balance_label_bins + 1)
            labels_bin_idx = np.digitize(labels, bins=bins)
            bin_values, bin_counts = np.unique(labels_bin_idx, return_counts=True)

            label_weight = np.zeros(len(labels))
            for value, count in zip(bin_values, bin_counts):
                label_weight[labels_bin_idx == value] = 1.0 / count

            if weight is not None:
                if len(weight) != label_weight.sum():
                    weight[weight != 0] = weight[weight != 0] * label_weight
                weight = weight * label_weight
            else:
                weight = label_weight
                num_samples = len(samples)

        if weight is not None:
            weight = weight / weight.sum()  # choice needs the sum == 1
            if not with_replacement:
                num_samples = min(num_samples, (weight != 0).sum())
            idx = self.rng.choice(
                len(weight), size=num_samples, replace=with_replacement, p=weight
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
                self.rng.shuffle(rest)  # Shuffle before sprinkling back in

                # Sprinkle duplicates uniformly into unique
                samples_sorted = np.empty_like(samples)
                duplicate_idx = self.rng.choice(
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
                random.seed(self.rng.bit_generator.random_raw())
                random.shuffle(
                    chunks
                )  # Using random because numpy can not handle N % chunk_size != 0
                idx_shuffled = np.concatenate(chunks)
                samples = samples[idx_shuffled]
            else:
                self.rng.shuffle(samples)

        self.samples = samples
        
        self._reload_kwargs = {
            "balance_size_by": balance_size_by,
            "balance_label_key": balance_label_key,
            "balance_label_bins": balance_label_bins,
            "shuffle": shuffle,
            "shuffle_chunk_size": shuffle_chunk_size,
            "with_replacement": with_replacement,
            "strict_size_balance": strict_size_balance,
        }

    def unload_wsi(self):
        for slide in self.slides:
            slide.unload_wsi()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slide_idx, region_idx = self.samples[idx]

        slide = self.slides[slide_idx]
        region = slide.regions[region_idx]
        location, size = region[:2], region[2:]
        if self.location_wiggle is not None:
            wiggle = self.location_wiggle * size * 2 * (self.rng.random(2) - 0.5)
            location += wiggle.astype(int)
        img = slide.read_region(location=location, size=size)

        if self.img_tfms is not None:
            img = self.img_tfms(img)

        out = {"img": img}

        if self.return_labels is not None:
            label = {k: slide.labels[k][region_idx] for k in self.return_labels}
            if self.label_tfms is not None:
                label = self.label_tfms(label)
            out.update(label)

        if self.return_index:
            out["slide_idx"] = slide_idx
            out["region_idx"] = region_idx

        return out
