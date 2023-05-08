import os
from argparse import ArgumentParser

# Typing
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.nn import Identity
from torch.utils.data import DataLoader

from ..objects import BalanceMode, LabelInterpolation, SizeUnit
from .tile_level_dataset import TileLevelDataset

__all__ = ["TileLevelDataModule"]


class TileLevelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = None,
        batch_size: int = 1,
        num_workers: int = 1,
        csv_train: Optional[str] = None,
        csv_valid: Optional[str] = None,
        csv_test: Optional[str] = None,
        tfms_train: Callable = Identity(),
        tfms_valid: Callable = Identity(),
        tfms_test: Callable = Identity(),
        column_slide: Optional[str] = None,
        column_annotation: Optional[str] = None,
        column_label: Optional[str] = None,
        columns_global_label: Optional[Sequence[str]] = None,
        slide_simplify_tolerance: int = 0,
        slide_interpolation: Union[str, LabelInterpolation] = "nearest",
        slide_load_keys: Optional[Sequence[str]] = None,
        slide_linear_fill_value: float = np.nan,
        regions_size: Optional[int] = None,
        regions_unit: Union[str, SizeUnit] = "pixel",
        regions_level: int = 0,
        regions_centroid_in_annotation: bool = False,
        regions_annotation_align: bool = False,
        regions_region_overlap: float = 0.0,
        regions_with_labels: bool = False,
        regions_return_labels: Optional[Union[Sequence[str], str]] = None,
        regions_filter_by_label_func: Optional[Callable] = None,
        regions_annotation_resolution_factor: float = 1.,
        epoch_balance_size_by: Optional[Union[BalanceMode, int, str]] = None,
        epoch_balance_label_key: Optional[str] = None,
        epoch_balance_label_bins: int = 10,
        epoch_shuffle: bool = False,
        epoch_shuffle_chunk_size: int = 1,
        epoch_with_replacement: bool = True,
        epoch_strict_size_balance: bool = False,
        verbose: bool = False,
        pin_memory: bool = False,
        **kwargs
    ):
        """
        Datamodule based on TileLevelDataset.

        Args:
            root (str): Everything (the csv path itsels and paths inside the csv) is relative to this
            batch_size (int): Batch size (see `torch.utils.data.DataLoader`)
            num_workers (int): Number of parallel workers filling the queue (see `torch.utils.data.DataLoader`)
            csv_train (str): Realtive path (to root) to csv containing columns with paths to slides (, annotations and labels)
            csv_valid (str): see csv_train
            csv_test (str): see csv_train
            tfms_train (Callable): Transform to be applied to training images
            tfms_valid (Callable): Transform to be applied to validation images
            tfms_test (Callable): Transform to be applied to test images
            column_slide (str): Column name in csv containing relative (to root) slide paths
            column_annotation (str): Column name in csv containing relative (to root) annotation paths
            column_label (str): Column name in csv containing relative (to root) slide paths
            columns_global_label (sequence of str): Column name(s) containing global labels
            slide_simplify_tolerance: see `slide_tools.Slide`
            slide_interpolation: see `slide_tools.Slide`
            slide_load_keys: see `slide_tools.Slide`
            slide_linear_fill_value: see `slide_tools.Slide`
            regions_size: see `slide_tools.TileLevelDataset.setup_regions()`
            regions_unit: see `slide_tools.TileLevelDataset.setup_regions()`
            regions_level: see `slide_tools.TileLevelDataset.setup_regions()`
            regions_centroid_in_annotation: see `slide_tools.TileLevelDataset.setup_regions()`
            regions_annotation_align:see `slide_tools.TileLevelDataset.setup_regions()`
            regions_region_overlap: see `slide_tools.TileLevelDataset.setup_regions()`
            regions_with_labels: see `slide_tools.TileLevelDataset.setup_regions()`
            regions_return_labels: see `slide_tools.TileLevelDataset.setup_regions()`
            regions_filter_by_label_func: see `slide_tools.TileLevelDataset.setup_regions()`
            regions_annotation_resolution_factor: see `slide_tools.TileLevelDataset.setup_regions()`
            epoch_balance_size_by: see `slide_tools.TileLevelDataset.setup_epoch()`
            epoch_balance_label_key: see `slide_tools.TileLevelDataset.setup_epoch()`
            epoch_balance_label_bins: see `slide_tools.TileLevelDataset.setup_epoch()`
            epoch_shuffle: see `slide_tools.TileLevelDataset.setup_epoch()`
            epoch_shuffle_chunk_size: see `slide_tools.TileLevelDataset.setup_epoch()`
            epoch_with_replacement: see `slide_tools.TileLevelDataset.setup_epoch()`
            epoch_strict_size_balance: see `slide_tools.TileLevelDataset.setup_epoch()`
            verbose: see `slide_tools.TileLevelDataset`
            pin_memory: see `torch.utils.data.DataLoader`


        Attributes:
            train_dataloader: Create dataloader from csv_train given all other arguments
            val_dataloader: Create dataloader from csv_train given all other arguments
            test_dataloader: Create dataloader from csv_train given all other arguments
            add_argparse_args: Add all arguments to a parser

        Note: You want to set reload_dataloaders_every_n_epochs=1 in your `pl.Trainer`.
              This datamodule works with DistributedDataParallel out of the box.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["tfms_train", "tfms_valid", "tfms_test"])
        self.correct_hparams()

        self.tfms_train = tfms_train
        self.tfms_valid = tfms_valid
        self.tfms_test = tfms_test

        if isinstance(self.hparams.columns_global_label, str):
            self.hparams.columns_global_label = [self.hparams.columns_global_label]

    def correct_hparams(self):
        """Helper function to split the hparams and correct them."""

        # Helper to split off hparams beginning with prefix
        def split_off_by(prefix, hparams):
            return pl.utilities.parsing.AttributeDict(
                {
                    key[len(prefix) :]: hparams[key]
                    for key in hparams
                    if key.startswith(prefix)
                }
            )

        self.hparams.root = os.path.abspath(self.hparams.root)
        self.slide_kwargs = split_off_by("slide_", self.hparams)
        self.region_kwargs = split_off_by("regions_", self.hparams)
        self.epoch_kwargs = split_off_by("epoch_", self.hparams)

        if isinstance(self.slide_kwargs.interpolation, str):
            self.slide_kwargs.interpolation = LabelInterpolation[
                self.slide_kwargs.interpolation.upper()
            ]
        if isinstance(self.region_kwargs.unit, str):
            self.region_kwargs.unit = SizeUnit[self.region_kwargs.unit.upper()]
        if isinstance(self.epoch_kwargs.balance_size_by, str):
            self.epoch_kwargs.balance_size_by = BalanceMode[
                self.epoch_kwargs.balance_size_by.upper()
            ]

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Helper to join root onto relative path series
        rootify_ = lambda path: os.path.join(self.hparams.root, path)

        def rootify(series):
            if series is None:
                return None
            else:
                return series.apply(rootify_)

        # Helper to turn label frame (mulitple columns) to dict
        def get_columns_as_records(frame, columns):
            if columns is None:
                return None
            else:
                return frame[columns].to_dict("records")

        if stage == "fit" or stage is None:
            frame_train = pd.read_csv(rootify_(self.hparams.csv_train))
            frame_valid = pd.read_csv(rootify_(self.hparams.csv_valid))

            self.ds_train = TileLevelDataset(
                slide_paths=rootify(frame_train[self.hparams.column_slide]),
                annotation_paths=rootify(
                    frame_train.get(self.hparams.column_annotation)
                ),
                label_paths=rootify(frame_train.get(self.hparams.column_label)),
                global_labels=get_columns_as_records(
                    frame_train, self.hparams.columns_global_label
                ),
                img_tfms=self.tfms_train,
                **self.slide_kwargs,
                **self.region_kwargs,
                verbose=self.hparams.verbose,
            )

            self.ds_valid = TileLevelDataset(
                slide_paths=rootify(frame_valid[self.hparams.column_slide]),
                annotation_paths=rootify(
                    frame_valid.get(self.hparams.column_annotation)
                ),
                label_paths=rootify(frame_valid.get(self.hparams.column_label)),
                global_labels=get_columns_as_records(
                    frame_valid, self.hparams.columns_global_label
                ),
                img_tfms=self.tfms_valid,
                **self.slide_kwargs,
                **self.region_kwargs,
                verbose=self.hparams.verbose,
            )

        if stage == "test" or stage is None:
            frame_test = pd.read_csv(rootify_(self.hparams.csv_test))

            self.ds_test = TileLevelDataset(
                slide_paths=rootify(frame_test[self.hparams.column_slide]),
                annotation_paths=rootify(
                    frame_test.get(self.hparams.column_annotation)
                ),
                label_paths=rootify(frame_test.get(self.hparams.column_label)),
                global_labels=get_columns_as_records(
                    frame_test, self.hparams.columns_global_label
                ),
                img_tfms=self.tfms_test,
                **self.slide_kwargs,
                **self.region_kwargs,
                verbose=self.hparams.verbose,
            )

    def train_dataloader(
        self,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ):
        self.ds_train.reload()
        return DataLoader(
            self.ds_train,
            shuffle=False,
            batch_size=batch_size or self.hparams.batch_size,
            drop_last=True,
            num_workers=num_workers or self.hparams.num_workers,
            pin_memory=pin_memory or self.hparams.pin_memory,
        )

    def val_dataloader(
        self,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ):
        return DataLoader(
            self.ds_valid,
            shuffle=False,
            batch_size=batch_size or self.hparams.batch_size,
            num_workers=num_workers or self.hparams.num_workers,
            pin_memory=pin_memory or self.hparams.pin_memory,
        )

    def test_dataloader(
        self,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ):
        return DataLoader(
            self.ds_test,
            shuffle=False,
            batch_size=batch_size or self.hparams.batch_size,
            num_workers=num_workers or self.hparams.num_workers,
            pin_memory=pin_memory or self.hparams.pin_memory,
        )

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument("--columns_global_label", nargs="+", type=str, default=None)
        parser.add_argument("--slide_load_keys", nargs="+", type=str, default=None)
        parser.add_argument(
            "--regions_return_labels", nargs="+", type=str, default=None
        )
        return pl.utilities.argparse.add_argparse_args(cls, parser, **kwargs)
