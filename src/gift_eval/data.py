# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import random
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow.compute as pc
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import get_seasonality, norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose

import datasets
from datasets import Dataset as HF_Dataset
from datasets.utils.logging import disable_progress_bar

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "H": 48,  # Hourly
    "h": 48,
    "D": 14,  # Daily
    "d": 14,
    "W": 13,  # Weekly
    "w": 13,
    "M": 18,  # Monthly
    "m": 18,
    "ME": 18,  # End of month
    "Q": 8,  # Quarterly
    "q": 8,
    "QE": 8,  # End of quarter
    "A": 6,  # Annualy/yearly
    "y": 6,
    "YE": 6,  # End of year
}

PRED_LENGTH_MAP = {
    "S": 60,  # Seconds
    "s": 60,
    "T": 48,  # Minutely
    "min": 48,
    "H": 48,  # Hourly
    "h": 48,
    "D": 30,  # Daily
    "d": 30,
    "W": 8,  # Weekly
    "w": 8,
    "M": 12,  # Monthly
    "m": 12,
    "ME": 12,
    "Q": 8,  # Quarterly
    "q": 8,
    "QE": 8,
    "y": 6,  # Annualy/yearly
    "A": 6,
}

# Prediction lengths from TFB: https://arxiv.org/abs/2403.20150
TFB_PRED_LENGTH_MAP = {
    "U": 8,
    "T": 8,  # Minutely
    "H": 48,  # Hourly
    "h": 48,
    "D": 14,  # Daily
    "W": 13,  # Weekly
    "M": 18,  # Monthly
    "Q": 8,  # Quarterly
    "A": 6,  # Annualy/yearly
}


class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15


def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry


def fix_1d_array(data_entry: DataEntry) -> DataEntry:
    """
    Converts arrays of shape (`time_steps`,) in a DataEntry into shape
    (1, `time_steps`). Excludes the "target" field.

    Use this to avoid "bad shape - expected 2 dimensions, got 1" errors.

    Args:
        data_entry (DataEntry): A dictionary representing a single time series.

    Returns:
        DataEntry: The modified dictionary with 1D arrays reshaped to 2D.
    """
    for key, value in data_entry.items():
        if key == "target" or not isinstance(value, np.ndarray):
            continue
        data_entry[key] = value[None, :] if value.ndim == 1 else value
    return data_entry


class MultivariateToUnivariate(Transformation):
    """
    Converts a multivariate time series into a univariate time series.
    """

    def __init__(self, field):
        self.field = field

    def __call__(
        self,
        data_it: Iterable[DataEntry],
        is_train: bool = False,
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry


class Dataset:
    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = True,
        storage_env_var: str = "GIFT_EVAL",
        metadata_directory: str = "resources",
        verbose: bool = False,
        limit: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = 42,
    ):
        """
        Wrapper for loading and processing a GIFT-Eval dataset.

        This class supports:
        - Loading datasets from disk using Hugging Face Datasets.
        - Setting a limit on the number of series used.
        - Subsampling a fraction of the dataset.
        - Automatically converting multivariate time series to univariate.
        - Providing metadata (e.g., frequency, prediction length, etc.).
        - Creating GluonTS-compatible training, validation, and test splits.

        **NOTE:**  Using `to_univariate=True` converts each multivariate time
        series of shape `(T, D)` (where `T` is the number of time steps and `D`
        is the number of dimensions) into `D` separate univariate time series
        of shape `(T,)`.
        - This increases the total number of time series in the dataset to
            by a factor of `self.target_dim`.
            - If `self.target_dim` is 1, the dataset remains unchanged.
        - `self.num_series` refers to the number of time series *after*
            conversion to univariate format.

        Args:
            name (str): Name of the dataset to load.

            term (Term | str): Forecast horizon term, which scales the base
                prediction length. Must be one of: "short", "medium", "long".

            to_univariate (bool): Whether to convert multivariate time series
                into multiple univariate ones. Defaults to True.

            storage_env_var (str): Environment variable pointing to the root
                directory of stored datasets.

            metadata_directory (str): Directory where metadata files are
                stored.

            verbose (bool): Whether to enable verbose output.

            limit (int, optional): Desired number of time series to use
                *after* converting the dataset to univariate format
                (if `to_univariate=True`). If specified for a multivariate
                dataset, it'll randomly sample approximately
                `limit // target_dim` multivariate series before applying the
                univariate transformation.
                - **Note:** The actual number of resulting univariate time
                    series may be slightly less than `limit` due to rounding
                    down during division by the number of target dimensions.

            fraction (float, optional): Fraction (0, 1] of the dataset to
                sample. If None, uses the entire dataset.

            seed (int, optional): Random seed used when sampling a subset of
                the dataset.
        """
        self.name = name
        self.term = Term(term)
        self.to_univariate = to_univariate
        self.storage_env_var = storage_env_var
        self.metadata_directory = metadata_directory
        self.verbose = verbose
        self.limit = limit
        self.fraction = fraction
        self.seed = seed

        if self.limit is not None and self.limit <= 0:
            raise ValueError(f"Limit must be a positive integer, got {self.limit}.")

        if self.fraction is not None and not (0 < self.fraction <= 1):
            raise ValueError(
                f"Fraction must be in the range (0, 1], got {self.fraction}."
            )

        if not self.verbose:
            disable_progress_bar()

        self.hf_dataset = datasets.load_from_disk(self.storage_path).with_format(
            "numpy"
        )

        # Select a random subset of series if `limit` or `fraction` are given.
        if self.limit is not None and self.limit < self._total_univariate_series:
            # Also assigns `self.num_series`
            self.hf_dataset = self._apply_limit()
        elif self.fraction is not None and self.fraction < 1:
            # Also assigns `self.num_series`
            self.hf_dataset = self._apply_fraction()
        else:
            # * Assumes multivariate datasets will be converted to univariate
            self.num_series = self._total_univariate_series

        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(
            compose(process, fix_1d_array, itemize_start),
            self.hf_dataset,
        )

        # For TEMPO, prefer converting multivariate datasets to univariate
        if self.to_univariate and self.target_dim > 1:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )

    def _apply_limit(self) -> HF_Dataset:
        """
        Applies the `limit` constraint to the Hugging Face dataset before
        `MultivariateToUnivariate` transformation.

        - This method computes the number of multivariate time series to sample
        based on the desired number of univariate series (`self.limit`) and the
        number of target dimensions (`self.target_dim`).
        - Then, it randomly samples that many series from the original dataset.
        - After converting to univariate, the resulting dataset will yield
        approximately `self.limit` univariate series.

        Returns:
            HF_Dataset: A Hugging Face dataset containing a subset of
            multivariate series that will yield approximately `self.limit`
            univariate series after transformation.
        """
        scaled_limit = max(self.limit // self.target_dim, 1)

        if scaled_limit >= len(self.hf_dataset):
            return self.hf_dataset

        if self.seed is not None:
            random.seed(self.seed)

        # ? Is this a safe assumption?
        self.num_series = scaled_limit * self.target_dim

        indices = random.sample(range(len(self.hf_dataset)), scaled_limit)
        return self.hf_dataset.select(indices)

    def _apply_fraction(self) -> HF_Dataset:
        """
        Apply the `fraction` constraint to the Hugging Face dataset.

        - This method samples a specified fraction of the multivariate time
        series in the original dataset, before applying the
        `MultivariateToUnivariate` transformation.
        - After the transformation, the number of univariate series will be
        approximately `fraction * total_univariate_series`.

        Returns:
            HF_Dataset: A subset of the Hugging Face dataset containing
            multivariate series that will yield the desired fraction of
            univariate series after transformation.
        """
        reduced_num_series = max(
            int(self.fraction * len(self.hf_dataset)),
            1,
        )

        if self.seed is not None:
            random.seed(self.seed)

        # ? Is this a safe assumption?
        self.num_series = reduced_num_series * self.target_dim

        indices = random.sample(range(len(self.hf_dataset)), reduced_num_series)
        return self.hf_dataset.select(indices)

    @cached_property
    def _total_univariate_series(self) -> int:
        """
        Returns the number of univariate time series in the dataset.If the
        dataset is multivariate, this is the number of dimensions multiplied by
        the number of multivariate series.
        """
        df = pd.read_csv(self.metadata_path)
        name_mask = df["name"] == self.name
        term_mask = df["term"] == self.term.value
        return df[name_mask & term_mask].iloc[0]["num_series"]

    @cached_property
    def metadata_path(self) -> Path:
        """
        Returns the path to the dataset's metadata file based on whether the
        dataset's in the pretraining or train-test split.
        """
        return Path(self.metadata_directory) / self.subdirectory / "metadata.csv"

    @cached_property
    def storage_path(self) -> str:
        """
        Returns the dataset's storage path based on whether it's in the
        pretraining or train-test split.
        """
        load_dotenv()
        directory = os.getenv(self.storage_env_var)
        return str(Path(directory) / self.subdirectory / self.name)

    @cached_property
    def subdirectory(
        self,
        file_name: str = "dataset_properties.json",
        directory: str = "notebooks",
    ) -> str:
        """
        Returns "pretrain" if the dataset is part of the pretraining split.
        Else, returns "train-test".

        Args:
            file_name (str, optional): Name of the JSON file containing dataset
                properties. Defaults to "dataset_properties.json".
            directory (str, optional): Parent directory of
                dataset_properties.json. Defaults to "notebooks".

        Returns:
            str: "pretrain" if the dataset belongs to the pretraining split,
                else "train_test".
        """
        dataset_properties = json.load(open(Path(directory) / file_name))
        return "train_test" if self.key in dataset_properties else "pretrain"

    @cached_property
    def key(self) -> str:
        """
        Returns the dataset's key for accessing dataset infomation in
        dataset_properties.json (e.g. domain and number of variates).
        """
        pretty_names = {
            "saugeenday": "saugeen",
            "temperature_rain_with_missing": "temperature_rain",
            "kdd_cup_2018_with_missing": "kdd_cup_2018",
            "car_parts_with_missing": "car_parts",
        }
        key = self.name.split("/")[0] if "/" in self.name else self.name
        key = key.lower()
        return pretty_names.get(key, key)

    @cached_property
    def config(self) -> str:
        """
        Returns the dataset's configuration formatted as `key`/`freq`/`term`.

        The dataset's configuration is used for formatting dataset names and
        terms in results files.

        Returns:
            str: The dataset's configuration.
        """
        return f"{self.key}/{self.freq}/{self.term.value}"

    @cached_property
    def seasonality(self) -> int:
        """
        Returns the dataset's seasonality (number of time steps per seasonal
        cycle) using the time series's frequency.

        Returns:
            int: The number of time steps in one seasonal cycle.
        """
        return get_seasonality(self.freq)

    @cached_property
    def prediction_length(self) -> int:
        base_freq = norm_freq_str(to_offset(self.freq).name)
        pred_len = (
            M4_PRED_LENGTH_MAP[base_freq]
            if "m4" in self.name
            else PRED_LENGTH_MAP[base_freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        freq = self.hf_dataset[0]["freq"]
        return freq if freq != "MS" else "M"  # Normalize "MS" to "M"

    @cached_property
    def target_dim(self) -> int:
        return (
            target.shape[0]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        if "past_feat_dynamic_real" not in self.hf_dataset[0]:
            return 0
        elif (
            len(
                (
                    past_feat_dynamic_real := self.hf_dataset[0][
                        "past_feat_dynamic_real"
                    ]
                ).shape
            )
            > 1
        ):
            return past_feat_dynamic_real.shape[0]
        else:
            return 1

    @cached_property
    def windows(self) -> int:
        if "m4" in self.name:
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(
                    pc.list_slice(self.hf_dataset.data.column("target"), 0, 1)
                )
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return min(lengths.to_numpy())

    @cached_property
    def sum_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(self.hf_dataset.data.column("target"))
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return sum(lengths.to_numpy())

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * (self.windows + 1),
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * self.windows,
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * self.windows,
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data
