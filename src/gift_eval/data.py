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
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator

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
from datasets.utils.logging import disable_progress_bar

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
    "YE": 6,
    "QE": 8,
    "h": 48,
    "m": 12,
    "s": 60,
    "w": 8,
    "d": 30,
    "q": 8,
    "y": 6,
    "ME": 12,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
    "min": 12,
    "QE": 8,
    "h": 48,
    "m": 12,
    "s": 60,
    "w": 8,
    "d": 30,
    "q": 8,
    "Q": 8,
    "y": 6,
    "A": 6,
    "ME": 12,
}

TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "H": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "W": 13,
    "U": 8,
    "T": 8,
    "h": 48,
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


class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False
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
        storage_env_var: str = "GIFT_EVAL",
        verbose: bool = True,
    ):
        self.name = name
        self.term = Term(term)

        if not verbose:
            disable_progress_bar()

        # Change storage path depending on whether dataset is in pretrain or
        # train-test split
        load_dotenv()
        directory = os.getenv(storage_env_var)
        storage_path = str(Path(directory) / self.subdirectory / self.name)

        self.hf_dataset = datasets.load_from_disk(storage_path).with_format("numpy")

        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)

        if self.target_dim > 1:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )
            
    @cached_property
    def num_entries(self) -> str:
        """
        Returns the number of time series entires in the dataset.
        """
        return len(self.training_dataset)

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
    def subdirectory(
        self,
        file_name: str = "dataset_properties.json",
        directory: str = "notebooks",
    ) -> str:
        """
        Determines whether the storage path's subdirectory is set to "pretrain"
        or "train_test".

        `subdirectory` is set to "pretrain" if the dataset is part of the
        pretraining split. Else, `subdirectory` is set to "train-test".

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

    @property
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
        freq = norm_freq_str(to_offset(self.freq).name)
        pred_len = (
            M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        return self.hf_dataset[0]["freq"]

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
            self.gluonts_dataset, offset=-self.prediction_length * (self.windows + 1)
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(
            self.gluonts_dataset, offset=-self.prediction_length * self.windows
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.gluonts_dataset, offset=-self.prediction_length * self.windows
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data
