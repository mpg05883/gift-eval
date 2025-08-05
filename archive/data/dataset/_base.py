import json
import os
import random
import warnings
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from dotted_dict import DottedDict
from gluonts.time_feature import get_seasonality, norm_freq_str
from pandas.tseries.frequencies import to_offset

from datasets import Dataset as HFDataset
from datasets import load_from_disk
from datasets.utils.logging import disable_progress_bar
from utils.common.enums import Domain, Term


class GiftEvalDataset:
    def __init__(
        self,
        name: str,
        term: Term | Literal["short", "medium", "long"] = Term.SHORT,
        to_univariate: bool = True,
        limit: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = 42,
        verbose: bool = False,
        test_split: float = 0.1,
        max_windows: int = 20,
        storage_env_var: str = "GIFT_EVAL",
        metadata_directory: str = "data",
        **kwargs,
    ):
        """
        Base class for loading GIFT-Eval datasets and their metadata.

        This class was based on GIFT-Eval's `Dataset` class, but was adapted to
        allow pretrain and train-test datasets to be processed differently. See
        here for the original implementation:
        https://github.com/SalesforceAIResearch/gift-eval/blob/main/src/gift_eval/data.py

        Args:
            name (str): Name of the dataset to load.
            term (Term | str): Forecast horizon term.
                - For pretrain datasets, the term's used to set the dataset's
                prediction length.
                - For train-test datasets, the term's used to optionally scale
                the dataset's original prediction length.
            to_univariate (bool): Whether to convert multivariate series to
                univariate. Defaults to True. NOTE: Prefer `to_univariate=True`
                for TEMPO.
            limit (int, optional): Desired number of *univariate* series to use
                after conversion from multivariate to univariate format.
                Defaults to None, which means the entire dataset is used.
            fraction (float, optional): Fraction (0, 1] of the total number of
                series to use. Defaults to None, which means the entire dataset
                is used.
            seed (int, optional): Random seed to use when subsampling. Defaults
                to 42.
            verbose (bool): Whether to display a progress bar when loading the
                dataset from disk. Defaults to False.
            storage_env_var (str): Environment variable pointing to the stored
                datasets' root directory.
            metadata_directory (str): Name of the root directory where metadata
                files are stored.
            **kwargs: Additional keyword arguments, if any.
        """
        self.name = name
        self.term = Term(term)
        self.to_univariate = to_univariate
        self.limit = limit
        self.fraction = fraction
        self.seed = seed
        self.verbose = verbose
        self.test_split = test_split
        self.max_windows = max_windows
        load_dotenv()
        self.dataset_directory = os.getenv(storage_env_var)
        self.metadata_directory = metadata_directory

        # Approximate number of (potentially multivariate) series before any
        # subsampling or conversion to univariate format
        max_limit = self._total_univariate_series // self.target_dim

        # Ignore limit if it's out of range
        if self.limit is not None and not (0 < self.limit <= max_limit):
            warnings.warn(
                f"Limit {self.limit} is not in range (0, {max_limit}]. "
                "Ignoring limit."
            )
            self.limit = None

        # Ignore fraction if it's out of range
        if self.fraction is not None and not (0 < self.fraction <= 1):
            warnings.warn(
                f"Fraction {self.fraction} is not in range (0, 1]. "
                "Ignoring fraction."
            )
            self.fraction = None

        if not self.verbose:
            disable_progress_bar()

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def storage_path(self) -> str:
        """
        Returns a path to where the dataset's stored on disk using the root
        directory specified by the storage enviornment variable.
        """
        return str(Path(self.dataset_directory) / self.gift_split / self.name)

    @cached_property
    def hf_dataset(self) -> HFDataset:
        """
        Loads the underlying Hugging Face from disk and performs subsampling if
        `self.limit` or `self.fraction` are specified.
        """
        dataset = load_from_disk(self.storage_path)
        if self.limit is not None or self.fraction is not None:
            dataset = self._get_subsample(dataset)
        return dataset

    @property
    def gift_split(self) -> str:
        """
        Returns "train_test" if the dataset is in the GIFT-Eval's train-test
        split. Else, retuns "pretrain".
        """
        file_path = Path(self.dataset_directory) / "train_test" / self.name
        return "train_test" if file_path.exists() else "pretrain"

    @cached_property
    def metadata(self) -> DottedDict:
        """
        Loads the dataset's metadata from a JSON file.

        Returns:
            DottedDict: A DottedDict containing the dataset's metadata. Allows
                key names to be accessed using dot notation.
        """
        dirpath = Path(self.metadata_directory) / "meta" / self.gift_split
        with open(dirpath / "metadata.json", "r") as f:
            metadata = json.load(f)
        return DottedDict(metadata[self.name])

    @property
    def domain(self) -> Domain:
        """
        Returns the dataset's domain.
        """
        return Domain(self.metadata.domain)

    @property
    def config(self) -> str:
        """
        Returns the dataset's configuration formatted as `name`/`freq`/`term`.
        This's is used for formatting dataset names and terms in results files.
        - `name` is the dataset's name in lowercase, with some additional
            formatting for specific datasets.
            - E.g. "saugeenday" is formatted as "saugeen".
        - `freq` is the dataset's frequency with the optional dash removed.
            - E.g. "W-SUN" is formatted as "W".
        - `term` is the dataset's term (short, medium, long).

        Returns:
            str: The dataset's configuration formatted as `name`/`freq`/`term`.
        """
        pretty_names = {
            "saugeenday": "saugeen",
            "temperature_rain_with_missing": "temperature_rain",
            "kdd_cup_2018_with_missing": "kdd_cup_2018",
            "car_parts_with_missing": "car_parts",
        }
        name = self.name.split("/")[0] if "/" in self.name else self.name
        cleaned_name = pretty_names.get(name.lower(), name.lower())
        cleaned_freq = self.freq.split("-")[0]
        return f"{cleaned_name}/{cleaned_freq}/{self.term}"

    @property
    def seasonality(self) -> int:
        """
        Computes the dataset's seasonality (number of time steps in one
        seasonal cycle). This's a thin wrapper around GluonTS's
        `get_seasonality`.

        Returns:
            int: The dataset's seasonality.
        """
        return get_seasonality(self.freq)

    @property
    def freq(self) -> str:
        """
        Returns the dataset's frequency.
        """
        return self.metadata.freq

    @property
    def base_freq(self) -> str:
        """
        Returns the dataset's base frequency, which is the frequency without
        any offsets (e.g., "5T" becomes "T", "W-SUN" becomes "W").
        """
        return norm_freq_str(to_offset(self.freq).name)

    @property
    def target_dim(self) -> int:
        """
        Returns the number of dimensions in the dataset's target. Assumes all
        series in dataset have the same target dimension.
        """
        return self.metadata.target_dim

    @property
    def sum_series_length(self) -> int:
        """
        Returns the total number of observations across all series in the
        dataset.
        """
        return self.metadata.sum_series_length

    @property
    def _min_series_length(self) -> int:
        """
        Returns the minimum series length across all series in the dataset.
        """
        return self.metadata._min_series_length

    @property
    def _total_univariate_series(self) -> int:
        """
        Returns the total number of univariate series in the dataset after
        conversion from multivariate to univariate format.

        **NOTE**: This is different from `num_series`, which returns the total
        number of (potentially multivariate) series before converting the
        dataset from multivariate to univariate format.
        """
        return self.metadata._total_univariate_series

    @property
    def context_length(self) -> int:
        """
        Returns the dataset's context length determined by the term.
        """
        return self.term.context_length

    @property
    @abstractmethod
    def prediction_length(self) -> int:
        """
        Returns the dataset's prediction length.
        - Pretrain datasets should use the prediction length specified by
        `self.term`.
        - Train-test datasets should use their original prediction length
        multiplied by `self.term`'s multiplier.
        """
        raise NotImplementedError("Subclasses must implement prediction_length.")

    @property
    @abstractmethod
    def windows(self) -> int:
        """
        Returns the number of windows to set aside for validation and testing.
        """
        raise NotImplementedError("Subclasses must implement windows.")

    @property
    def training_offset(self) -> int:
        """
        Returns the number of observations to exclude from the end of each
        series when creating the training split.
        """
        return self.prediction_length * (self.windows + 1)

    @property
    def val_test_offset(self) -> int:
        """
        Returns the number of observations to exclude from the end of each
        series when creating the validation split. For the test split, this's
        the number of observations to use at the end of each series.
        """
        return self.prediction_length * self.windows

    def _get_subsample(self, dataset: HFDataset) -> HFDataset:
        """
        Subsamples the underlying Hugging Face dataset to yield either
        ~`self.limit` or ~`self.fraction` univariate series after conversion to
        univariate format.

        If both `self.limit` and `self.fraction` are set, then `self.limit`
        takes precedence.

        Args:
            dataset (HFDataset): The Hugging Face dataset to subsample.

        Returns:
            HFDataset: A subset of the original dataset that'll yield
                either ~`self.limit` or ~`self.fraction` univariate series
                after conversion to univariate format.
        """
        # Number of multivariate series to sample to yield ~`self.limit` or
        # ~`self.fraction` unvariate series
        num_samples = (
            self.limit // self.target_dim
            if self.limit
            else int(self.fraction * len(dataset))
        )

        # Ensure at least one series is sampled
        num_samples = max(num_samples, 1)

        if self.seed is not None:
            random.seed(self.seed)

        indices = random.sample(range(len(dataset)), num_samples)
        return dataset.select(indices)
