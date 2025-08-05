import math
from functools import cached_property
from typing import Iterable, Iterator, Optional

from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.transform import Transformation
from toolz import compose

from utils.common.enums import Term

from .dataset import GiftEvalDataset

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


def itemize_start(data_entry: DataEntry) -> DataEntry:
    """
    Converts the `start` field into a native Python type.
    """
    data_entry[FieldName.START] = data_entry[FieldName.START].item()
    return data_entry


class MultivariateToUnivariate(Transformation):
    """
    Unpacks a single `D` dimensional multivariate time series into `D`
    separate univariate time series.
    """

    def __init__(self, field: str = FieldName.TARGET):
        self.field = field

    def __call__(
        self,
        dataset: Iterable[DataEntry],
        is_train: bool = False,
    ) -> Iterator:
        """
        Converts a multivariate dataset into univariate by unpacking each
        dimension into a separate entry.

        Args:
            dataset (Iterable[DataEntry]): The dataset to convert from
                multivariate to univariate format.
            is_train (bool, optional): Whether the transformation is being used
                during training (not used in this case). Defaults to False.
            NOTE: Keep `is_train=False` to maintain compatibility with GluonTS.


        Yields:
            Iterator: An iterator over the univariate entries, where each entry
                has the same fields as the original dataset, but with the
                target field containing only one dimension of the original
                multivariate target, and the item_id field modified to reflect
                the dimension.
        """
        for data_entry in dataset:
            item_id = data_entry[FieldName.ITEM_ID]
            multivariate_target = list(data_entry[self.field])
            for id, univariate_target in enumerate(multivariate_target):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = univariate_target
                univariate_entry[FieldName.ITEM_ID] = f"{item_id}_dim{id}"
                yield univariate_entry


class TrainTestDataset(GiftEvalDataset):
    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = True,
        storage_env_var: str = "GIFT_EVAL",
        metadata_directory: str = "data",
        verbose: bool = False,
        limit: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        """
        Wrapper around a GIFT-Eval dataset from the train-test split.

        This class was based on GIFT-Eval's `Dataset` class. See here for the
        original implementation:
        https://github.com/SalesforceAIResearch/gift-eval/blob/main/src/gift_eval/data.py

        Args:
            name (str): Name of the dataset to load.
            term (Term | str): Forecast horizon term. Used to optionally scale
                the dataset's prediction length.
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
        super().__init__(
            name=name,
            term=term,
            to_univariate=to_univariate,
            limit=limit,
            fraction=fraction,
            seed=seed,
            verbose=verbose,
            storage_env_var=storage_env_var,
            metadata_directory=metadata_directory,
        )

        # Set format to numpy before creating gluonts dataset
        self.hf_dataset.set_format("numpy")

        process_data_entry = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(
            compose(process_data_entry, itemize_start),
            self.hf_dataset,
        )

        if self.to_univariate and self.target_dim > 1:
            self.gluonts_dataset = MultivariateToUnivariate().apply(
                self.gluonts_dataset
            )

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    @cached_property
    def prediction_length(self) -> int:
        pred_len = (
            M4_PRED_LENGTH_MAP[self.base_freq]
            if "m4" in self.name
            else PRED_LENGTH_MAP[self.base_freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def windows(self) -> int:
        if "m4" in self.name:
            return 1
        w = math.ceil(
            self.test_split * self._min_series_length / self.prediction_length
        )
        return min(max(1, w), self.max_windows)

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.training_offset,
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.val_test_offset,
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.gluonts_dataset,
            offset=-self.val_test_offset,
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data
