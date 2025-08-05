from functools import cached_property
from typing import Literal, Optional

import numpy as np
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
)
from torch.utils.data import Dataset as TorchDataset

from utils.common.enums import Term

from .dataset import GiftEvalDataset
from .indexer.hf_indexer import HFDatasetIndexer
from .transform.pretrain import Transformation
from .transform.pretrain.calendar import AddCalendarFeatures
from .transform.pretrain.field import RemoveFields, SelectFields
from .transform.pretrain.imputation import (
    DummyValueImputation,
    ImputationMethod,
    ImputeTimeSeries,
    LastValueImputation,
)
from .transform.pretrain.observed import AddObservedValuesIndicator
from .transform.pretrain.offset import ApplyOffset
from .transform.pretrain.pad import Pad
from .transform.pretrain.process import MakeWritable, ProcessDataEntry
from .transform.pretrain.reshape import Transpose
from .transform.pretrain.split import SplitInstance


class PretrainDataset(GiftEvalDataset, TorchDataset):
    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = True,
        limit: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = 42,
        verbose: bool = False,
        storage_env_var: str = "GIFT_EVAL",
        metadata_directory: str = "data",
        mode: Literal["training", "validation"] = "training",
        imputation_strategy: Literal["last_value", "dummy"] = "last_value",
        random_sampling: Optional[Literal["uniform", "proportional"]] = None,
        sampling_multiplier: float = 1.0,
        time_feat: bool = False,
        **kwargs,
    ):
        """
        Wrapper around a GIFT-Eval pretraining dataset loaded as a Hugging Face
        dataset. Uses a `HFDatasetIndexer` for fast access to the underlying
        data and lazily applies GluonTS-style transformations to each time
        series.

        This class was based on uni2ts's `TimeSeriesDataset` class. See here
        for the original implementation:
        https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/data/dataset.py

        Args:
            name (str): Name of the dataset to load.
            term (Term | str): Forecast horizon term. Used for setting the
                context and prediction lengths.
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
            mode (Literal["training", "validation"]): Whether the dataset's
                used for training or validation. Determines the number of
                observations to exclude from the end of each series. Defaults
                to "training".
            imputation_strategy (Literal["last_value", "dummy"]): Strategy for
                imputing missing values in each series. Options are:
                - "last_value": Use the last observed value to fill missing
                values.
                - "dummy": Use a dummy value (0.0) to fill missing values.
            random_sampling (Literal[, "uniform", "proportional"], optional):
                Specifies if and how series are randomly sampled from the
                dataset. Options are:
                - None: No random sampling, directly uses index to sample each
                series.
                - "uniform": Randomly samples each series with uniform
                probability.
                - "proportional": Randomly samples each series with a
                probability proportional to its length.
                Defaults to None, which means no random sampling is applied.
            sampling_multiplier (float): Scaling factor used in __len__ to
                "trick" PyTorch's data loader into sampling datasets based on
                the number of observations they have instead of number of
                series.
            time_feat (bool): Whether to add calendar features to the time
                series. Defaults to False.
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
        self.mode = mode
        self.imputation_strategy = imputation_strategy
        self.random_sampling = random_sampling
        self.sampling_multiplier = sampling_multiplier
        self.time_feat = time_feat

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def prediction_length(self) -> int:
        """
        Returns the prediction length to use based on the term. Assumes that
        multiple datasets will be combined during pretraining and they'll all
        use the same prediction length.
        """
        return self.term.prediction_length

    @property
    def windows(self) -> int:
        """
        Returns the number of windows to set aside for validation.
        - 2 windows will be set aside during training
        - 1 window will be set aside during validation
        """
        return 1

    @property
    def is_train(self) -> bool:
        """
        Returns whether the dataset is being used for training.
        """
        return self.mode == "training"

    @property
    def offset(self) -> int:
        return self.training_offset if self.is_train else self.val_test_offset

    @cached_property
    def apply_offset(self) -> ApplyOffset:
        return ApplyOffset(offset=-self.offset)

    @cached_property
    def imputation_method(self) -> ImputationMethod:
        """
        Returns the imputation method to use in the `ImputeTimeSeries`
        transformation.
        """
        return {
            "last_value": LastValueImputation(),
            "dummy": DummyValueImputation(value=0.0),
        }[self.imputation_strategy]

    @cached_property
    def indexer(self) -> HFDatasetIndexer:
        """
        Returns a Hugging Face dataset indexer for fast access to the
        underlying data.
        """
        return HFDatasetIndexer(self.hf_dataset, self.metadata)

    @cached_property
    def probabilities(self) -> np.ndarray:
        """
        Returns series-level probabilities for optional random sampling.
        """
        return {
            "uniform": self.indexer.get_uniform_probabilities(),
            "proportional": self.indexer.get_proportional_probabilities(),
        }.get(self.random_sampling, None)

    @cached_property
    def process_data_entry(self) -> ProcessDataEntry:
        """
        Returns a `ProcessDataEntry` transformation that converts the start
        field into a pandas Period and the target field into a numpy array.
        """
        return ProcessDataEntry(freq=self.freq, one_dim_target=True)

    @cached_property
    def remove_fields(self) -> RemoveFields:
        """
        Returns a transformation that removes the past dynamic real features
        from the data entry.
        """
        return RemoveFields(fields=[FieldName.PAST_FEAT_DYNAMIC_REAL])

    @cached_property
    def pad(self) -> Pad:
        """
        Returns a transformation that pads the target field to a minimum
        length.
        """
        return Pad(
            fields=[FieldName.TARGET],
            min_length=self.context_length + self.prediction_length,
        )

    @cached_property
    def add_observed_values_indicator(self) -> AddObservedValuesIndicator:
        """
        Returns a transformation that adds a field indicating which values
        were observed vs missing in the target field.
        """
        return AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

    @cached_property
    def impute_time_series(self) -> ImputeTimeSeries:
        """
        Returns a transformation that imputes missing values in the target
        field using the specified imputation method.
        """
        return ImputeTimeSeries(
            fields=[FieldName.TARGET],
            imputation_method=self.imputation_method,
        )

    @cached_property
    def add_calendar_features(self) -> AddCalendarFeatures:
        """
        Returns a transformation that adds calendar features to the series if
        `time_feat` is True.
        """
        return AddCalendarFeatures(
            prediction_length=self.term.prediction_length,
            is_train=self.is_train,
            start_field=FieldName.START,
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_TIME,
        )

    @cached_property
    def split_instance(self) -> SplitInstance:
        """
        Returns a transformation that splits the series into past and future
        instances based on the `context_length` and `prediction_length`.
        """
        if self.is_train:
            instance_sampler = ExpectedNumInstanceSampler(
                num_instances=1,
                min_future=self.prediction_length,
            )
        else:
            instance_sampler = ValidationSplitSampler(
                min_future=self.prediction_length,
            )

        time_series_fields = (
            [FieldName.OBSERVED_VALUES, FieldName.FEAT_TIME]
            if self.time_feat
            else [FieldName.OBSERVED_VALUES]
        )

        return SplitInstance(
            past_length=self.context_length,
            future_length=self.prediction_length,
            instance_sampler=instance_sampler,
            target_field=FieldName.TARGET,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            time_series_fields=time_series_fields,
        )

    @cached_property
    def select_fields(self) -> SelectFields:
        """
        Returns a transformation that selects the fields to keep in the final
        data entry.
        """
        fields = [
            f"past_{FieldName.TARGET}",
            f"future_{FieldName.TARGET}",
            f"past_{FieldName.OBSERVED_VALUES}",
            f"future_{FieldName.OBSERVED_VALUES}",
        ]

        if self.time_feat:
            fields += [
                f"past_{FieldName.FEAT_TIME}",
                f"future_{FieldName.FEAT_TIME}",
            ]

        return SelectFields(fields=fields)

    @cached_property
    def transpose(self) -> Transpose:
        """
        Returns a transformation that transposes the time feature fields to
        have shape (T, D).
        """
        return Transpose(
            fields=[
                f"past_{FieldName.FEAT_TIME}",
                f"future_{FieldName.FEAT_TIME}",
            ],
            axes=(1, 0),
        )

    @cached_property
    def make_writable(self) -> MakeWritable:
        return MakeWritable()

    @cached_property
    def transform(self) -> Transformation:
        """
        Returns a transformation pipeline that applies the following
        transformations in order:
        - `ProcessDataEntry`: Converts the start field into a pandas Periord
        and the target field into a numpy array.
        - `RemoveFields`: Removes the past dynamic real features from the
        data entry.
        - `Pad`: Pads the target field to a minimum length.
        - `RandomInstanceSlice`: Extracts a random slice of the time series that's
        `context_length + prediction_length` time steps long.
        - `AddObservedValuesIndicator`: Adds a field indicating which values
        were observed vs missing in the target field.
        - `ImputeTimeSeries`: Imputes missing values in the target field using
        the specified imputation method.
        - `AddCalendarFeatures`: Adds calendar features to the series if
        `time_feat` is True.
        - `SplitInstance`: Splits the series into past and future.
        instances based on the `context_length` and `prediction_length`.
        """
        # ? Consider moving this outside of the class and passing it in as
        # ? an argument to the constructor?
        transform = self.process_data_entry + self.remove_fields + self.apply_offset

        transform += (
            self.pad + self.add_observed_values_indicator + self.impute_time_series
        )

        if self.time_feat:
            transform += self.add_calendar_features

        return (
            transform
            + self.split_instance
            + self.select_fields
            + self.transpose
            + self.make_writable
        )

    def _multivariate_to_univariate(
        self,
        data_entry: DataEntry,
        series_idx: int,
    ) -> DataEntry:
        """
        Returns a new data entry that contains a single dimension extracted
        from a multivariate time series.

        Args:
            data_entry (DataEntry): A single time series from the dataset. Must
                be multivariate with shape (D, T).
            series_idx (int): The series-level index used when iterating over
                the dataset. This is used to determine which dimension to
                extract from the multivariate time series.

        Returns:
            DataEntry: A univariate time series with the same fields as the
            original `data_entry`, but with the "target" field only containing
            one of the dimensions and an updated "item_id".
        """
        # Specify which dimension to extract from the multivariate series
        dim_idx = series_idx % self.target_dim

        # Copy all fields except "target" and "item_id"
        univariate_entry = {
            key: value
            for key, value in data_entry.items()
            if key != FieldName.TARGET and key != FieldName.ITEM_ID
        }

        # Handle "target" and "item_id" fields
        univariate_entry[FieldName.ITEM_ID] = (
            f"{data_entry[FieldName.ITEM_ID]}_dim{dim_idx}"
        )
        univariate_entry[FieldName.TARGET] = data_entry[FieldName.TARGET][dim_idx]
        return univariate_entry

    def __len__(self) -> int:
        """
        Returns the number of series in the dataset scaled by the sampling
        multiplier. This is used to "trick" PyTorch's data loader into sampling
        datasets based on the number of observations they have instead of
        number of series.
        """
        return int(np.ceil(self.sampling_multiplier * len(self.hf_dataset)))

    def __getitem__(self, idx: int) -> DataEntry:
        """
        Samples a time series from the dataset based on the specified sampling
        strategy and lazily applies a transformation to it. Also converts
        multivariate series to univariate before applying the transformation if
        `to_univariate` is True.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(
                f"Index {idx} out of range for {self.name}, which has a "
                f"length of {len(self)}"
            )

        if self.random_sampling is not None:
            idx = np.random.choice(len(self.hf_dataset), p=self.probabilities)

        # Specify which series to sample from the dataset
        series_idx = idx // self.target_dim
        item = self.indexer[series_idx % len(self.hf_dataset)]

        if self.to_univariate and self.target_dim > 1:
            item = self._multivariate_to_univariate(item, idx)

        # Split into past and future windows + remove non-input fields
        return self.transform(item)
