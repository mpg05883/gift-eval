from dataclasses import dataclass, field
from typing import Type

import numpy as np
import pandas as pd
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.exceptions import GluonTSDataError

from ._base import Transformation


@dataclass
class ProcessStartField(Transformation):
    """
    Transform the start field into a pandas Period with the given frequency.

    This transformation is based on GluonTS's `ProcessStartField` but was
    modified to be applied to each time series in a dataset instead of the
    entire dataset.

    See here for the original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.common.html?highlight=processstartfield#gluonts.dataset.common.ProcessStartField
    """

    freq: str | pd.DateOffset
    use_timestamp: bool = False
    name: str = FieldName.START

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        try:
            if self.use_timestamp:
                data_entry[self.name] = pd.Timestamp(data_entry[self.name])
            else:
                # Convert start to a native Python type first
                data_entry[self.name] = pd.Period(
                    data_entry[self.name].item(),
                    self.freq,
                )
        except (TypeError, ValueError) as e:
            raise GluonTSDataError(
                f'Error "{e}" occurred, when reading field "{self.name}"'
            ) from e

        return data_entry


@dataclass
class ProcessTimeSeriesField(Transformation):
    """
    Converts a time series field identified by name from a list of numbers into
    a numpy array.

    If is_required=True, throws a GluonTSDataError if the field is not present
    in the Data dictionary.

    If is_cat=True, the array type is np.int32, otherwise it is np.float32.

    If is_static=True, asserts that the resulting array is 1D, otherwise
    asserts that the resulting array is 2D. 2D dynamic arrays of shape (T) are
    automatically expanded to shape (1,T).

    This transformation is based on GluonTS's `ProcessTimeSeriesField` but was
    modified to be applied to each time series in a dataset instead of the
    entire dataset.

    See here for the original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.common.html#gluonts.dataset.common.ProcessTimeSeriesField
    """

    is_cat: bool
    is_static: bool
    name: str = FieldName.TARGET
    is_required: bool = True

    def __post_init__(self):
        self.req_ndim: int = 1 if self.is_static else 2
        self.dtype: Type = np.int32 if self.is_cat else np.float32

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        value = data_entry.get(self.name, None)
        if value is not None:
            value = np.asarray(value, dtype=self.dtype)

            if self.req_ndim != value.ndim:
                raise GluonTSDataError(
                    f"Array '{self.name}' has bad shape - expected "
                    f"{self.req_ndim} dimensions, got {value.ndim}. "
                )

            data_entry[self.name] = value

            return data_entry
        elif not self.is_required:
            return data_entry
        else:
            raise GluonTSDataError(f"Object is missing a required field `{self.name}`")


@dataclass
class ProcessDataEntry(Transformation):
    """
    Converts the start time to a pandas Period and the target value to a numpy
    array.

    This transformation was based on GluonTS's `ProcessDataEntry` but was
    modified to be applied to each time series in a dataset instead of the
    entire dataset.

    See here for the original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.common.html#gluonts.dataset.common.ProcessDataEntry
    """

    freq: str
    one_dim_target: bool = True
    use_timestamp: bool = False
    transformations: list = field(default_factory=list)

    def __post_init__(self):
        if not self.transformations:
            self.transformations = [
                ProcessStartField(
                    name=FieldName.START,
                    freq=self.freq,
                    use_timestamp=self.use_timestamp,
                ),
                ProcessTimeSeriesField(
                    name=FieldName.TARGET,
                    is_required=True,
                    is_cat=False,
                    is_static=self.one_dim_target,
                ),
            ]

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        for transform in self.transformations:
            data_entry = transform(data_entry)
        return data_entry


@dataclass
class MakeWritable(Transformation):
    """
    Makes all NumPy arrays in a data entry writable by copying before returning
    the data entry from the dataset's __getitem__ method.
    """

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        for key, value in data_entry.items():
            if isinstance(value, np.ndarray) and not value.flags.writeable:
                data_entry[key] = value.copy()
        return data_entry
