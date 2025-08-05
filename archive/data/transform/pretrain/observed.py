from dataclasses import dataclass
from typing import Type

import numpy as np
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName

from ._base import Transformation


@dataclass
class AddObservedValuesIndicator(Transformation):
    """
    Replaces missing values in a numpy array (NaNs) with a dummy value and adds
    an "observed"-indicator that is `1` when values are observed and `0` when
    values are missing.

    This transformation's based on GluonTS's `ObservedValuesIndicator`, but was
    modified to be applied to each time series in a dataset instead of the
    entire dataset.

    See here for the original implementation:
    https://ts.gluon.ai/stable/_modules/gluonts/transform/feature.html#AddObservedValuesIndicator
    """

    target_field: str = FieldName.TARGET
    output_field: str = FieldName.OBSERVED_VALUES
    dtype: Type = np.float32

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        value = data_entry[self.target_field]
        nan_entries = np.isnan(value)

        data_entry[self.output_field] = np.invert(
            nan_entries,
            out=nan_entries,
        ).astype(self.dtype, copy=False)
        return data_entry
