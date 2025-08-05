from dataclasses import dataclass

import numpy as np
from gluonts.dataset import DataEntry

from ._base import Transformation


@dataclass
class ApplyOffset(Transformation):
    """
    Removes `offset` observations from the end of a time series.

    This transformation's based on GluonTS's `OffsetSplitter`. See here for the
    original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.split.html?highlight=offsetsplit#gluonts.dataset.split.OffsetSplitter
    """

    offset: int

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        data_entry["target"] = np.asarray(
            data_entry["target"][: self.offset], dtype=np.float32
        )
        return data_entry
