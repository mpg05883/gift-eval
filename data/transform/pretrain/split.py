from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import InstanceSampler
from gluonts.zebras._util import pad_axis

from ._base import Transformation


@dataclass
class SplitInstance(Transformation):
    """
    Reimplementation of GluonTS's `InstanceSplitter` that splits a time series
    into past and future segments, with the past segment containing the last
    `past_length` observations and the future segment containing the next
    `future_length` observations after a specified lead time. Adapted to work
    with uni2ts's `Transformation` interface.

    See here for the original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.transform.html?highlight=instancesplitter#gluonts.transform.InstanceSplitter
    """

    past_length: int
    future_length: int
    instance_sampler: InstanceSampler
    target_field: str = FieldName.TARGET
    is_pad_field: str = FieldName.IS_PAD
    start_field: str = FieldName.START
    forecast_start_field: str = FieldName.FORECAST_START
    lead_time: int = 0
    output_NTC: bool = False
    time_series_fields: List[str] = None
    dummy_value: float = 0.0
    max_retries: int = 100

    def _past(self, col_name: str) -> str:
        return f"past_{col_name}"

    def _future(self, col_name: str) -> str:
        return f"future_{col_name}"

    def _split_array(
        self,
        array: np.ndarray,
        idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if idx >= self.past_length:
            past_piece = array[..., idx - self.past_length : idx]
        else:
            past_piece = pad_axis(
                array[..., :idx],
                axis=-1,
                left=self.past_length - idx,
                value=self.dummy_value,
            )

        future_start = idx + self.lead_time
        future_slice = slice(future_start, future_start + self.future_length)
        future_piece = array[..., future_slice]
        return past_piece, future_piece

    def _split_instance(self, entry: DataEntry, idx: int) -> DataEntry:
        slice_cols = self.time_series_fields + [self.target_field]
        dtype = entry[self.target_field].dtype

        entry = entry.copy()

        for ts_field in slice_cols:
            past_piece, future_piece = self._split_array(entry[ts_field], idx)

            if self.output_NTC:
                past_piece = past_piece.transpose()
                future_piece = future_piece.transpose()

            entry[self._past(ts_field)] = past_piece
            entry[self._future(ts_field)] = future_piece
            del entry[ts_field]

        pad_indicator = np.zeros(self.past_length, dtype=dtype)
        pad_length = max(self.past_length - idx, 0)
        pad_indicator[:pad_length] = 1

        entry[self._past(self.is_pad_field)] = pad_indicator
        entry[self.forecast_start_field] = (
            entry[self.start_field] + idx + self.lead_time
        )

        return entry

    def __call__(self, entry: DataEntry) -> DataEntry:
        target = entry[self.target_field]

        # Loop until we get a valid index or reach the max number of retries
        for i in range(self.max_retries):
            if len(sampled_indices := self.instance_sampler(target)):
                break

        # If we didn't sample any indices, raise an exception
        if i == self.max_retries and not sampled_indices:
            raise Exception(
                "Reached maximum number of idle transformation"
                " calls.\nThis means the transformation looped over"
                f" {self.max_retries} inputs without returning any"
                " output.\nThis occurred in the following"
                f" transformation:\n{self}"
            )

        # Split the instance at the first sampled index
        idx = next(iter(sampled_indices))
        return self._split_instance(entry, idx)
