from dataclasses import dataclass

import gluonts.time_feature
import numpy as np
import pandas as pd
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName

from ._base import Transformation


class BasicFeatures:
    """
    Implementation of TabPFN-TS's RunningIndexFeature and CalendarFeature
    classes.

    See here for the original implementation:
    https://github.com/PriorLabs/tabpfn-time-series/blob/main/tabpfn_time_series/features/basic_features.py
    """

    @staticmethod
    def add_running_index(df: pd.DataFrame) -> pd.Series:
        """
        Encodes each time step's index as a linearly spaced value between 0 and
        1.

        See the TabPFN-TS paper for more details:
        https://arxiv.org/abs/2501.02945
        """
        df["running_index"] = np.linspace(0.0, 1.0, num=len(df), dtype="float32")
        return df

    @staticmethod
    def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Uses each timestamp to encode the following cyclic calendar features:
        - Hour of the day
        - Day of the week
        - Day of the month
        - Day of the year
        - Week of the year
        - Month of the year
        - Year (as a linear feature scaled by 0.001)

        See the TabPFN-TS paper for more details:
        https://arxiv.org/abs/2501.02945
        """
        calendar_components = ["year"]

        # (feature, natural period)
        calendar_features = [
            ("hour_of_day", 24),
            ("day_of_week", 7),
            ("day_of_month", 30.5),
            ("day_of_year", 365),
            ("week_of_year", 52),
            ("month_of_year", 12),
        ]

        df.set_index("timestamp", inplace=True)
        timestamps = df.index.get_level_values("timestamp")

        for component_name in calendar_components:
            df[component_name] = getattr(timestamps, component_name)

        for feature_name, period in calendar_features:
            feature_function = getattr(gluonts.time_feature, f"{feature_name}_index")
            feature = feature_function(timestamps).astype(np.int32)

            if period is not None:
                period = period - 1  # Adjust for 0-based indexing
                df[f"{feature_name}_sin"] = np.sin(2 * np.pi * feature / period)
                df[f"{feature_name}_cos"] = np.cos(2 * np.pi * feature / period)
            else:
                df[feature_name] = feature

        df["year"] = df["year"] * 0.001
        return df


@dataclass
class AddCalendarFeatures(Transformation):
    """
    Uses timestamps to add cyclic calendar features and a running index as
    described in TabPFN-TS. Based on GluonTS's `AddTimeFeatures` transformation.

    See here for TabPFN-TS's calendar features and running index implementation:
    https://github.com/PriorLabs/tabpfn-time-series/blob/main/tabpfn_time_series/features/basic_features.py

    See here for GluonTS's `AddTimeFeatures` implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.transform.html?highlight=addtimefeature#gluonts.transform.AddTimeFeatures
    """

    prediction_length: int
    is_train: bool = False
    start_field: str = FieldName.START
    target_field: str = FieldName.TARGET
    output_field: str = FieldName.FEAT_TIME

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        start = data_entry[self.start_field]

        # Include forecast window if not training
        length = (
            len(data_entry[self.target_field])
            if self.is_train
            else len(data_entry[self.target_field]) + self.prediction_length
        )

        # Ensure `start` is a timestamp
        timestamp = (
            start.to_timestamp(freq=start.freq, how="start")
            if isinstance(start, pd.Period)
            else start
        )

        # Use seconds to get a larger time span
        index = pd.date_range(
            start=timestamp, periods=length, freq=start.freq, unit="s"
        )

        timestamps = np.array(index.values, dtype="datetime64[s]")
        timestamp_df = pd.DataFrame(timestamps, columns=["timestamp"])

        running_index = BasicFeatures.add_running_index(timestamp_df)
        calendar_features = BasicFeatures.add_calendar_features(timestamp_df)

        feature_df = pd.concat([running_index, calendar_features], axis=1)
        data_entry[self.output_field] = feature_df.values.T.astype("float32")
        return data_entry
