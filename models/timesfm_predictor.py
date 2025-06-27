from typing import List
import numpy as np
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
import timesfm

class TimesFmPredictor:

    def __init__(
        self,
        tfm,
        prediction_length: int,
        ds_freq: str,
        *args,
        **kwargs,
    ):
        self.tfm = tfm
        self.prediction_length = prediction_length
        if self.prediction_length > self.tfm.horizon_len:
            self.tfm.horizon_len = (
                (self.prediction_length + self.tfm.output_patch_len - 1)
                // self.tfm.output_patch_len
            ) * self.tfm.output_patch_len
            print("Jitting for new prediction length.")
        self.freq = timesfm.freq_map(ds_freq)

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            context = []
            for entry in batch:
                arr = np.array(entry["target"])
                context.append(arr)
            freqs = [self.freq] * len(context)
            _, full_preds = self.tfm.forecast(context, freqs, normalize=True)
            full_preds = full_preds[:, 0 : self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))
        forecast_outputs = np.concatenate(forecast_outputs)

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, self.tfm.quantiles)),
                    start_date=forecast_start_date,
                )
            )

        return forecasts
