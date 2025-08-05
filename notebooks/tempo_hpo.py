from typing import Iterable

import pandas as pd
from gluonts.dataset.common import DataEntry
from gluonts.dataset.split import split
from gluonts.evaluation import Evaluator
from gluonts.torch.model.deepar import DeepAREstimator


def dataentry_to_dataframe(entry: DataEntry) -> pd.DataFrame:
    """
    Converts a GluonTS DataEntry to a pandas DataFrame.

    Args:
        entry (DataEntry): A dictionary representing a time series. Must
            contain a `start` field and a `target` field.


    Returns:
        pd.DataFrame: The input time series represented as a pandas DataFrame.
    """
    return pd.DataFrame(
        entry["target"],
        columns=[entry.get("item_id")],
        index=pd.period_range(
            start=entry["start"],
            periods=len(entry["target"]),
            freq=entry["start"].freq,
        ),
    )


class TEMPOTuningObjective:
    def __init__(
        self,
        dataset: Iterable[DataEntry],
        prediction_length: int,
        freq: str,
        metric_type: str = "mean_wQuantileLoss",  # Change this to mean wQuantile Loss
    ):
        self.dataset = dataset
        self.prediction_length = prediction_length
        self.freq = freq
        self.metric_type = metric_type
        self.train, test_template = split(dataset, offset=-self.prediction_length)

        # Include model config

        validation = test_template.generate_instances(
            prediction_length=prediction_length
        )
        self.validation_input = [entry[0] for entry in validation]
        self.validation_label = [
            dataentry_to_dataframe(entry[1]) for entry in validation
        ]

    def get_params(self, trial) -> dict:
        """
        Get the parameters for the DeepAR model based on the trial.
        """
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 5),
            "hidden_size": trial.suggest_int("hidden_size", 10, 50),
        }

    def __call__(self, trial):
        params = self.get_params(trial)

        # Initialize new DeepAR estimator with the suggested parameters
        estimator = DeepAREstimator(
            num_layers=params["num_layers"],
            hidden_size=params["hidden_size"],
            prediction_length=self.prediction_length,
            freq=self.freq,
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "max_epochs": 10,
            },
        )

        # Train estimator
        predictor = estimator.train(self.train, cache_data=True)

        # Evaluate model

        # Return CRPS

        # Generate forecasts
        forecasts = list(predictor.predict(self.validation_input))

        # Create new evaluator
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

        # Aggregate metrics across all time series
        agg_metrics, item_metrics = evaluator(
            self.validation_label,
            forecasts,
            num_series=len(self.dataset),
        )

        return agg_metrics[self.metric_type]
