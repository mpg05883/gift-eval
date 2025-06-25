import argparse
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, ForecastType
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.itertools import batcher
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm.auto import tqdm

from src.gift_eval.data import Dataset
from utils import format_elapsed_time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    quantile_levels: Optional[List[float]] = None
    forecast_keys: List[str] = field(init=False)
    statsforecast_keys: List[str] = field(init=False)
    intervals: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return

        intervals = set()

        for quantile_level in self.quantile_levels:
            interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")

        self.intervals = sorted(intervals)


class ChronosPredictor:
    def __init__(
        self,
        model_path,
        num_samples: int,
        prediction_length: int,
        *args,
        **kwargs,
    ):
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            *args,
            **kwargs,
        )
        logger.info(f"Device: {self.pipeline.model.device}")
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        pipeline = self.pipeline
        predict_kwargs = (
            {"num_samples": self.num_samples}
            if pipeline.forecast_type == ForecastType.SAMPLES
            else {}
        )
        while True:
            try:
                # Generate forecast samples
                forecast_outputs = []
                for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                    context = [torch.tensor(entry["target"]) for entry in batch]
                    forecast_outputs.append(
                        pipeline.predict(
                            context,
                            prediction_length=self.prediction_length,
                            **predict_kwargs,
                        ).numpy()
                    )
                forecast_outputs = np.concatenate(forecast_outputs)
                break
            except torch.cuda.OutOfMemoryError:
                logger.info(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])

            if pipeline.forecast_type == ForecastType.SAMPLES:
                forecasts.append(
                    SampleForecast(samples=item, start_date=forecast_start_date)
                )
            elif pipeline.forecast_type == ForecastType.QUANTILES:
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=item,
                        forecast_keys=list(map(str, pipeline.quantiles)),
                        start_date=forecast_start_date,
                    )
                )

        return forecasts


def main(args):
    df = pd.read_csv(Path("resources") / args.split_name / "metadata.csv")
    if args.term:
        df = df[df["term"] == args.term]
    df = df.sort_values(by="num_entries", ascending=True)
    name, term = df.iloc[args.index][["name", "term"]]

    logger.info(f"Loading dataset: {name} ({term})")
    dataset = Dataset(name=name, term=term, fraction=args.fraction)
    logger.info(f"Loaded {dataset.num_entries} entries")

    logger.info(f"Loading model: {args.model}")
    predictor = ChronosPredictor(
        model_path=f"amazon/{args.model.replace('_', '-')}",
        num_samples=20,
        prediction_length=dataset.prediction_length,
        device_map="auto",
    )

    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]

    dirpath = Path("results") / args.model / args.split_name / dataset.config
    dirpath.mkdir(parents=True, exist_ok=True)
    file_name = "results.csv"
    output_path = dirpath / file_name

    if not output_path.exists():
        with open(output_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "eval_metrics/MSE[mean]",
                    "eval_metrics/MSE[0.5]",
                    "eval_metrics/MAE[0.5]",
                    "eval_metrics/MASE[0.5]",
                    "eval_metrics/MAPE[0.5]",
                    "eval_metrics/sMAPE[0.5]",
                    "eval_metrics/MSIS",
                    "eval_metrics/RMSE[mean]",
                    "eval_metrics/NRMSE[mean]",
                    "eval_metrics/ND[0.5]",
                    "eval_metrics/mean_weighted_sum_quantile_loss",
                ]
            )

    logger.info("Starting evaluation...")
    start_time = time.time()
    res = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=metrics,
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=dataset.seasonality,
    )
    end_time = time.time()
    elapsed_time = format_elapsed_time(start_time, end_time)
    logger.info(f"Finished evaluation! Time taken: {elapsed_time}")

    logger.info(f"Saving results to {output_path}")
    with open(output_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                dataset.config,
                args.model,
                res["MSE[mean]"][0],
                res["MSE[0.5]"][0],
                res["MAE[0.5]"][0],
                res["MASE[0.5]"][0],
                res["MAPE[0.5]"][0],
                res["sMAPE[0.5]"][0],
                res["MSIS"][0],
                res["RMSE[mean]"][0],
                res["NRMSE[mean]"][0],
                res["ND[0.5]"][0],
                res["mean_weighted_sum_quantile_loss"][0],
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates a Chronos model on a specified dataset."
    )
    parser.add_argument(
        "--split_name",
        choices=["pretrain", "train_test"],
        default="pretrain",
        help="Name of the split the dataset belongs to.",
    )
    parser.add_argument(
        "--term",
        choices=["short", "medium", "long"],
        default="short",
        help="""Use this to only evaluate the model on datasets of a specific
        term.""",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="""Index of the dataset to load from the metadata CSV file. This
        specifies the name and term of the dataset to load.""",
    )
    parser.add_argument(
        "--model",
        choices=["chronos_base", "chronos_bolt_base", "chronos_bolt_small"],
        default="chronos_bolt_base",
        help="""Name of the model to evaluate.""",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="Percent of the dataset to use expressed as a decmial.",
    )
    args = parser.parse_args()
    main(args)
