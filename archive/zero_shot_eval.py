import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import timesfm
from gluonts.ev.metrics import (
    MAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model

from models.chronos_predictor import ChronosPredictor
from models.timesfm_predictor import TimesFmPredictor
from src.gift_eval.data import Dataset
from utils import format_elapsed_time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
)

logger = logging.getLogger(__name__)


def main(args):
    df = pd.read_csv(Path("resources") / args.split_name / "metadata.csv")
    if args.term:
        df = df[df["term"] == args.term]
    df = df.sort_values(by="num_series", ascending=True)
    name, term, num_series = df.iloc[args.index][["name", "term", "num_series"]]

    fraction = args.fraction if num_series > args.threshold else 1.0

    logger.info(f"Loading dataset: {name} ({term})")
    dataset = Dataset(name=name, term=term, fraction=fraction)
    logger.info(f"Number of series: {dataset.num_series} series")

    logger.info(f"Loading model: {args.model_name}")
    if "chronos" in args.model_name:
        predictor = ChronosPredictor(
            model_path=f"amazon/{args.model_name.replace('_', '-')}",
            num_samples=20,
            prediction_length=dataset.prediction_length,
            device_map="auto",
        )
    else:
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                num_layers=50,
                horizon_len=128,
                context_len=2048,
                use_positional_embedding=False,
                output_patch_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

        predictor = TimesFmPredictor(
            tfm=tfm,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
        )

    metrics = [
        MAPE(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]

    dirpath = Path("results") / args.model_name / args.split_name / dataset.config
    dirpath.mkdir(parents=True, exist_ok=True)
    file_name = "results.csv"
    output_path = dirpath / file_name

    # Exit if the results file already exists
    if output_path.exists():
        logger.info(f"Results file already exists: {output_path}")
        sys.exit()

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "dataset",
                "model",
                "eval_metrics/MAPE[0.5]",
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
                args.model_name,
                res["MAPE[0.5]"][0],
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
        "--model_name",
        choices=[
            "chronos_base",
            "chronos_bolt_base",
            "chronos_bolt_small",
            "timesfm_2_0_500m",
        ],
        default="timesfm_2_0_500m",
        help="""Name of the model to evaluate.""",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10000,
        help="""Sample a fraction of the dataset if it has more than this 
        number of series.""",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="Percent of the dataset to use expressed as a decmial.",
    )
    args = parser.parse_args()
    main(args)
