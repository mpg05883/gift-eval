import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.gift_eval.data import Dataset


def get_key(name) -> str:
    """
    Returns the dataset's key for accessing dataset infomation in
    dataset_properties.json (e.g. domain and number of variates)

    Args:
        name (str): Name of the dataset.
    """
    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }
    key = name.split("/")[0] if "/" in name else name
    key = key.lower()
    return pretty_names.get(key, key)


def main(args):
    input_df = pd.read_csv(args.csv_path)
    dataset_properties = json.load(open(args.json_path))

    kwargs = {
        "desc": "Reading datasets",
        "total": len(input_df),
        "unit": "dataset",
    }

    rows = []
    for _, row in tqdm(input_df.iterrows(), **kwargs):
        name, term = row["name"], row["term"]
        dataset = Dataset(name, term)
        key = get_key(name)

        row = {
            "name": name,
            "term": term,
            "freq": dataset.freq,
            "domain": dataset_properties[key]["domain"],
            "num_series": dataset.num_series,
            "target_dim": dataset.target_dim,
            "_min_series_length": dataset._min_series_length,
            "sum_series_length": dataset.sum_series_length,
            "prediction_length": dataset.prediction_length,
            "windows": dataset.windows,
        }
        rows.append(row)

    output_df = pd.DataFrame(rows)
    output_df.to_csv(args.csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Saves information on all train-test datasets to a CSV file."""
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("resources") / "train_test" / "metadata.csv",
        help="Path to the CSV file where dataset information will be saved.",
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        default=Path("notebooks") / "dataset_properties.json",
        help="Path to the JSON file containing dataset information.",
    )
    args = parser.parse_args()
    main(args)
