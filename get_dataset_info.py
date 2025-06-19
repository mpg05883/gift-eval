import argparse
import json
from typing import Literal

import pandas as pd
from tqdm import tqdm
from pathlib import Path
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
    base_dir = Path('./datasets/pretrain')
    subdirs = [p.name for p in base_dir.iterdir() if p.is_dir()]
    print(len(subdirs), "datasets found in", base_dir)

    # print(subdirs)
    # df = pd.read_csv("train_test_datasets.csv")
    # dataset_properties = json.load(open("notebooks/dataset_properties.json"))

    # kwargs = {
    #     "desc": "Reading datasets",
    #     "total": len(df),
    #     "unit": "dataset",
    # }

    # rows = []
    # for i, row in tqdm(df.iterrows(), **kwargs):
    #     name, term = row["name"], row["term"]
    #     dataset = Dataset(name, term)
    #     key = get_key(name)
    #     row = {
    #         "name": name,
    #         "term": term,
    #         "freq": dataset.freq,
    #         "prediction_length": dataset.prediction_length,
    #         "target_dim": dataset.target_dim,
    #         "windows": dataset.windows,
    #         "_min_series_length": dataset._min_series_length,
    #         "sum_series_length": dataset.sum_series_length,
    #         "domain": dataset_properties[key]["domain"],
    #         "num_variates": dataset_properties[key]["num_variates"],
    #     }
    #     rows.append(row)

    # new_df = pd.DataFrame(rows)
    # new_df.to_csv("train_test_datasets.csv", index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Get dataset information and save it a CSV file"
    )
    argparser.add_argument(
        "--split",
        type=Literal["train_test", "pretrain"],
        default="train_test",
        help="Split to use (train_test or pretrain)",
    )
    args = argparser.parse_args()
    main(args)
