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


def main():
    input_path = Path("resources") / "train_test" / "info.csv"
    df = pd.read_csv(input_path)
    dataset_properties = json.load(open("notebooks/dataset_properties.json"))

    kwargs = {
        "desc": "Reading datasets",
        "total": len(df),
        "unit": "dataset",
    }

    rows = []
    for _, row in tqdm(df.iterrows(), **kwargs):
        name, term = row["name"], row["term"]
        dataset = Dataset(name, term)
        key = get_key(name)
        row = {
            "name": name,
            "term": term,
            "freq": dataset.freq,
            "prediction_length": dataset.prediction_length,
            "target_dim": dataset.target_dim,
            "windows": dataset.windows,
            "_min_series_length": dataset._min_series_length,
            "sum_series_length": dataset.sum_series_length,
            "domain": dataset_properties[key]["domain"],
            "num_variates": dataset_properties[key]["num_variates"],
            "num_entries": dataset.num_entries,
        }
        rows.append(row)

    new_df = pd.DataFrame(rows)
    new_df.to_csv("train_test_datasets.csv", index=False)


if __name__ == "__main__":
    main()
