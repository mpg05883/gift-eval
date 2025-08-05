from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.gift_eval.data import Dataset


def main():
    input_path = Path("resources") / "pretrain" / "metadata.csv"
    df = pd.read_csv(input_path)

    kwargs = {
        "desc": "Reading datasets",
        "total": len(df) / 3,
        "unit": "dataset",
    }

    errors = {}

    for i, row in tqdm(df.iloc[::3].iterrows(), **kwargs):
        try:
            dataset = Dataset(row["name"])
            freq = dataset.freq
            num_series = dataset.num_series
            target_dim = dataset.target_dim
            _min_series_length = dataset._min_series_length
            sum_series_length = dataset.sum_series_length
            prediction_length = dataset.prediction_length
            windows = dataset.windows

            df.at[i, "freq"] = freq
            df.at[i, "num_series"] = num_series
            df.at[i, "target_dim"] = target_dim
            df.at[i, "_min_series_length"] = _min_series_length
            df.at[i, "sum_series_length"] = sum_series_length
            df.at[i, "prediction_length"] = prediction_length
            df.at[i, "windows"] = windows
        except Exception as e:
            print(f"Error reading dataset {row['name']}: {e}")
            errors[row["name"]] = str(e)
            continue

        for j in range(1, 3):
            df.at[i + j, "freq"] = freq
            df.at[i + j, "num_series"] = num_series
            df.at[i + j, "target_dim"] = target_dim
            df.at[i + j, "_min_series_length"] = _min_series_length
            df.at[i + j, "sum_series_length"] = sum_series_length
            multiplier = 10 if j == 1 else 15
            df.at[i + j, "prediction_length"] = prediction_length * multiplier
            df.at[i + j, "windows"] = windows

    # Reorder columns
    df = df[
        [
            "name",
            "term",
            "freq",
            "domain",
            "num_series",
            "target_dim",
            "_min_series_length",
            "sum_series_length",
            "prediction_length",
            "windows",
        ]
    ]
    df.to_csv(input_path, index=False)


if __name__ == "__main__":

    main()
