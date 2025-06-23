import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.gift_eval.data import Dataset


def main(args):
    input_path = Path("resources") / args.split / "info.csv"
    df = pd.read_csv(input_path)

    names = df["name"].unique()
    terms = ["short", "medium", "long"]

    kwargs = {
        "desc": f"Loading {args.split} datasets",
        "total": len(names),
        "unit": "dataset",
    }

    failed = []
    errors = []
    lengths = []

    for name in tqdm(names, **kwargs):
        # Try loading each dataset
        try:
            dataset = Dataset(name=name, verbose=False)
            training_dataset = dataset.training_dataset
            validation_dataset = dataset.validation_dataset
            test_data = dataset.test_data
            num_entries = dataset.num_entries
        except Exception as e:
            failed.append(name)
            errors.append(str(e))
            num_entries = np.nan

        for _ in terms:
            lengths.append(num_entries)

    if not failed:
        print(f"All {len(names)} datasets loaded successfully.")
        return

    df["num_entries"] = lengths

    # Reorder columns
    df = df[
        [
            "name",
            "term",
            "freq",
            "domain",
            "num_entries",
            "target_dim",
            "windows",
            "_min_series_length",
            "sum_series_length",
            "prediction_length",
        ]
    ]
    df.to_csv(input_path, index=False)

    print(f"Successfully loaded {len(names) - len(failed)} datasets.")
    print(f"Failed to load {len(failed)} datasets:")
    for name, error in zip(failed, errors):
        print(f"  {name}: {error}")

    error_df = pd.DataFrame(
        {
            "name": failed,
            "error": errors,
        }
    )
    output_path = Path("outputs") / "load_split" / "load_errors.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error_df.to_csv(output_path, index=False)
    print(f"Errors saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Loads all the datasets in a specified split to ensure
        they can be loaded correctly."""
    )
    parser.add_argument(
        "--split",
        choices=["pretrain", "train_test"],
        default="pretrain",
        help="Specifies which split to load datasets from",
    )
    args = parser.parse_args()
    main(args)
