import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.gift_eval.data import Dataset


def main(args):
    input_path = Path("resources") / "pretrain" / "info.csv"
    df = pd.read_csv(input_path)
    names = df["name"].unique()

    terms = ["short", "medium", "long"]

    kwargs = {
        "desc": "Reading datasets",
        "total": len(names),
        "unit": "dataset",
    }

    print(f"Number of names: {len(names)}")
    print(f"Number of terms: {len(terms)}")
    print(f"Number of name-term combinations: {len(names) * len(terms)}")

    lengths = []

    for name in tqdm(names, **kwargs):
        dataset = Dataset(name)
        num_entries = dataset.num_entries
        for _ in terms:
            lengths.append(num_entries)

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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Gets dataset information and saves it to a CSV file"
    )
    argparser.add_argument(
        "--split",
        choices=["train_test", "pretrain"],
        default="pretrain",
        help="Split to use (train_test or pretrain)",
    )
    args = argparser.parse_args()
    main(args)
