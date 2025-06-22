import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.gift_eval.data import Dataset


def main(args):
    csv_path = Path("resources") / args.split / "info.csv"
    df = pd.read_csv(csv_path)
    names = df["name"].unique()

    kwargs = {
        "desc": f"Loading {args.split} datasets",
        "total": len(names),
        "unit": "dataset",
    }

    failed = []
    errors = []

    for name in tqdm(names, **kwargs):
        try:
            dataset = Dataset(name=args.name, verbose=False)
            training_dataset = dataset.training_dataset
            validation_dataset = dataset.validation_dataset
            test_data = dataset.test_data
        except Exception as e:
            failed.append(name)
            errors.append(str(e))

    print(f"Failed to load {len(failed)} datasets:")
    for name, error in zip(failed, errors):
        print(f"  {name}: {error}")


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
