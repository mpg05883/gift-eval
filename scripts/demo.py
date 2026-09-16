import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gift_eval.data import Dataset


def main(args: argparse.Namespace) -> None:
    names = ["subseasonal", "subseasonal_precip"]
    # names = [
    #     "gfc14_load"
    # ]
    
    print(f"Names: {', '.join(names)}")
    
    for name in names:
        dataset = Dataset(name)
        print(f"Name: {name}")
        print(f"  # Time Series: {len(dataset.hf_dataset)}")
        print(f"  # Targets: {dataset.target_dim}")
        print(f"  # Covariates: {dataset.past_feat_dynamic_real_dim}")
        print(f"  # Obs.: {dataset.sum_series_length}")
        print()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save the metadata CSV after processing every N datasets",
    )
    args = parser.parse_args()
    main(args)
