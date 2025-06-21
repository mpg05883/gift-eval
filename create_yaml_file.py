import argparse
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def main(args):
    df = pd.read_csv(Path("resources") / args.split / "info.csv")

    kwargs = {
        "desc": f"Creating {args.split} yaml file",
        "unit": "dataset",
        "total": len(df),
    }

    datasets = []
    for _, row in tqdm(df.iterrows(), **kwargs):
        mapping = {
            "_target_": "gift_eval.data.Dataset",
            "name": row["name"],
            "term": row["term"],
        }
        datasets.append(mapping)

    data = {
        "name": "train_test",
        "datasets": datasets,
    }

    yaml_dirpath = Path("cli") / "conf" / "analysis" / "datasets"
    yaml_dirpath.mkdir(parents=True, exist_ok=True)
    yaml_path = yaml_dirpath / f"{args.split}.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"YAML file saved to: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a yaml file for computing time series features"
    )
    parser.add_argument(
        "--split",
        choices=["train_test", "pretrain"],
        default="pretrain",
        help="Split to use (train_test or pretrain)",
    )
    args = parser.parse_args()
    main(args)
