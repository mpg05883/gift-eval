from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def main():
    csv_path = Path("resources") / "train_test_datasets.csv"
    df = pd.read_csv(csv_path)

    kwargs = {
        "desc": "Writing datasets to yaml file",
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

    yaml_path = Path("cli") / "conf" / "analysis" / "datasets" / "train_test.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"YAML file saved to: {yaml_path}")


if __name__ == "__main__":
    main()
