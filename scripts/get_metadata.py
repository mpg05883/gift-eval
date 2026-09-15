import argparse
from pathlib import Path


def main(args: argparse.Namespace):
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / args.split
    for dataset in data_dir.glob("*"):
        if dataset.is_dir() and dataset != ".cache":
            print(dataset.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="GiftEvalPretrain")
    args = parser.parse_args()
    main(args)
