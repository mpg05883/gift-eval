import argparse
from pathlib import Path
from src.gift_eval.data import Dataset

def main(args: argparse.Namespace):
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / args.split
    names = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name != ".cache"]
    names = sorted(names, key=lambda x: x.lower())
    datasets = [Dataset(name) for name in names]
    print(f"Number of datasets: {len(datasets)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="GiftEvalPretrain")
    args = parser.parse_args()
    main(args)
