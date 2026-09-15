import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gift_eval.data import Dataset


def main(args: argparse.Namespace):
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / args.corpus
    names = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name != ".cache"]
    names = sorted(names, key=lambda x: x.lower())

    kwargs = {
        "desc": "Loading datasets",
        "total": len(names),
        "unit": "dataset",
    }
    rows = []
    for name in tqdm(names, **kwargs):
        dataset = Dataset(name)
        rows.append(
            {
                "name": dataset.name,
                "freq": dataset.freq,
                "target_dim": dataset.target_dim,
                "past_feat_dynamic_real_dim": dataset.past_feat_dynamic_real_dim,
                "num_series": len(dataset.hf_dataset),
                "min_series_length": dataset._min_series_length,
                "sum_series_length": dataset.sum_series_length,
            }
        )

    df = pd.DataFrame(rows)
    print(f"Number of datasets: {len(df)}")

    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_path = outputs_dir / f"{args.corpus}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {args.corpus} metadata to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="GiftEvalPretrain")
    args = parser.parse_args()
    main(args)
