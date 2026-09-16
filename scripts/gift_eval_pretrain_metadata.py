import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gift_eval.data import Dataset


def main(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "gift_eval_pretrain"
    names = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name != ".cache"]
    names = sorted(names, key=lambda x: x.lower())
    print(f"Number of names: {len(names)}")

    metadata_dir = root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_path = metadata_dir / "gift_eval_pretrain.csv"

    rows = []
    done_names = set()
    if output_path.exists():
        rows = pd.read_csv(output_path).to_dict("records")
        done_names = {row["name"] for row in rows}
        print(f"Found existing metadata for {len(done_names)}/{len(names)} datasets")

    missing_names = [n for n in names if n not in done_names]

    kwargs = {
        "desc": "Processing datasets",
        "total": len(missing_names),
        "unit": "dataset",
    }
    for i, name in enumerate(tqdm(missing_names, **kwargs), start=1):
        dataset = Dataset(name)
        rows.append(
            {
                "name": dataset.name,
                "domain": "",
                "freq": dataset.freq,
                "num_series": len(dataset.hf_dataset),
                "min_series_length": dataset._min_series_length,
                "sum_series_length": dataset.sum_series_length,
                "target_dim": dataset.target_dim,
                "past_feat_dynamic_real_dim": dataset.past_feat_dynamic_real_dim,
            }
        )

        if args.save_every > 0 and i % args.save_every == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)
    print(f"Number of datasets: {len(df)}")

    df.to_csv(output_path, index=False)
    print(f"Saved metadata to: {output_path}")


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
