import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gift_eval.data import Dataset


def main(args: argparse.Namespace):
    short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    short_datasets = sorted(short_datasets.split(), key=lambda x: x.lower())
    med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    med_long_datasets = sorted(med_long_datasets.split(), key=lambda x: x.lower())

    root = Path(__file__).resolve().parents[1]
    metadata_dir = root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_path = metadata_dir / "gift_eval.csv"

    rows = []
    done_names = set()
    if output_path.exists():
        rows = pd.read_csv(output_path).to_dict("records")
        done_names = {row["name"] for row in rows}
        print(f"Resuming from {output_path} ({len(done_names)} datasets already done)")

    remaining_short_names = [name for name in short_datasets if name not in done_names]

    dataset_properties_map = json.load(
        open("notebooks/gift_eval/dataset_properties.json")
    )

    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }

    kwargs = {
        "desc": "Loading short datasets",
        "total": len(remaining_short_names),
        "unit": "dataset",
    }

    for i, name in enumerate(tqdm(remaining_short_names, **kwargs), start=1):
        dataset = Dataset(name)

        if "/" in name:
            ds_key = name.split("/")[0]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)

        rows.append(
            {
                "name": dataset.name,
                "term": dataset.term.value,
                "domain": dataset_properties_map[ds_key]["domain"],
                "freq": dataset.freq,
                "num_series": len(dataset.hf_dataset),
                "min_series_length": dataset._min_series_length,
                "sum_series_length": dataset.sum_series_length,
                "target_dim": dataset_properties_map[ds_key]["num_variates"],
                "past_feat_dynamic_real_dim": dataset.past_feat_dynamic_real_dim,
                "prediction_length": dataset.prediction_length,
                "validation_windows": dataset.validation_windows,
                "test_windows": dataset.test_windows,
            }
        )

        if name in med_long_datasets:
            dataset = Dataset(name, term="medium")
            rows.append(
                {
                    "name": dataset.name,
                    "term": dataset.term.value,
                    "domain": dataset_properties_map[ds_key]["domain"],
                    "freq": dataset.freq,
                    "num_series": len(dataset.hf_dataset),
                    "min_series_length": dataset._min_series_length,
                    "sum_series_length": dataset.sum_series_length,
                    "target_dim": dataset_properties_map[ds_key]["num_variates"],
                    "past_feat_dynamic_real_dim": dataset.past_feat_dynamic_real_dim,
                    "prediction_length": dataset.prediction_length,
                    "validation_windows": dataset.validation_windows,
                    "test_windows": dataset.test_windows,
                }
            )

            dataset = Dataset(name, term="long")
            rows.append(
                {
                    "name": dataset.name,
                    "term": dataset.term.value,
                    "domain": dataset_properties_map[ds_key]["domain"],
                    "freq": dataset.freq,
                    "num_series": len(dataset.hf_dataset),
                    "min_series_length": dataset._min_series_length,
                    "sum_series_length": dataset.sum_series_length,
                    "target_dim": dataset_properties_map[ds_key]["num_variates"],
                    "past_feat_dynamic_real_dim": dataset.past_feat_dynamic_real_dim,
                    "prediction_length": dataset.prediction_length,
                    "validation_windows": dataset.validation_windows,
                    "test_windows": dataset.test_windows,
                }
            )

        if args.save_every > 0 and i % args.save_every == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)
    print(f"Number of datasets: {len(df)}")

    df.to_csv(output_path, index=False)
    print(f"Saved {args.corpus} metadata to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=str,
        choices=["GiftEval"],
        default="GiftEval",
        help="The corpus to get metadata for",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save the metadata CSV to disk every N datasets processed",
    )
    args = parser.parse_args()
    main(args)
