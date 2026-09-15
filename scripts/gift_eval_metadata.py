import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gift_eval.data import Dataset


def get_short_datasets() -> list[str]:
    names = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    return sorted(names.split(), key=lambda x: x.lower())


def get_med_long_datasets() -> list[str]:
    names = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    return sorted(names.split(), key=lambda x: x.lower())


def get_dataset_properties_map() -> dict:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "gift_eval" / "dataset_properties.json"
    with path.open() as f:
        return json.load(f)


def get_ds_key(name: str) -> str:
    pretty_datasets = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }

    if "/" in name:
        ds_key = name.split("/")[0]
        ds_key = ds_key.lower()
        ds_key = pretty_datasets.get(ds_key, ds_key)
    else:
        ds_key = name.lower()
        ds_key = pretty_datasets.get(ds_key, ds_key)
    return ds_key


def main() -> None:
    short_datasets = get_short_datasets()
    med_long_datasets = get_med_long_datasets()
    dataset_properties_map = get_dataset_properties_map()

    kwargs = {
        "desc": "Processing short datasets",
        "total": len(short_datasets),
        "unit": "dataset",
    }

    # `short_datasets` includes all datasets, even the ones that have medium-
    # and long-term versions
    rows = []
    for name in tqdm(short_datasets, **kwargs):
        terms = ["short", "medium", "long"] if name in med_long_datasets else ["short"]

        for term in terms:
            dataset = Dataset(name, term=term)
            ds_key = get_ds_key(name)

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

    df = pd.DataFrame(rows)
    print(f"Number of datasets: {len(df)}")

    root = Path(__file__).resolve().parents[1]
    metadata_dir = root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "gift_eval.csv"

    df.to_csv(metadata_path, index=False)
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
