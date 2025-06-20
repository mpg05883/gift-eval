import argparse


import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.gift_eval.data import Dataset



def main(args):
    dataset_directory = Path('datasets') / args.split
    names = [dir.name for dir in dataset_directory.iterdir() if dir.is_dir()]
    
    terms = ['short','medium', 'long']
    
    kwargs = {
        "desc": "Reading datasets",
        "total": len(names),
        "unit": "dataset",
    }
    
    print(f'Number of names: {len(names)}')
    print(f'Number of terms: {len(terms)}')
    print(f'Number of name-term combinations: {len(names) * len(terms)}')
    
    rows = []
    
    for name in tqdm(names, **kwargs):
        for term in terms:
            try:
                dataset = Dataset(name, term, verbose=False)
            except Exception as e:
                continue
            
            row = {
                "name": name,
                "term": term,
                "freq": dataset.freq,
                "prediction_length": dataset.prediction_length,
                "target_dim": dataset.target_dim,
                "windows": dataset.windows,
                "_min_series_length": dataset._min_series_length,
                "sum_series_length": dataset.sum_series_length,
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    
    csv_path = Path("resources") / f"{args.split}_info.csv"
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Gets dataset information and saves it to a CSV file"
    )
    argparser.add_argument(
        "--split",
        choices=["train_test", "pretrain"],
        default="pretrain",
        help="Split to use (train_test or pretrain)",
    )
    args = argparser.parse_args()
    main(args)
