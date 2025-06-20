import argparse


import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.gift_eval.data import Dataset



def main(args):
    dataset_directory = Path('datasets') / args.split
    names = [dir.name for dir in dataset_directory.iterdir() if dir.is_dir()]
    print(f"Found {len(names)} datasets in {dataset_directory}")
    
    df = pd.read_csv("./resources/pretrain_info.csv")
    
    absent_names = set(names) - set(df['name'].tolist())
    print(f"Found {len(absent_names)} datasets not in pretrain_info.csv: {absent_names}")

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
