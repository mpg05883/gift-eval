import pandas as pd
from pathlib import Path
from src.gift_eval.data import Dataset
from tqdm import tqdm

def main():
    file_path = Path("resources") / "pretrain" / "info.csv"
    df = pd.read_csv(file_path)    
    names = df["name"].unique()
    
    kwargs = {
        "desc": "Loading datasets",
        "total": len(names),
        "unit": "dataset",
    }
    
    failed = []
    
    for name in tqdm(names, **kwargs):
        try:
            Dataset(name, verbose=False).num_entries
        except Exception as e:
            failed.append(name)
        
    print(f"Finished loading all {len(names)} names")
    print(f"Failed to load {len(failed)} datasets")
    
    if failed:
        print(f"Failed to load the following datasets:")
        for name in failed:
            print(f"- {name}")
        
    


if __name__ == "__main__":
    main()