import pandas as pd
from pathlib import Path

def main():
    directory = Path("results") / "chronos_bolt_base" / "pretrain"
    csv_paths = list(directory.rglob("*.csv"))
    
    empty_files = []
    
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            print(f"Empty DataFrame found in {path}")
            empty_files.append(path)
            
    print(f"Read {len(csv_paths)} CSV files.")
    print(f"Found {len(empty_files)} empty DataFrames.")
    
if __name__ == "__main__":
    main()
    from pathlib import Path
