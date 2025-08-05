import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from gluonts.dataset.field_names import FieldName
from tqdm import tqdm

import datasets


def main():
    # Load dataset names
    with open(Path("datasets.txt"), "r") as f:
        names = f.read().splitlines()

    # Load metadata for all pretraining datasets
    metadata_path = Path("metadata") / "pretrain" / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata_dict = json.load(f)

    # Load root directory to datasets
    load_dotenv()
    root = os.getenv("GIFT_EVAL")

    # Store datasets whose HF dataset's target dim doesn't match the metadata
    malformed = []

    print(f"Found {len(names)} datasets in datasets.txt")

    kwargs = {
        "desc": "Checking target dimensions",
        "unit": "dataset",
        "total": len(names),
    }

    for name in tqdm(names, **kwargs):
        # Load HF dataset
        hf_dataset = datasets.load_from_disk(str(Path(root) / "pretrain") / name)

        # Get first time series in dataset and its target dimension
        entry = hf_dataset[0]
        target = entry[FieldName.TARGET]
        hf_target_dim = target.shape[0] if len(target.shape) > 1 else 1

        # Load metadata
        metadata = metadata_dict[name]

        # Get target dimension from metadata
        metadata_target_dim = metadata["target_dim"]

        # Check if target dimensions match
        if hf_target_dim != metadata_target_dim:
            malformed.append(
                {
                    "name": name,
                    "hf_target_dim": hf_target_dim,
                    "metadata_target_dim": metadata_target_dim,
                }
            )

    # Print malformed datasets
    if not malformed:
        print("All datasets have matching target dimensions.")
        sys.exit()

    print(f"Found {len(malformed)} malformed datasets:")
    for entry in malformed:
        print(
            f"Dataset {entry['name']} has HF target dim {entry['hf_target_dim']} "
            f"but metadata target dim {entry['metadata_target_dim']}."
        )


if __name__ == "__main__":
    main()
