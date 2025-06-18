import argparse
from typing import Literal

from src.gift_eval.data import Dataset


def main(args):
    Dataset(name=args.name, term=args.term)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Loads a pretraining dataset with a given name and term
        to check if the name-term pair is valid.""",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="m4_hourly",
        help="Name of the dataset to instantiate.",
    )
    parser.add_argument(
        "--term",
        type=str,
        default=Literal["short", "medium", "long"],
        help="Term of the dataset to instantiate.",
    )
    args = parser.parse_args()
    main(args)
