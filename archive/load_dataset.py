import argparse

from src.gift_eval.data import Dataset


def main(args):
    dataset = Dataset(name=args.name, term=args.term, verbose=False)
    training_dataset = dataset.training_dataset
    validation_dataset = dataset.validation_dataset
    test_data = dataset.test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads a dataset to make sure it can be loaded correctly."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the dataset to load",
    )
    parser.add_argument(
        "--term",
        choices=["short", "medium", "long"],
        default="short",
        help="Specifies the dataset's prediction length",
    )
    args = parser.parse_args()
    main(args)
