import argparse
from PDMD import run, benchmark, calculate_MAE
from config import config


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Train or benchmark the PDMD module.")

    # Add a mutually exclusive group (you can run only one at a time)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the PDMD with the given configuration.")
    group.add_argument("--benchmark", action="store_true", help="Test the PDMD module.")
    group.add_argument("--mae", action="store_true", help="Calculate the MAE")

    # Parse the arguments
    args = parser.parse_args()

    # Run the corresponding function based on the argument
    if args.train:
        run(config)
    elif args.benchmark:
        benchmark()
    elif args.mae:
        calculate_MAE()


if __name__ == "__main__":
    main()
