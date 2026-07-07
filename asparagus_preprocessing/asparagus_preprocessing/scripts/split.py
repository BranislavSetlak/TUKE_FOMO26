import argparse
import asparagus_preprocessing
import os
from asparagus_preprocessing.paths import get_data_path
from asparagus_preprocessing.utils import dynamic_split, load_json, split


def main():
    parser = argparse.ArgumentParser(
        "Parser for dataset splitting. \n "
        "EITHER use --fn to use a predefined function OR --vals to do an UNSTRATIFIED split using \
            specified ratios for TRAIN/VAL/TEST. \n"
        "e.g. --vals 40 10 50 for 40% training, 10% validation, 50% testing."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Name of the dataset to preprocess. Should be of format: PT123_NAME, SEG123_NAME, CLS123_NAME, REG123_NAME",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fn", help="Splitting method to use.")
    group.add_argument("--vals", nargs="*", type=int, help="Values for splitting function.")
    args = parser.parse_args()

    files = load_json(os.path.join(get_data_path(), args.dataset, "paths.json"))
    if args.fn:
        save_path = os.path.join(get_data_path(), args.dataset, args.fn + ".json")
        split(
            files=files,
            fn=getattr(asparagus_preprocessing.utils.splitting, args.fn),
            save_path=save_path,
        )
    elif args.vals:
        if sum(args.vals) != 100 or len(args.vals) != 3:
            raise ValueError("Please provide three integers that sum to 100 for TRAIN/VAL/TEST split.")
        save_path = os.path.join(
            get_data_path(),
            args.dataset,
            f"split_{args.vals[0]:02d}_{args.vals[1]:02d}_{args.vals[2]:02d}" + ".json",
        )
        dynamic_split(
            files=files,
            train_ratio=args.vals[0] / 100,
            val_ratio=args.vals[1] / 100,
            test_ratio=args.vals[2] / 100,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
