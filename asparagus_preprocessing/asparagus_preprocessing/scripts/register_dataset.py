import argparse
from asparagus_preprocessing.utils.dataset_registration import register_dataset
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Register a preprocessed dataset for asparagus training without using asparagus_preprocessing.",
        epilog=(
            "Examples:\n"
            "  # Directory mode: auto-discover files and split\n"
            "  asp_register_dataset -t PT100_MyBrains -d /data/brains --extension .nii.gz\n\n"
            "  # Explicit mode: provide your own train/val lists\n"
            "  asp_register_dataset -t PT100_MyBrains --train /abs/f1.nii.gz /abs/f2.nii.gz --val /abs/f3.nii.gz\n\n"
            "  # Custom ratios (integers that sum to 100)\n"
            "  asp_register_dataset -t PT100_MyBrains -d /data/brains --train_ratio 99 --val_ratio 1 --test_ratio 0\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-t", "--task_name", required=True, help="Dataset/task name (e.g. PT100_MyBrains)")
    parser.add_argument("--n_classes", type=int, default=1, help="Number of output classes (default: 1)")
    parser.add_argument("--n_modalities", type=int, default=1, help="Number of input modalities (default: 1)")

    # Directory mode
    parser.add_argument("-d", "--data_dir", help="Directory containing preprocessed files (directory mode)")
    parser.add_argument("--extension", default=".nii.gz", help="File extension to search for (default: .nii.gz)")

    # Explicit mode
    parser.add_argument("--train", nargs="+", help="Training file paths (explicit mode)")
    parser.add_argument("--val", nargs="+", help="Validation file paths (explicit mode)")
    parser.add_argument("--test", nargs="+", help="Test file paths (optional)")

    # Split options
    parser.add_argument("--train_ratio", type=int, default=80, help="Train %% as integer 0-100 (default: 80)")
    parser.add_argument("--val_ratio", type=int, default=10, help="Val %% as integer 0-100 (default: 10)")
    parser.add_argument("--test_ratio", type=int, default=10, help="Test %% as integer 0-100 (default: 10)")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--split_name", default=None, help="Custom split name (default: auto-generated)")

    # Output
    parser.add_argument("-o", "--output_dir", help="Output directory (default: $ASPARAGUS_DATA/{task_name})")

    args = parser.parse_args()

    if args.data_dir is not None:
        ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
        if ratio_sum != 100:
            parser.error(
                f"Ratios must sum to 100, got {ratio_sum} ({args.train_ratio} + {args.val_ratio} + {args.test_ratio})"
            )

    register_dataset(
        task_name=args.task_name,
        n_classes=args.n_classes,
        n_modalities=args.n_modalities,
        data_dir=args.data_dir,
        extension=args.extension,
        train=args.train,
        val=args.val,
        test=args.test,
        train_ratio=args.train_ratio / 100,
        val_ratio=args.val_ratio / 100,
        test_ratio=args.test_ratio / 100,
        n_folds=args.n_folds,
        seed=args.seed,
        output_dir=args.output_dir,
        split_name=args.split_name,
    )


if __name__ == "__main__":
    main()
