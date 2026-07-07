import logging
from asparagus_preprocessing.utils.parser import asparagus_parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = asparagus_parser
    parser.add_argument(
        "--dataset",
        help="Name of the dataset to preprocess. Should be of format: PT001_NAME or SEG001_NAME etc.",
        required=True,
    )
    args = parser.parse_args()
    module = find_module(args.dataset)
    module.main(
        processes=args.num_workers,
        bidsify=args.bidsify,
        save_dset_metadata=args.save_dset_metadata,
        save_as_tensor=args.save_as_tensor,
    )


def find_module(substring: str):
    import asparagus_preprocessing
    import importlib
    import os

    root = asparagus_preprocessing.__path__[0]
    for dir, subdirs, files in os.walk(root):
        for file in files:
            if (file.startswith(substring + "_") and file.endswith(".py")) or (file == substring + ".py"):
                module_path = f"asparagus_preprocessing.{dir.replace(root + '/', '')}.{file[:-3]}"
                return importlib.import_module(module_path)
    raise ValueError(f"Module with substring {substring} not found")


if __name__ == "__main__":
    main()
