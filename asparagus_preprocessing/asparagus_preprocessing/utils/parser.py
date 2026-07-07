import argparse


def get_asparagus_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=12, help="Number of processes to use.")
    parser.add_argument("--bidsify", action="store_true", help="Restructure dataset in BIDS format.")
    parser.add_argument("--save_dset_metadata", action="store_true", help="Save dataset level metadata.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--save_as_tensor",
        action="store_true",
        help="Save processed images as .pt tensors",
    )
    group.add_argument(
        "--save_as_nifti",
        action="store_true",
        help="Save processed images as .nii.gz files",
    )
    return parser


asparagus_parser = get_asparagus_parser()
