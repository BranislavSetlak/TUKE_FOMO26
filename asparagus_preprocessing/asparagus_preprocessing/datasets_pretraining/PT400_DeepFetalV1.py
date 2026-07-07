import logging
import polars as pl
from asparagus_preprocessing.utils.dataset_registration import register_dataset
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.splitting import dynamic_split


def main(
    path: str = "/projects/users/people/seblla/repos/EHR_extract/all_images_2026-05-29.csv",
    # path: str = "/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/test_table.csv",
    subdir: str = "",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=False,
):
    logging.warning("Loading DF")
    df = pl.read_csv(path)
    df = df.select(["phair_hash", "no_ocr_preprocessed_file_path"]).drop_nulls()
    subjects = df["phair_hash"].to_list()

    logging.warning("Splitting on subject level")
    train_subjects, val_subjects, test_subjects = dynamic_split(
        files=subjects, train_ratio=0.9, val_ratio=0.01, test_ratio=0.09, folds=1
    )

    logging.warning("Filtering image paths on into train,val,test on subject splits")
    train_df = df.filter(pl.col("phair_hash").is_in(train_subjects))
    train_files = train_df["no_ocr_preprocessed_file_path"].to_list()

    val_df = df.filter(pl.col("phair_hash").is_in(val_subjects))
    val_files = val_df["no_ocr_preprocessed_file_path"].to_list()

    test_df = df.filter(pl.col("phair_hash").is_in(test_subjects))
    test_files = test_df["no_ocr_preprocessed_file_path"].to_list()

    logging.warning("Saving")
    register_dataset(
        task_name="PT400_DeepFetalV1",
        n_classes=1,
        n_modalities=1,
        train=train_files,
        val=val_files,
        test=test_files,
        n_folds=1,
    )


if __name__ == "__main__":
    args = asparagus_parser.parse_args()
    main(
        processes=args.num_workers,
        bidsify=args.bidsify,
        save_dset_metadata=args.save_dset_metadata,
        save_as_tensor=args.save_as_tensor,
    )
