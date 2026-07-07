import logging
import re
from asparagus_preprocessing.utils.saving import enhanced_save_json
from sklearn.model_selection import train_test_split


def split_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    return non_stratified_split(files, 0.40, 0.10, 0.50, test, seed_increment, base_seed=28300211)


def split_80_10_10(files: list, test: bool = False, seed_increment: int = 0):
    return non_stratified_split(files, 0.80, 0.10, 0.10, test, seed_increment, base_seed=28300211)


def BIDSsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"/sub-\d+/"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def ABVIBsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"ABVIB/\d+/"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def PatientIDsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"/PatientID_[0-9]+/"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def MCSAsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"/MCSA_\d+/"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def UCSDsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"/UCSD-PTGBM-\d+_"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def non_stratified_split(
    files: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    test: bool = False,
    seed_increment: int = 0,
    base_seed: int = 0,
):
    assert train_ratio + val_ratio + test_ratio == 1.0, (
        "Train, validation, and test ratios must sum to 1, but got {}, {}, {}".format(train_ratio, val_ratio, test_ratio)
    )

    files = sorted(files)

    if test:
        if test_ratio == 0.0:
            return files, []
        return train_test_split(files, test_size=test_ratio, random_state=base_seed)

    return train_test_split(
        files,
        test_size=val_ratio / (val_ratio + train_ratio),
        random_state=base_seed + seed_increment + 1,
    )


def stratified_split(
    files: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    test: bool,
    pattern: str,
    seed_increment: int,
    base_seed: int,
):
    assert train_ratio + val_ratio + test_ratio == 1.0, (
        "Train, validation, and test ratios must sum to 1, but got {}, {}, {}".format(train_ratio, val_ratio, test_ratio)
    )

    subjects = []
    for i in files:
        subjects.append(re.findall(pattern, i)[0])

    subjects = sorted(list(set(subjects)))

    if test:
        train_subs, test_subs = train_test_split(subjects, test_size=test_ratio, random_state=base_seed)
        train_sub_files = [file for file in files if any(tr_sub in file for tr_sub in train_subs)]
        test_sub_files = [file for file in files if any(test_sub in file for test_sub in test_subs)]
        return train_sub_files, test_sub_files

    train_subs, val_subs = train_test_split(
        subjects,
        test_size=val_ratio / (val_ratio + train_ratio),
        random_state=base_seed + seed_increment,
    )
    train_sub_files = [file for file in files if any(tr_sub in file for tr_sub in train_subs)]
    val_sub_files = [file for file in files if any(val_sub in file for val_sub in val_subs)]
    return train_sub_files, val_sub_files


def split(files: list, fn: callable, folds=5, save_path: str = None, split_pattern: str = None):
    if len(re.findall(r"/sub-\d+/", files[0])) > 0:
        logging.warning(
            "BIDS format detected. Consider switching to BIDSsplit_XXX to avoid data leakage (same subject in train/val/test) \
                if you are not already doing it."
        )

    splits_trval = []
    trainval, test = fn(files, test=True)

    for i in range(folds):
        train, val = fn(trainval, test=False, seed_increment=i)
        splits_trval.append({"train": train, "val": val})
    if save_path is not None:
        enhanced_save_json(obj=splits_trval, file=save_path)
        enhanced_save_json(obj=test, file=save_path.replace("split_", "TEST_"))
    return train, val, test


def dynamic_split(
    files: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    folds: int = 5,
    save_path: str = None,
):
    splits_trval = []
    trainval, test = non_stratified_split(files, train_ratio, val_ratio, test_ratio, test=True)

    for i in range(folds):
        train, val = non_stratified_split(trainval, train_ratio, val_ratio, test_ratio, test=False, seed_increment=i)
        splits_trval.append({"train": train, "val": val})

    if save_path is not None:
        enhanced_save_json(obj=splits_trval, file=save_path)
        enhanced_save_json(obj=test, file=save_path.replace("split_", "TEST_"))

    return train, val, test
