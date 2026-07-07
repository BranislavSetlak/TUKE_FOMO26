from .detect import (
    find_and_add_test_splits,
    find_and_add_train_splits,
    find_processed_dataset,
)
from .loading import load_json
from .splitting import (
    BIDSsplit_40_10_50,
    MCSAsplit_40_10_50,
    PatientIDsplit_40_10_50,
    dynamic_split,
    split,
    split_40_10_50,
)

all_split_fn = [
    split_40_10_50.__name__,
    BIDSsplit_40_10_50.__name__,
    PatientIDsplit_40_10_50.__name__,
    MCSAsplit_40_10_50.__name__,
]
