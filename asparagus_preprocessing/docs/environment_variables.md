### ASPARAGUS_DATA: 
This specifies where to save *FORMATTED* and *PROCESSED* data. For datasets with labels the stored data is volume+label pairs, and for datasets without labels the stored data is volumes alone. For pretraining the data will be stored as a tensor of shape [C, H, W, D] where C corresponds to number of channels/modalities. For segmentation the data will be stored as a tensor of shape [C+1, H, W, D] where C+1 corresponds to the number of channels/modalities AND the segmentation map - this means the spatial dimensions of the segmentation map must be identical to the spatial dimensions of the images. The segmentation is *always* the last "channel". For classification and regression the data will be stored as a list of length 2 [data, label] where label is a tensor of shape [C, H, W, D] and label is a tensor of shape [n_classes]. For 2D data the last dimension (D) is omitted for all task types.

### ASPARAGUS_SOURCE: 
This specifies the directory where asparagus can find *UNFORMATTED* and *UNPROCESSED* datasets to (pre)process. The individual preprocessing scripts should use ASPARAGUS_SOURCE to construct paths pointing to the dataset in question, so that preprocessing scripts can be reused across users and environments. For example, if your ASPARAGUS_SOURCE directory is "/home/users/MyName/datasets" and the dataset in question is located at "/home/users/MyName/datasets/MyLlamaDataset" the full path should be constructed using:
```
import os
from asparagus_preprocessing.paths import get_source_path

path = os.path.join(get_source_path(), "MyLlamaDataset")
```

### ASPARAGUS_RAW_LABELS: 
Specifies where to save copies of the *FORMATTED* but *UNPROCESSED* (raw) labels. During training we employ potentially mildly manipulated labels for loss and metric calculations, as the alternative would be computationally inefficient. For example the segmentation map may be reoriented or resampled. However, the final test results should *always* be obtained using the untampered labels - but, they should still be formatted to fit the task. For example, if the dataset includes (1) label resampling to obtain a certain resolution and (2) label remapping from labels (0,1,4,7) to (0,1,2,3) ONLY the formatting/label-remapping should be done on the raw labels.