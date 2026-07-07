# Data preparation and preprocessing

# Resources
- [Environment Variables](./docs/environment_variables.md)
- [Example Directory Structure](./docs/example_directory_structure.md)
- [How to write an Asparagus Preprocessing script](./docs/how_to_write_a_preprocessing_script.md)

## Introduction
Asparagus Preprocessing is a combined formatting and preprocessing process. If a script already exists for your dataset go to [running scripts](#running-pre-made-scripts). If a script does not exist for your dataset *YOU* must write a script that takes your raw data and puts it in the place that Asparagus expects and in the structure that Asparagus expects. See [How to write an Asparagus Preprocessing script](./docs/how_to_write_a_preprocessing_script.md) for a guide on how to do this.

## 1. Setup

### 1.1 Installation
To install asparagus_preprocessing using git and pip:
```
git clone https://github.com/Sllambias/asparagus_preprocessing.git
cd asparagus_preprocessing
pip install .
```

## 1.2 Environment Variables
Asparagus Preprocessing relies on environment variables to locate unformatted and unprocessed datasets and know where to save formatted and processed datasets and copies of the raw labels. For a detailed description of the variables and their purpose see [Environment Variables](./docs/environment_variables.md). The required environment variables are ```ASPARAGUS_DATA```, ```ASPARAGUS_SOURCE``` and ```ASPARAGUS_RAW_LABELS```.

To set environment variables create a ".env" file inside asparagus_preprocessing with the following content (where you adapt the paths appropriately):
```
ASPARAGUS_DATA=/path/to/where/to/save/processed/datasets
ASPARAGUS_SOURCE=/path/to/where/source/datasets/are
ASPARAGUS_RAW_LABELS=/path/to/where/to/save/raw/labels
```

## 2. Writing the preprocessing script.
See [How to write an Asparagus Preprocessing script](./docs/how_to_write_a_preprocessing_script.md)

## 3. Running the preprocessing script.
Whether you are using a pre-made script or have just created your own, the next step is to run it.
To run an asparagus preprocessing script you specify the :

1. Install asparagus_preprocessing
2. Run ```asp_process --dataset PT001 --save_as_tensor```

To edit the path of an existing preprocessing script:

```python
if __name__ == "__main__":
    args = asparagus_parser.parse_args()
    root_path = "/path/to/my/local/datasets/dir"
    data_dir = "my_penguin_dataset"
    main(
        path = root_path,
        subdir = data_dir,
        processes=args.num_workers,
        bidsify=args.bidsify,
        save_metadata=args.save_metadata,
        save_as_tensor=args.save_as_tensor,
    )
```

## 4. Creating split files

A dataset specific default split is always created during step 3., but if you need to create additional splits the ```asp_split``` command can do this.

To create split files using the ```split_40_10_50``` function for a task called PT123_DrainagePipes:
```
asp_split --dataset PT123 --fn split_40_10_50
```
And `fn` can be any of the functions defined in asparagus_preprocessing/utils/splitting.py.
If you just want to do a basic train/val/test split, you can provide the split ratios as
```
asp_split --dataset PTXXX --vals 80 10 10
```
for an 80/10/10 split. 

### 4.1 Updating split file paths

Sometimes data is moved to a new cluster / scratch etc, but we want to reuse the same split files. To recompute the file names on the new destination use
```
asp_update_paths PTXXX

or

asp_update_paths --all
```

If you want to update the path on scratch, change the environment variable of the data dir and try again
```
ASPARAGUS_DATA=/scratch/FOMO300K asp_update_paths PTXXX
```
but please note this will delete the original split file, so you have to manually copy the old one if you want to keep both (which you probably do). 
