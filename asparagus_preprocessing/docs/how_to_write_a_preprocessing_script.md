## Purpose
The purpose of the preprocessing script is to take an unstructured, unprocessed dataset and process it and save the outputs in a structured format that is designed to make subsequent model training seamless. The end product should be a dataset directory containing data and metadata files (see the [Example Directory Structure](./docs/example_directory_structure.md)). The ```dataset.json``` and ```paths.json``` must be in the root dataset directory, but the data itself can be in arbitrary subdirectories with arbitrary file names as long as the extension is .pt the ```paths.json``` in the root dataset directory reflects these paths in absolute terms.

## Formal Requirements:
### Req. 1:
Datasets names must be prefixed with an ID consiting of the task type, 3 digits and an underscore, such as "SEG785_". Everything that comes after the underscore is up to you. This ID is used to locate data in later stages. Valid task types are PT (Pretraining), CLS (classification), SEG (segmentation), and REG (regression). The three digits must be unique _for the task type_. I.e. having both PT785 and CLS785 is valid.
### Req. 2:
Processed data must be saved in the `ASPARAGUS_DATA` directory in a subdirectory named after the dataset ID, see [here](./docs/example_directory_structure.md) for an example. The `ASPARAGUS_DATA` variable can be accessed in the code using the `get_data_path()` function found [here](asparagus_preprocessing/paths.py).
### Req. 3:
Unprocessed (raw) labels must also be saved at `ASPARAGUS_RAW_LABELS` which can be accessed using the `get_raw_labels_path()` function found [here](asparagus_preprocessing/paths.py).
#### 3.1 (Classificatio and Regression):
All labels are stored in a table automatically when using the ```save_clsreg_data_and_metadata``` function using the ```label``` and ```table_out``` parameters. 
#### 3.2 (Segmentation):
To save segmentation labels you can use either the `save_raw_label` or `save_modified_label` functions from `asparagus_preprocessing.utils.saving`. If you do not re-format the labels use the `save_raw_label` function. If you re-format the label map (e.g. collapse two labels to one), please store the formatted (but otherwise unpreprocessed) label map using `save_modified_label` function. See [SEG005_IvyGap](asparagus_preprocessing/datasets_segmentation/SEG005_IvyGap.py) for an example. If you are in doubt use `save_modified_label`.
### Req. 4: 
After processing all files the newly generated dataset directory must be postprocessed to generate two metadata json files: dataset.json and paths.json. The dataset.json must include the number of modalities and the number of labels. This is used to configure the input and output channels of networks. The paths.json must include the paths to all processed samples in the dataset. These paths are used to generate train, val and test splits. Both files can be generated using the `postprocess_standand_dataset` function found [here](asparagus_preprocessing/utils/metadata_generation.py)
### Req. 5 (segmentation only): 
For segmentation: a metadata dictionary must be saved for each image as a pickle. The path should be identical, except for the extension (i.e. PATH_XYZ.pkl instead of PATH_XYZ.pt). The dict must contain the key "foreground_locations" pointing to a list that is of indices of non-zero labels in the ground truth (the list can be left empty). If not empty these indices are used to oversample underrepresented classes. We also save these metadata files for other task types but they are not strictly required.

## Step-by-step guide
### Step 0: Define main
Set the base path using the get_source_path() which uses the ```ASPARAGUS_SOURCE``` environment variable and the dataset specific subdir inside the base path. Additionally set default processes, whether to bidsify and save dataset metadata (if implemented), and whether to save as a tensor or a nifti.
```
def main(
    path: str = get_source_path(),
    subdir: str = "SBM-Stanford-Brain-Metastasis-3",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=True,
):
```
### Step 1: Config Setup
Define ```DatasetConfig```, ```SavingConfig``` and ```PreprocessingConfig``` objects. 
#### 1.1 DatasetConfig. 
The dataset config can be seen below. From top to bottom it defines 
- df_columns: which dataframe columns to keep for datasets saving metadata tables.
- task_name: The unique name of your task/dataset, consisting of a name living up to [Req. 1](#req-1)
- n_classes: The number of unique classes for the dataset. For regression this is 1.
- n_modalities: The number of unique channels/modalities for the dataset. If you treat CT/MRI as separate images this would be 1 but if they're concatenated in the C dimension (resulting in a dimension size of 2) of a [C, H, W, D] datapoint, this would be 2.
- in_extensions: The valid extensions to search for in the input folder. Only files with these extensions are included in the dataset.  
- patterns_exclusion: Patterns used to exclude files that would otherwise be included (i.e. have to correct extensions). Useful to exclude labels or modalities intended to be concatenated in order to not process them multiple times. See [SEG002_SBM](asparagus_preprocessing/datasets_segmentation/SEG002_SBM.py) for an example of this, where all sequences and labels are excluded except the "bravo" sequence, which is then used to load corresponding modalities and label.
- patters_DWI: Patterns to separate DWI files from other files, useful for sequence specific processing.
- patters_PET: Patterns to separate PET files from other files, useful for sequence specific processing.
- patters_DWI: Patterns to separate perfusion files from other files, useful for sequence specific processing.
- patters_DWI: Patterns to separate m0-perfusion files from other files, useful for sequence specific processing.
- patterns_bidsify: Patterns used to extract BIDS-relevant information to restructure dataset into BIDS format.
- split: The default split to be run with the preprocessing. The splitting function should split on a subject level, so if subjects have multiple datapoints the splitting function must use a regex to identify the unique subjects and split on this level rather than on file level. 
```
@dataclass
class DatasetConfig:
    df_columns: list 
    task_name: str
    n_classes: int
    n_modalities: int
    in_extensions: str
    patterns_exclusion: list
    patterns_DWI: list
    patterns_PET: list
    patterns_perfusion: list
    patterns_m0: list
    patterns_bidsify: list
    split: str
```
#### 1.2 SavingConfig
The SavingConfig can be seen below. From top to bottom it defines 
- save_as_tensor: whether to save as tensor or nifti file.
- tensor_dtype: torch dtype used if save_as_tensor is true
- bidsify: whether to format the dataset in the BIDS-format. Requires dataset specific code for this. Usually only relevant for select pretraining datasets.
- save_dset_metadata: whether to save additional dataset metadata. Requires dataset specific code for this. Usually only relevant for select pretraining datasets.
- save_file_metadata: whether to save file metadata in a .pkl file. For Segmentation tasks this must be true.
```
@dataclass
class SavingConfig:
    save_as_tensor: bool
    tensor_dtype: str
    bidsify: bool
    save_dset_metadata: bool
    save_file_metadata: bool
```

#### 1.3 PreprocessingConfig
The PreprocessingConfig can be seen below. From top to bottom it defines
- normalization_operation: A list of normalization operations with length equal to the number of modalities, where normalization operation at index 0 will be applied to the channel/modality at index 0 in the arrays.
- target_spacing: The spacing to resample data to. 
- background_pixel_value: The pixel value used to remove background if crop_to_nonzero is True.
- crop_to_nonzero: Whether to crop to smallest nonzero (or non-background_pixel_value) square.
- keep_aspect_ratio_when_using_target_size: Whether to stretch data to target size or zero-fill after 1 dimension is equal to its target size.
- intensities: list of intensities for the dataset, which can be used for some normalization operations using dataset-wide statistics.
- target_orientation: Which orientation the training data should be in for medical images. 
- target_size: Specific size all data will be resampled to.
- min_slices: filter used to discard data with any dimension less than.
- remove_nans: whether to remove data where nans are found. 
```
@dataclass
class PreprocessingConfig:
    normalization_operation: List
    target_spacing: Optional[List]
    background_pixel_value: int = 0
    crop_to_nonzero: bool = True
    keep_aspect_ratio_when_using_target_size: bool = False
    image_properties: Optional[dict] = field(default_factory=dict)
    intensities: Optional[List] = None
    target_orientation: Optional[str] = "RAS"
    target_size: Optional[List] = None
    min_slices: int = 0
    remove_nans: bool = True
```

### Step 2: Set up paths: 

Using the asparagus environment variables and the path and subdir defined in Step 0 set up where to find (source) the dataset and where to save (target) it.
```
source_dir = os.path.join(path, subdir)
target_dir = os.path.join(get_data_path(), dataset_config.task_name)
prepare_target_dir(target_dir, saving_config.save_as_tensor)
```
For some datasets it may be necessary to look into a table to include/exclude files and for classification/regression tasks a table of labels must also be saved. These two cases are handled as below:
```
table_in = pd.read_csv(os.path.join(source_dir, "swu_slim_phenodata_time1.tsv"), encoding="unicode_escape", sep="\t")

table_out = os.path.join(get_raw_labels_path(), dataset_config.task_name, "labels.csv")
```

### Step 3: Find files:

If you've followed the Asparagus_preprocessing format so far all your training files can be automatically found using the ```recursive_find_and_group_files``` function. This will locate all files with the correct extensions in your dataset directory, exclude files matching the exclusion patterns, and stratify into standard, dwi, pet and perfusion files if any patterns are included. Anything not matched by any pattern but with a valid extension is considered "standard".

After finding all files we create output paths for each of them by substituting the part of the path matching the source_dir with the target_dir defined in Step 2. If 
```
files_standard, files_DWI, files_PET, files_Perf, files_excluded = recursive_find_and_group_files(
    source_dir,
    extensions=dataset_config.in_extensions,
    patterns_dwi=dataset_config.patterns_DWI,
    patterns_pet=dataset_config.patterns_PET,
    patterns_perfusion=dataset_config.patterns_perfusion,
    patterns_exclusion=dataset_config.patterns_exclusion,
    processes=processes,
)

files_standard_out = get_image_output_paths(files_standard, source_dir, target_dir, dataset_config.in_extensions)
```

If you are using any of the stratification patterns (excluding the exclusion patterns) you will also need to adapt the paths of the stratified files as such:
```
files_DWI_out = get_image_output_paths(files_DWI, source_dir, target_dir, dataset_config.in_extensions)
files_PET_out = get_image_output_paths(files_PET, source_dir, target_dir, dataset_config.in_extensions)
files_Perf_out = get_image_output_paths(files_Perf, source_dir, target_dir, dataset_config.in_extensions)
```

### Step 4: Process the dataset.
Below are outlined the most common ways to process each type of dataset. This appropriate function can vary.

#### 4.1 Pretraining:
Pretraining often involves different file types (standard MRI, DWI, PET, Perfusion etc.,) that may need to be processed individually. For this we use the ```multiprocess_mri_dwi_pet_perf_cases``` function. Under the hood this processes each modality with a modality-specific script, and then returns them in a compatible format. E.g. DWI is often 4D and needs to become 3D to be compatible with other modalities.
```
multiprocess_mri_dwi_pet_perf_cases(
    files_standard=files_standard,
    files_standard_out=files_standard_out,
    files_DWI=files_DWI,
    bvals_DWI=bvals_DWI,
    bvecs_DWI=bvecs_DWI,
    files_DWI_out=files_DWI_out,
    files_PET=files_PET,
    files_PET_out=files_PET_out,
    files_Perf=files_Perf,
    files_Perf_out=files_Perf_out,
    patterns_m0=dataset_config.patterns_m0,
    preprocessing_config=preprocessing_config,
    saving_config=saving_config,
    processes=processes,
    strict=False,
)
```

#### 4.2 Segmentation:
In segmentation we rarely need to stratify data and rarely rely on tables for exclusion/inclusion. Therefore we can use the standard ```process_dataset_without_table``` with a ```process_sample``` function to process the files found in Step 3. according to the PreprocessingConfig and save them according to the SavingConfig. 
```
   process_dataset_without_table(
        process_fn=process_sample,
        files_in=files_standard,
        files_out=files_standard_out,
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        processes=processes,
    )
```

The ```process_sample``` function not using tables can be reduced to the following, but can also be adapted:

```
def process_sample(file_in, file_out, dataset_config, preprocess_config, saving_config):
    if (os.path.exists(file_out + ".pt") and saving_config.save_as_tensor) or (
        os.path.exists(file_out + ".nii.gz") and not saving_config.save_as_tensor
    ):
        return

    bravo_path = file_in
    label_path = bravo_path.replace("bravo.nii.gz", "seg.nii.gz")

    bravo = nib.load(bravo_path)
    label_raw = nib.load(label_path)

    image, label, image_props = preprocess_case_with_label(
        images=[bravo],
        label=label_raw,
        **asdict(preprocess_config),
        strict=False,
    )

    image_props["src_label_path"] = label_path

    save_data_and_metadata(
        data=image + [label],
        metadata=image_props,
        data_path=file_out,
        saving_config=saving_config,
    )

    save_raw_label(file_out, label_path)
```

#### 4.3 Regression/Classification
Regression or Classification often relies on a table to extract target (label) values and possibly exclude datapoints with missing values. For this we use the ```process_dataset_with_table``` function which operates like the function used in 4.2, but with an included table and a path to save the table with raw labels.

```
process_dataset_with_table(
    process_fn=process_sample,
    files_in=files_standard,
    files_out=files_standard_out,
    dataset_config=dataset_config,
    preprocessing_config=preprocessing_config,
    saving_config=saving_config,
    table_in=table_in,
    table_out=table_out,
    processes=processes,
)
```

The process_sample function using tables can be reduced to the following:
```
def process_sample(file_in, file_out, dataset_config, preprocessing_config, saving_config, table_in, table_out):
    if (os.path.exists(file_out + ".pt") and saving_config.save_as_tensor) or (
        os.path.exists(file_out + ".nii.gz") and not saving_config.save_as_tensor
    ):
        return

    MRI = nib.load(file_in)
    img_id = os.path.split(file_in)[-1].split("_")[0]

    subject_id = table_in[table_in["Image Data ID"] == img_id]["Subject"].item()
    label = table_in[table_in["Match_ID"] == subject_id]["Age"]
    
    image, image_props = preprocess_case_without_label(images=[MRI], **asdict(preprocessing_config), strict=False)
    save_clsreg_data_and_metadata(
        data=image,
        label=label,
        metadata=image_props,
        data_path=file_out,
        saving_config=saving_config,
        table_path=table_out,
    )
```

### Step 5: Postprocess

If you've used the asparagus_processing pipeline so far the postprocessing can be handled by ```postprocess_standard_dataset``` using the Configs and variables you've already set up. If you haven't created any e.g. files_DWI variable an empty list can be passed, and this is true for many of the less critical parts.

```
postprocess_standard_dataset(
    dataset_config=dataset_config,
    preprocessing_config=preprocessing_config,
    saving_config=saving_config,
    target_dir=target_dir,
    source_files_standard=files_standard,
    source_files_DWI=files_DWI,
    source_files_PET=files_PET,
    source_files_Perf=files_Perf,
    source_files_excluded=files_excluded,
    processes=processes,
)
```

