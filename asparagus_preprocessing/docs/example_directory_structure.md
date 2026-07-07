# Task Conversion Guide

## Concept
Asparagus Task Conversion is designed to handle large datasets. With large datasets we want to avoid storing intermediate version of the dataset, and therefore all formatting, restructuring and preprocessing should be done in 1 step. However, as it is a tedious process to redo this only essential and constant processing should be done here. E.g. normalization operations change more frequently and are therefore applied online during.

## Example Folder Structure 
```
processed_data/
├── SEG785_MyAlpacaCollection/
|   ├── dataset.json
|   ├── paths.json
|   ├── split_75_15_10.json
|   ├── TEST_75_15_10.json
|   ├── alpaca_images_from2019/
|   |   ├── alpaca_05.pt
|   |   ├── alpaca_05.pkl
|   |   ├── maybe_llama_03.pt
|   |   ├── maybe_llama_03.pkl
|   |   ├── ...
|   ├── alpaca_images_from2022/
|   |   ├── definitely_alpaca_01.pt
|   |   ├── definitely_alpaca_01.pkl
|   |   ├── guanaco_03142.pt
|   |   ├── guanaco_03142.pkl
├── CLS962_FavouriteOysters/
...
```


